#!/usr/bin/env julia

"""
Compute kinetic helicity h = u · (∇×u) from each merged NetCDF file in a time range
and save the time-averaged helicity field to a compressed JLD2 file.

Assumptions:
- Inputs are merged global NetCDF outputs (contain spectral velocity coefficients and coordinates).
- Velocity is provided in spectral toroidal/poloidal form: velocity_toroidal_{real,imag}, velocity_poloidal_{real,imag}

Usage:
  julia --project=. script/time_average_helicity.jl <output_dir> --start=<t0> --end=<t1> [--prefix=<name>] [--out=<file.jld2>]

Saves:
  - helicity: time-averaged helicity field (nlat, nlon, nr)
  - counts: number of snapshots accumulated
  - times: times used for averaging
  - metadata: Dict with basic info (geometry if available)
  - coords: Dict with theta, phi, r (if available)
"""

using Printf
using NetCDF
using JLD2
using SHTnsKit

function usage()
    println("Usage: time_average_helicity.jl <output_dir> --start=<t0> --end=<t1> [--prefix=name] [--out=path.jld2]")
end

function parse_args(args)
    isempty(args) && (usage(); error("missing arguments"))
    outdir = abspath(args[1])
    t0 = nothing
    t1 = nothing
    prefix = "combined_global"
    outpath = ""
    for a in args[2:end]
        if startswith(a, "--start="); t0 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--end="); t1 = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--prefix="); prefix = split(a, "=", limit=2)[2]
        elseif startswith(a, "--out="); outpath = split(a, "=", limit=2)[2]
        else
            @warn "Unknown argument $a (ignored)"
        end
    end
    (t0 === nothing || t1 === nothing) && (usage(); error("--start and --end required"))
    return outdir, t0, t1, prefix, outpath
end

format_time_str(t::Float64) = replace(@sprintf("%.6f", t), "." => "p")

function scan_times(dir::String, prefix::String)
    files = filter(f -> endswith(f, ".nc") && occursin(prefix, f), readdir(dir))
    times = Float64[]
    pat = r"time_(\d+p\d+)"
    for f in files
        if (m = match(pat, f)) !== nothing
            push!(times, parse(Float64, replace(m.captures[1], "p"=>".")))
        end
    end
    return sort(unique(times))
end

function build_filename(dir::String, prefix::String, t::Float64)
    ts = format_time_str(t)
    cands = [
        joinpath(dir, "$(prefix)_time_$(ts).nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts)_rank_0000.nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts).nc"),
    ]
    for c in cands
        if isfile(c); return c; end
    end
    for f in readdir(dir)
        full = joinpath(dir, f)
        if endswith(f, ".nc") && occursin("time_$(ts)", f) && occursin(prefix, f)
            return full
        end
    end
    return ""
end

read_var(nc, name) = (NetCDF.varid(nc, name) == -1 ? nothing : NetCDF.readvar(nc, name))

function build_sht_from_nc(nc)
    lvals = Int.(read_var(nc, "l_values"))
    mvals = Int.(read_var(nc, "m_values"))
    lmax = maximum(lvals); mmax = maximum(mvals)
    nlat = (NetCDF.varid(nc, "theta") != -1) ? length(read_var(nc, "theta")) : (lmax+2)
    nlon = (NetCDF.varid(nc, "phi") != -1) ? length(read_var(nc, "phi")) : max(2*lmax+1, 4)
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    θ = read_var(nc, "theta"); φ = read_var(nc, "phi"); r = read_var(nc, "r")
    return cfg, lvals, mvals, (θ, φ, r)
end

function vector_components(cfg::SHTnsKit.SHTConfig, lvals, mvals,
                           tor_r, tor_i, pol_r, pol_i, rvec)
    lmax=cfg.lmax; mmax=cfg.mmax; nlat=cfg.nlat; nlon=cfg.nlon
    nlm, nr = size(tor_r)
    vt = Array{Float64}(undef, nlat, nlon, nr)
    vp = Array{Float64}(undef, nlat, nlon, nr)
    vr = Array{Float64}(undef, nlat, nlon, nr)
    tor = zeros(ComplexF64, lmax+1, mmax+1)
    pol = zeros(ComplexF64, lmax+1, mmax+1)
    for k in 1:nr
        fill!(tor, 0); fill!(pol, 0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                tor[l+1,m+1]=complex(tor_r[i,k], tor_i[i,k])
                pol[l+1,m+1]=complex(pol_r[i,k], pol_i[i,k])
            end
        end
        vt_slice, vp_slice = SHTnsKit.SHsphtor_to_spat(cfg, pol, tor; real_output=true)
        vt[:,:,k] = vt_slice; vp[:,:,k] = vp_slice
        if rvec !== nothing
            rr = rvec[min(k, length(rvec))]
            if rr > eps()
                pol_rad = zeros(ComplexF64, lmax+1, mmax+1)
                for i in 1:nlm
                    l=lvals[i]; m=mvals[i]
                    if l<=lmax && m<=mmax
                        pol_rad[l+1,m+1] = pol[l+1,m+1] * (l*(l+1)/rr)
                    end
                end
                vr[:,:,k] = SHTnsKit.synthesis(cfg, pol_rad; real_output=true)
            else
                vr[:,:,k] .= 0
            end
        else
            vr[:,:,k] .= 0
        end
    end
    return vr, vt, vp
end

function central_diff_φ(A, φ)
    # periodic
    nlat, nlon, nr = size(A)
    dA = similar(A)
    Δφ = φ[2] - φ[1]  # assume uniform
    @inbounds for k in 1:nr, i in 1:nlat
        @views row = A[i, :, k]
        for j in 1:nlon
            jp = (j % nlon) + 1
            jm = (j-2) % nlon + 1
            dA[i,j,k] = (row[jp] - row[jm]) / (2Δφ)
        end
    end
    return dA
end

function central_diff_θ(A, θ)
    nlat, nlon, nr = size(A)
    dA = similar(A)
    @inbounds for k in 1:nr, j in 1:nlon
        for i in 1:nlat
            if i == 1
                dA[i,j,k] = (A[i+1,j,k] - A[i,j,k]) / (θ[i+1] - θ[i])
            elseif i == nlat
                dA[i,j,k] = (A[i,j,k] - A[i-1,j,k]) / (θ[i] - θ[i-1])
            else
                dA[i,j,k] = (A[i+1,j,k] - A[i-1,j,k]) / (θ[i+1] - θ[i-1])
            end
        end
    end
    return dA
end

function central_diff_r(A, r)
    nlat, nlon, nr = size(A)
    dA = similar(A)
    @inbounds for i in 1:nlat, j in 1:nlon
        for k in 1:nr
            if r === nothing || nr == 1
                dA[i,j,k] = 0.0
            elseif k == 1
                dA[i,j,k] = (A[i,j,k+1] - A[i,j,k]) / (r[k+1] - r[k])
            elseif k == nr
                dA[i,j,k] = (A[i,j,k] - A[i,j,k-1]) / (r[k] - r[k-1])
            else
                dA[i,j,k] = (A[i,j,k+1] - A[i,j,k-1]) / (r[k+1] - r[k-1])
            end
        end
    end
    return dA
end

function helicity_field(vr, vt, vp, θ, φ, r)
    nlat, nlon, nr = size(vr)
    # Derivatives
    sinθ = θ === nothing ? ones(nlat) : sin.(θ)
    sinθ_clamped = clamp.(sinθ, 1e-8, Inf)
    dφ_vθ = central_diff_φ(vt, φ)
    dφ_vr = central_diff_φ(vr, φ)
    dθ_vφ = central_diff_θ(vp, θ)
    dθ_vr = central_diff_θ(vr, θ)
    dr_vθ = central_diff_r(vt, r)
    dr_vφ = central_diff_r(vp, r)

    ωr = similar(vr); ωθ = similar(vt); ωφ = similar(vp)
    @inbounds for k in 1:nr
        rr = (r === nothing || k > length(r)) ? 1.0 : r[k]
        for i in 1:nlat, j in 1:nlon
            sθ = sinθ_clamped[i]
            ωr[i,j,k] = (1/(rr*sθ)) * ( (dθ_vφ[i,j,k]*sθ) + vp[i,j,k]*cos(θ[i]) - dφ_vθ[i,j,k] )
            ωθ[i,j,k] = (1/rr) * ( (1/sθ)*dφ_vr[i,j,k] - (dr_vφ[i,j,k] + vp[i,j,k]/rr) )
            ωφ[i,j,k] = (1/rr) * ( dr_vθ[i,j,k] + vt[i,j,k]/rr - dθ_vr[i,j,k] )
        end
    end
    h = vr .* ωr .+ vt .* ωθ .+ vp .* ωφ
    return h
end

function compute_omega_r_from_Tlm(cfg::SHTnsKit.SHTConfig, lvals, mvals, tor_r, tor_i, rvec)
    lmax=cfg.lmax; mmax=cfg.mmax; nlat=cfg.nlat; nlon=cfg.nlon
    nlm, nr = size(tor_r)
    ωr = Array{Float64}(undef, nlat, nlon, nr)
    Tlm = zeros(ComplexF64, lmax+1, mmax+1)
    for k in 1:nr
        fill!(Tlm, 0)
        for i in 1:nlm
            l=lvals[i]; m=mvals[i]
            if l<=lmax && m<=mmax
                Tlm[l+1,m+1]=complex(tor_r[i,k], tor_i[i,k])
            end
        end
        ζ = SHTnsKit.vorticity_grid(cfg, Tlm)  # unit-sphere normal vorticity
        rr = (rvec === nothing || k > length(rvec)) ? 1.0 : rvec[k]
        ωr[:,:,k] = rr > eps() ? (ζ ./ rr) : zeros(nlat, nlon)
    end
    return ωr
end

function volume_average_helicity(h::Array{<:Real,3}, cfg::SHTnsKit.SHTConfig, r::Union{Nothing,Vector{<:Real}})
    nlat, nlon, nr = size(h)
    # Theta weights (include sinθ): cfg.w
    wθ = cfg.w
    Δφ = 2π / nlon
    # Radial weights via trapezoidal on r (if available), else assume unit spacing
    if r === nothing || length(r) != nr
        r = collect(1:nr)
    else
        r = Float64.(r)
    end
    Δr = similar(r)
    for k in 1:nr
        if k == 1
            Δr[k] = (r[min(2,nr)] - r[1])
        elseif k == nr
            Δr[k] = (r[nr] - r[nr-1])
        else
            Δr[k] = 0.5*(r[k+1] - r[k-1])
        end
    end
    vol = 0.0
    integral = 0.0
    @inbounds for k in 1:nr
        rk2 = r[k]^2
        vol += Δφ * Δr[k] * rk2 * sum(wθ)
        # integrate over θ,φ for each k
        # Sum over j: Δφ, over i: wθ[i]
        shell_int = 0.0
        for i in 1:nlat
            wi = wθ[i]
            # sum over φ at fixed (i,k)
            sφ = 0.0
            for j in 1:nlon
                sφ += h[i,j,k]
            end
            shell_int += wi * sφ * Δφ
        end
        integral += rk2 * Δr[k] * shell_int
    end
    return integral / max(vol, eps())
end

function time_average_helicity(dir::String, t0::Float64, t1::Float64, prefix::String)
    times = scan_times(dir, prefix)
    sel = [t for t in times if t0 <= t <= t1]
    isempty(sel) && error("No files found in $dir for prefix '$prefix' and time range [$t0, $t1]")

    sum_h = nothing
    sum_hr = nothing
    sum_hθ = nothing
    sum_hφ = nothing
    count = 0
    coords = Dict{String,Any}()
    meta = Dict{String,Any}()
    last_cfg = Ref{Union{Nothing,SHTnsKit.SHTConfig}}(nothing)

    for t in sel
        fname = build_filename(dir, prefix, t)
        isempty(fname) && (@warn "No file for time $t"; continue)
        nc = NetCDF.open(fname, NC_NOWRITE)
        try
            vtor_r = read_var(nc, "velocity_toroidal_real"); vtor_i = read_var(nc, "velocity_toroidal_imag")
            vpol_r = read_var(nc, "velocity_poloidal_real"); vpol_i = read_var(nc, "velocity_poloidal_imag")
            if any(x->x===nothing, (vtor_r, vtor_i, vpol_r, vpol_i))
                @warn "Velocity spectral variables missing in $(basename(fname)); skipping"
                continue
            end
            cfg, lvals, mvals, (θ, φ, r) = build_sht_from_nc(nc)
            coords["theta"] = θ; coords["phi"] = φ; coords["r"] = r
            try meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end

            vr, vt, vp = vector_components(cfg, lvals, mvals, Float64.(vtor_r), Float64.(vtor_i), Float64.(vpol_r), Float64.(vpol_i), r)
            last_cfg[] = cfg
            ωr = compute_omega_r_from_Tlm(cfg, lvals, mvals, Float64.(vtor_r), Float64.(vtor_i), r)
            # Compute ωθ, ωφ via finite differences on grid
            # Reuse helicity_field to get full h (includes FD-computed ωr), then replace ωr term with accurate ωr
            # Alternatively, compute ωθ/ωφ directly:
            nlat, nlon, nr = size(vr)
            sinθ = θ === nothing ? ones(nlat) : sin.(θ)
            sinθ_clamped = clamp.(sinθ, 1e-8, Inf)
            dφ_vθ = central_diff_φ(vt, φ)
            dφ_vr = central_diff_φ(vr, φ)
            dθ_vφ = central_diff_θ(vp, θ)
            dθ_vr = central_diff_θ(vr, θ)
            dr_vθ = central_diff_r(vt, r)
            dr_vφ = central_diff_r(vp, r)
            ωθ = similar(vt); ωφ = similar(vp)
            @inbounds for k2 in 1:nr
                rr2 = (r === nothing || k2 > length(r)) ? 1.0 : r[k2]
                for i2 in 1:nlat, j2 in 1:nlon
                    sθ = sinθ_clamped[i2]
                    ωθ[i2,j2,k2] = (1/rr2) * ( (1/sθ)*dφ_vr[i2,j2,k2] - (dr_vφ[i2,j2,k2] + vp[i2,j2,k2]/rr2) )
                    ωφ[i2,j2,k2] = (1/rr2) * ( dr_vθ[i2,j2,k2] + vt[i2,j2,k2]/rr2 - dθ_vr[i2,j2,k2] )
                end
            end
            # Component-wise helicity contributions
            hr = vr .* ωr
            hθ = vt .* ωθ
            hφ = vp .* ωφ
            h = hr .+ hθ .+ hφ
            if sum_h === nothing
                sum_h = zero(h); sum_hr = zero(hr); sum_hθ = zero(hθ); sum_hφ = zero(hφ)
            end
            sum_h .+= h
            sum_hr .+= hr
            sum_hθ .+= hθ
            sum_hφ .+= hφ
            count += 1
        finally
            NetCDF.close(nc)
        end
    end

    count == 0 && error("No samples accumulated in range [$t0,$t1]")
    return (sum_h ./ count, sum_hr ./ count, sum_hθ ./ count, sum_hφ ./ count), count, sel, coords, meta, last_cfg[]
end

function main()
    outdir, t0, t1, prefix, outpath = parse_args(copy(ARGS))
    (h_avg, hr_avg, hθ_avg, hφ_avg), count, times_used, coords, meta, cfg = time_average_helicity(outdir, t0, t1, prefix)
    if isempty(outpath)
        ttag = string(replace(@sprintf("%.6f", t0), "."=>"p"), "_", replace(@sprintf("%.6f", t1), "."=>"p"))
        outpath = joinpath(outdir, "timeavg_helicity_$(prefix)_$(ttag).jld2")
    end
    isdir(dirname(outpath)) || mkpath(dirname(outpath))
    # Volume-averaged helicity from time-averaged field
    Hvol = (cfg === nothing) ? NaN : volume_average_helicity(h_avg, cfg, get(coords, "r", nothing))
    jldopen(outpath, "w"; compress=true) do f
        write(f, "helicity", h_avg)
        write(f, "helicity_r", hr_avg)
        write(f, "helicity_theta", hθ_avg)
        write(f, "helicity_phi", hφ_avg)
        write(f, "count", count)
        write(f, "times", times_used)
        write(f, "coords", coords)
        write(f, "metadata", meta)
        write(f, "time_range", (t0, t1))
        write(f, "prefix", prefix)
        write(f, "volume_avg_helicity", Hvol)
    end
    println(@sprintf("Saved time-averaged helicity to %s (samples=%d)", outpath, count))
end

isinteractive() || main()
