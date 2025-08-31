#!/usr/bin/env julia

# Time-average z-helicity (u_z * ω_z) separated into positive and negative contributions.
#
# - Reads a sequence of NetCDF outputs (one per time) with velocity spectral coefficients
#   `velocity_toroidal_real/imag` and `velocity_poloidal_real/imag`, plus `theta, phi, r`.
# - Uses SHTnsKit to synthesize tangential components (u_θ, u_φ) on the stored grid and
#   reconstructs u_r from poloidal coefficients via u_r = L2/r * P_lm (orthonormal Ylm).
# - Computes vorticity ω = ∇ × u in spherical coordinates, then u_z = u_r cosθ - u_θ sinθ,
#   ω_z = ω_r cosθ - ω_θ sinθ, and z-helicity h_z = u_z * ω_z.
# - Accumulates time-averages separately for h_z > 0 and h_z < 0 at each grid cell.
# - Writes a JLD2 file with fields: `avg_pos`, `avg_neg`, `count_pos`, `count_neg`, and axes.
#
# Usage:
#   # Average all files in a directory (prefers rank_0000) and write JLD2
#   julia --project examples/timeavg_zhelicity.jl outdir --out zhel_timeavg.jld2
#   # Restrict to time range
#   julia --project examples/timeavg_zhelicity.jl outdir --tmin 1.0 --tmax 5.0 --out zhel_1_5.jld2

using NCDatasets
using JLD2

const HAVE_SHTNSKIT = try
    @eval using SHTnsKit
    true
catch
    false
end

mutable struct Opts
    dir::String
    out::String
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
end

function parse_args()
    dir = get(ARGS, 1, "")
    dir != "" || error("Provide directory containing NetCDF outputs as first argument")
    opts = Opts(dir, "zhelicity_timeavg.jld2", nothing, nothing)
    i = 2
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--out" && i+1 <= length(ARGS)
            opts.out = ARGS[i+1]; i += 2
        elseif a == "--tmin" && i+1 <= length(ARGS)
            opts.tmin = parse(Float64, ARGS[i+1]); i += 2
        elseif a == "--tmax" && i+1 <= length(ARGS)
            opts.tmax = parse(Float64, ARGS[i+1]); i += 2
        else
            error("Unknown option $a. Use --out FILE --tmin T --tmax T")
        end
    end
    HAVE_SHTNSKIT || error("SHTnsKit.jl required. Install with: import Pkg; Pkg.add(\"SHTnsKit\")")
    return opts
end

time_of_file(path::AbstractString) = begin
    ds = NCDataset(path)
    try
        if haskey(ds, "time"); t = ds["time"][:]; return length(t)==1 ? float(t[1]) : float(first(t)); end
        if haskey(ds.attrib, "current_time"); return float(ds.attrib["current_time"]); end
        m = match(r"_time_([0-9p]+)", basename(path))
        m === nothing && return NaN
        parse(Float64, replace(m.captures[1], 'p' => '.'))
    finally
        close(ds)
    end
end

function list_nc(dir::String)
    files = filter(f->endswith(f, ".nc"), joinpath.(dir, readdir(dir)))
    r0 = filter(f->occursin("rank_0000", f), files)
    files = isempty(r0) ? files : r0
    times = [(f, time_of_file(f)) for f in files]
    sort(times; by=x->isfinite(x[2]) ? x[2] : Inf)
end

function coeffs_at_r(ds::NCDataset, r_idx::Int)
    l = vec(ds["l_values"][:]); m = vec(ds["m_values"][:])
    T_re = Array(ds["velocity_toroidal_real"][:]); T_im = Array(ds["velocity_toroidal_imag"][:])
    P_re = Array(ds["velocity_poloidal_real"][:]); P_im = Array(ds["velocity_poloidal_imag"][:])
    lmax = maximum(l); mmax = maximum(m)
    Tor = zeros(ComplexF64, lmax+1, mmax+1)
    Pol = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for i in eachindex(l)
        ell = l[i]; mm = m[i]
        Tor[ell+1, mm+1] = complex(T_re[i, r_idx], T_im[i, r_idx])
        Pol[ell+1, mm+1] = complex(P_re[i, r_idx], P_im[i, r_idx])
    end
    return l, m, Tor, Pol
end

function synthesize_velocity!(u_r, u_θ, u_φ, sht, theta, phi, rvals, r_idx, l::Vector{Int}, Pol::Matrix{ComplexF64}, Tor::Matrix{ComplexF64})
    # Tangential from SHTnsKit
    vt, vp = SHTnsKit.SHsphtor_to_spat(sht, Pol, Tor; real_output=true)
    u_θ[:, :, r_idx] .= vt
    u_φ[:, :, r_idx] .= vp
    # Radial from poloidal scalar: u_r = L2/r * P_lm Y_lm
    lmax = maximum(l); mmax = size(Pol, 2)-1
    C = zeros(ComplexF64, lmax+1, mmax+1)
    r = rvals[r_idx]
    @inbounds for i in 1:(lmax+1)
        C[i,1:end] .= ( (i-1)*(i) / r ) .* Pol[i,1:end]
    end
    u_r[:, :, r_idx] .= SHTnsKit.synthesis(sht, C; real_output=true)
end

function curl_spherical!(ω_r, ω_θ, ω_φ, u_r, u_θ, u_φ, rvals, theta)
    nθ, nφ, nr = size(u_r)
    # Precompute sin/cos and Δ grids
    sinθ = sin.(theta); cosθ = cos.(theta)
    # Finite differences in r and θ, periodic in φ
    function d_dθ(A)
        d = similar(A)
        for k in 1:nr, j in 2:(nθ-1), i in 1:nφ
            d[j,i,k] = (A[j+1,i,k] - A[j-1,i,k]) / (theta[j+1]-theta[j-1])
        end
        d[1, :, :] .= (A[2,:,:] .- A[1,:,:]) ./ (theta[2]-theta[1])
        d[end, :, :] .= (A[end,:,:] .- A[end-1,:,:]) ./ (theta[end]-theta[end-1])
        d
    end
    function d_dφ(A)
        d = similar(A)
        for k in 1:nr, j in 1:nθ
            @inbounds for i in 1:nφ
                ip = (i % nφ) + 1; im = (i-2) % nφ + 1
                d[j,i,k] = (A[j,ip,k] - A[j,im,k]) / (2*(phi[ip]-phi[i]))
            end
        end
        d
    end
    function d_dr(A)
        d = similar(A)
        for j in 1:nθ, i in 1:nφ
            d[j,i,1] = (A[j,i,2] - A[j,i,1])/(rvals[2]-rvals[1])
            for k in 2:(nr-1)
                d[j,i,k] = (A[j,i,k+1] - A[j,i,k-1])/(rvals[k+1]-rvals[k-1])
            end
            d[j,i,nr] = (A[j,i,nr] - A[j,i,nr-1])/(rvals[nr]-rvals[nr-1])
        end
        d
    end
    # Needed combinations
    d_sin_uφ_dθ = d_dθ(sinθ .* u_φ)
    du_θ_dφ = d_dφ(u_θ)
    du_r_dφ = d_dφ(u_r)
    d_ruθ_dr = d_dr(rvals' .* u_θ)
    d_ruφ_dr = d_dr(rvals' .* u_φ)
    du_r_dθ = d_dθ(u_r)
    # Curl formulas in spherical (r,θ,φ)
    # ω_r = 1/(r sinθ) [ ∂(sinθ u_φ)/∂θ - ∂u_θ/∂φ ]
    for k in 1:nr, j in 1:nθ, i in 1:nφ
        denom = rvals[k]*sinθ[j]
        ω_r[j,i,k] = (d_sin_uφ_dθ[j,i,k] - du_θ_dφ[j,i,k]) / denom
    end
    # ω_θ = 1/r [ (1/sinθ) ∂u_r/∂φ - ∂(r u_φ)/∂r ]
    for k in 1:nr, j in 1:nθ, i in 1:nφ
        ω_θ[j,i,k] = ( (1/sinθ[j]) * du_r_dφ[j,i,k] - d_ruφ_dr[j,i,k] ) / rvals[k]
    end
    # ω_φ = 1/r [ ∂(r u_θ)/∂r - ∂u_r/∂θ ]
    for k in 1:nr, j in 1:nθ, i in 1:nφ
        ω_φ[j,i,k] = ( d_ruθ_dr[j,i,k] - du_r_dθ[j,i,k] ) / rvals[k]
    end
end

function main()
    opts = parse_args()
    files_times = list_nc(opts.dir)
    # Filter by time range
    if opts.tmin !== nothing || opts.tmax !== nothing
        files_times = [(f,t) for (f,t) in files_times if (opts.tmin === nothing || t >= opts.tmin) && (opts.tmax === nothing || t <= opts.tmax)]
    end
    isempty(files_times) && error("No .nc files found matching time range")
    # Read grid from first file and init SHTnsKit
    ds0 = NCDataset(files_times[1][1])
    theta = vec(ds0["theta"][:]); phi = vec(ds0["phi"][:]); rvals = vec(ds0["r"][:])
    l = vec(ds0["l_values"][:]); m = vec(ds0["m_values"][:])
    lmax = maximum(l); mmax = maximum(m)
    sht = SHTnsKit.create_gauss_config(lmax, length(theta); mmax=mmax, nlon=length(phi), norm=:orthonormal)
    SHTnsKit.prepare_plm_tables!(sht)
    close(ds0)

    nθ, nφ, nr = length(theta), length(phi), length(rvals)
    sum_pos = zeros(Float64, nθ, nφ, nr)
    sum_neg = zeros(Float64, nθ, nφ, nr)
    cnt_pos = zeros(Int, nθ, nφ, nr)
    cnt_neg = zeros(Int, nθ, nφ, nr)

    for (path, t) in files_times
        ds = NCDataset(path)
        try
            # Allocate velocity fields for this timestep
            u_r = zeros(Float64, nθ, nφ, nr)
            u_θ = similar(u_r); u_φ = similar(u_r)
            # Synthesize per radius to limit memory
            for k in 1:nr
                l, m, Tor, Pol = coeffs_at_r(ds, k)
                synthesize_velocity!(u_r, u_θ, u_φ, sht, theta, phi, rvals, k, l, Pol, Tor)
            end
            # Curl
            ω_r = zeros(Float64, nθ, nφ, nr)
            ω_θ = similar(ω_r); ω_φ = similar(ω_r)
            curl_spherical!(ω_r, ω_θ, ω_φ, u_r, u_θ, u_φ, rvals, theta)
            # z-components and z-helicity
            cosθ = cos.(theta); sinθ = sin.(theta)
            # Broadcast over phi,r dims via reshape
            Cos = reshape(cosθ, nθ, 1, 1); Sin = reshape(sinθ, nθ, 1, 1)
            u_z = u_r .* Cos .- u_θ .* Sin
            ω_z = ω_r .* Cos .- ω_θ .* Sin
            h = u_z .* ω_z
            # Accumulate conditional time averages
            posmask = h .> 0
            negmask = h .< 0
            sum_pos .+= h .* posmask
            sum_neg .+= h .* negmask
            cnt_pos .+= posmask
            cnt_neg .+= negmask
        finally
            close(ds)
        end
    end

    avg_pos = Array{Float64}(undef, nθ, nφ, nr)
    avg_neg = Array{Float64}(undef, nθ, nφ, nr)
    @inbounds for k in 1:nr, j in 1:nφ, i in 1:nθ
        cp = cnt_pos[i,j,k]; cn = cnt_neg[i,j,k]
        avg_pos[i,j,k] = cp>0 ? sum_pos[i,j,k]/cp : NaN
        avg_neg[i,j,k] = cn>0 ? sum_neg[i,j,k]/cn : NaN
    end

    jldsave(opts.out;
        theta=theta, phi=phi, r=rvals,
        avg_pos=avg_pos, avg_neg=avg_neg,
        count_pos=cnt_pos, count_neg=cnt_neg,
        description="Time-averaged z-helicity (u_z*ω_z) separated by sign")
    println("Saved ", opts.out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

