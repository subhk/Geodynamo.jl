#!/usr/bin/env julia

"""
Time-average selected quantities from merged NetCDF outputs and save to a compressed JLD2 file.

Assumptions:
- Inputs are merged (global) NetCDF files produced by Geodynamo's combiner/output system.
- Variables are consistent across the files in shape and meaning.

Usage examples:

  # Average temperature from t=0.0 to t=2.0 and save default JLD2
  julia --project=. script/time_average.jl ./output --start=0.0 --end=2.0 --vars=temperature

  # Average multiple quantities and write to an explicit file
  julia --project=. script/time_average.jl ./output --start=1.0 --end=3.0 \
       --vars=temperature,velocity_toroidal_real,velocity_toroidal_imag \
       --out=./output/avg_1p0_3p0.jld2

  # Limit to files with a particular prefix (default combined_global)
  julia --project=. script/time_average.jl ./output --start=0 --end=5 --vars=temperature --prefix=combined_global

Notes:
- The script scans for files matching "<prefix>_time_<time>.nc" and averages over times in [start,end].
- The output JLD2 contains one dataset per requested variable plus metadata.
"""

using Printf
using NetCDF
using JLD2
using SHTnsKit

function usage()
    println("Usage: time_average.jl <output_dir> --start=<t0> --end=<t1> --vars=v1[,v2,...] [--prefix=name] [--out=path.jld2]")
end

function parse_args(args)
    if isempty(args)
        usage(); error("missing arguments")
    end
    outdir = abspath(args[1])
    t0 = nothing
    t1 = nothing
    varlist = String[]
    prefix = "combined_global"
    outpath = ""
    for a in args[2:end]
        if startswith(a, "--start="); t0 = parse(Float64, split(a, "=", limit=2)[2]);
        elseif startswith(a, "--end="); t1 = parse(Float64, split(a, "=", limit=2)[2]);
        elseif startswith(a, "--vars=")
            v = split(split(a, "=", limit=2)[2], ",")
            append!(varlist, v)
        elseif startswith(a, "--prefix=")
            prefix = split(a, "=", limit=2)[2]
        elseif startswith(a, "--out=")
            outpath = split(a, "=", limit=2)[2]
        else
            @warn "Unknown argument $a (ignored)"
        end
    end
    (t0 === nothing || t1 === nothing || isempty(varlist)) && (usage(); error("--start, --end, and --vars are required"))
    return outdir, t0, t1, varlist, prefix, outpath
end

format_time_str(t::Float64) = replace(@sprintf("%.6f", t), "." => "p")

function scan_times(dir::String, prefix::String)
    files = filter(f -> endswith(f, ".nc") && occursin(prefix, f), readdir(dir))
    times = Float64[]
    pattern = r"time_(\d+p\d+)"
    for f in files
        if (m = match(pattern, f)) !== nothing
            push!(times, parse(Float64, replace(m.captures[1], "p" => ".")))
        end
    end
    return sort(unique(times))
end

function build_filename(dir::String, prefix::String, t::Float64)
    ts = format_time_str(t)
    # Accept both naming patterns used in outputs
    candidates = [
        joinpath(dir, "$(prefix)_time_$(ts).nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts)_rank_0000.nc"),
        joinpath(dir, "$(prefix)_output_time_$(ts).nc"),
    ]
    for c in candidates
        if isfile(c); return c; end
    end
    # Fallback: any file in dir with the correct time token
    for f in readdir(dir)
        full = joinpath(dir, f)
        if endswith(f, ".nc") && occursin("time_$(ts)", f) && occursin(prefix, f)
            return full
        end
    end
    return ""
end

function read_var_if_exists(nc, name)
    varid = NetCDF.varid(nc, name)
    varid == -1 && return nothing
    return NetCDF.readvar(nc, name)
end

function build_sht_config_from_file(nc)
    l_values = Int.(read_var_if_exists(nc, "l_values"))
    m_values = Int.(read_var_if_exists(nc, "m_values"))
    lmax = maximum(l_values)
    mmax = maximum(m_values)
    nlat = (NetCDF.varid(nc, "theta") != -1) ? length(read_var_if_exists(nc, "theta")) : (lmax+2)
    nlon = (NetCDF.varid(nc, "phi") != -1) ? length(read_var_if_exists(nc, "phi")) : max(2*lmax+1, 4)
    cfg = SHTnsKit.create_gauss_config(lmax, nlat; mmax=mmax, nlon=nlon, norm=:orthonormal)
    return cfg, l_values, m_values
end

function vector_synthesis_theta_phi(cfg::SHTnsKit.SHTConfig,
                                    l_values::Vector{Int}, m_values::Vector{Int},
                                    tor_r::AbstractMatrix, tor_i::AbstractMatrix,
                                    pol_r::AbstractMatrix, pol_i::AbstractMatrix)
    lmax = cfg.lmax; mmax = cfg.mmax
    nlat = cfg.nlat; nlon = cfg.nlon
    nlm, nr = size(tor_r)
    vt = Array{Float64}(undef, nlat, nlon, nr)
    vp = Array{Float64}(undef, nlat, nlon, nr)
    # Temporary coefficient matrices
    tor = zeros(ComplexF64, lmax+1, mmax+1)
    pol = zeros(ComplexF64, lmax+1, mmax+1)
    for r in 1:nr
        fill!(tor, 0); fill!(pol, 0)
        for i in 1:nlm
            l = l_values[i]; m = m_values[i]
            if l <= lmax && m <= mmax
                tor[l+1, m+1] = complex(tor_r[i,r], tor_i[i,r])
                pol[l+1, m+1] = complex(pol_r[i,r], pol_i[i,r])
            end
        end
        vt_slice, vp_slice = SHTnsKit.SHsphtor_to_spat(cfg, pol, tor; real_output=true)
        vt[:,:,r] = vt_slice
        vp[:,:,r] = vp_slice
    end
    return vt, vp
end

function time_average(dir::String, t0::Float64, t1::Float64, vars::Vector{String}, prefix::String)
    times = scan_times(dir, prefix)
    sel = [t for t in times if t0 <= t <= t1]
    isempty(sel) && error("No files found in $dir for prefix '$prefix' and time range [$t0, $t1]")

    # Initialize accumulators on first successful read per var
    sums = Dict{String, Any}()
    counts = Dict{String, Int}()
    shapes = Dict{String, Tuple}()
    sample_meta = Dict{String,Any}()

    for t in sel
        fname = build_filename(dir, prefix, t)
        if isempty(fname)
            @warn "No file found for time $t (skipping)"; continue
        end
        nc = NetCDF.open(fname, NC_NOWRITE)
        try
            # Grab time/step/geometry once
            if isempty(sample_meta)
                try sample_meta["time"] = read_var_if_exists(nc, "time")[1] catch end
                try sample_meta["step"] = read_var_if_exists(nc, "step")[1] catch end
                try sample_meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry") catch end
            end
            # Handle vector conversion if requested
            convert_velocity = any(v -> v in ("velocity","velocity_vector"), vars)
            convert_magnetic = any(v -> v in ("magnetic","magnetic_vector"), vars)

            if convert_velocity
                # Load spectral pairs; skip if missing
                vtor_r = read_var_if_exists(nc, "velocity_toroidal_real"); vtor_i = read_var_if_exists(nc, "velocity_toroidal_imag")
                vpol_r = read_var_if_exists(nc, "velocity_poloidal_real"); vpol_i = read_var_if_exists(nc, "velocity_poloidal_imag")
                if vtor_r !== nothing && vtor_i !== nothing && vpol_r !== nothing && vpol_i !== nothing
                    cfg, l_values, m_values = build_sht_config_from_file(nc)
                    vt, vp = vector_synthesis_theta_phi(cfg, l_values, m_values, Float64.(vtor_r), Float64.(vtor_i), Float64.(vpol_r), Float64.(vpol_i))
                    for (name, A) in zip(("velocity_theta","velocity_phi"), (vt, vp))
                        if !haskey(sums, name)
                            sums[name] = zero(A); shapes[name] = size(A); counts[name] = 0
                        end
                        if size(A) != shapes[name]
                            @warn "Shape mismatch for '$name' in $(basename(fname)) (skipping)"
                        else
                            sums[name] .+= A; counts[name] += 1
                        end
                    end
                else
                    @warn "Velocity spectral variables not found in $(basename(fname)); skipping conversion"
                end
            end

            if convert_magnetic
                mtor_r = read_var_if_exists(nc, "magnetic_toroidal_real"); mtor_i = read_var_if_exists(nc, "magnetic_toroidal_imag")
                mpol_r = read_var_if_exists(nc, "magnetic_poloidal_real"); mpol_i = read_var_if_exists(nc, "magnetic_poloidal_imag")
                if mtor_r !== nothing && mtor_i !== nothing && mpol_r !== nothing && mpol_i !== nothing
                    cfg, l_values, m_values = build_sht_config_from_file(nc)
                    bt, bp = vector_synthesis_theta_phi(cfg, l_values, m_values, Float64.(mtor_r), Float64.(mtor_i), Float64.(mpol_r), Float64.(mpol_i))
                    for (name, A) in zip(("magnetic_theta","magnetic_phi"), (bt, bp))
                        if !haskey(sums, name)
                            sums[name] = zero(A); shapes[name] = size(A); counts[name] = 0
                        end
                        if size(A) != shapes[name]
                            @warn "Shape mismatch for '$name' in $(basename(fname)) (skipping)"
                        else
                            sums[name] .+= A; counts[name] += 1
                        end
                    end
                else
                    @warn "Magnetic spectral variables not found in $(basename(fname)); skipping conversion"
                end
            end

            # Average plain variables (non-converted)
            for v in vars
                if v in ("velocity","velocity_vector","magnetic","magnetic_vector")
                    continue
                end
                data = read_var_if_exists(nc, v)
                if data === nothing
                    @warn "Variable '$v' not in $(basename(fname)) (skipping this file for the var)"; continue
                end
                A = Float64.(data)
                if !haskey(sums, v)
                    sums[v] = zero(A); shapes[v] = size(A); counts[v] = 0
                elseif size(A) != shapes[v]
                    @warn "Shape mismatch for '$v' in $(basename(fname)): got $(size(A)), expected $(shapes[v]) (skipping)"; continue
                end
                sums[v] .+= A; counts[v] += 1
            end
        finally
            NetCDF.close(nc)
        end
    end

    # Compute averages
    avgs = Dict{String, Any}()
    # Include converted variable names if requested
    allkeys = copy(vars)
    if any(v -> v in ("velocity","velocity_vector"), vars)
        append!(allkeys, ["velocity_theta","velocity_phi"])
    end
    if any(v -> v in ("magnetic","magnetic_vector"), vars)
        append!(allkeys, ["magnetic_theta","magnetic_phi"])
    end

    for v in allkeys
        if !haskey(counts, v) || counts[v] == 0
            @warn "No samples accumulated for variable '$v'"
            continue
        end
        avgs[v] = sums[v] ./ counts[v]
    end

    return avgs, counts, sel, sample_meta
end

function main()
    outdir, t0, t1, varlist, prefix, outpath = parse_args(copy(ARGS))
    avgs, counts, times_used, meta = time_average(outdir, t0, t1, varlist, prefix)

    if isempty(outpath)
        ttag = string(replace(@sprintf("%.6f", t0), "."=>"p"), "_", replace(@sprintf("%.6f", t1), "."=>"p"))
        outpath = joinpath(outdir, "timeavg_$(prefix)_$(ttag).jld2")
    end
    isdir(dirname(outpath)) || mkpath(dirname(outpath))

    # Save with compression
    jldopen(outpath, "w"; compress=true) do f
        write(f, "variables", varlist)
        write(f, "averages", avgs)
        write(f, "counts", counts)
        write(f, "times", times_used)
        write(f, "metadata", meta)
        write(f, "time_range", (t0, t1))
        write(f, "prefix", prefix)
    end
    println(@sprintf("Saved time-averaged variables to %s", outpath))
end

isinteractive() || main()
