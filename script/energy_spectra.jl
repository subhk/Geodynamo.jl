#!/usr/bin/env julia

"""
Compute kinetic and magnetic energy spectra E_k(l,m) and E_b(l,m) from a NetCDF output file,
and save results to a single JLD2 file.

Usage:
  julia --project=. script/energy_spectra.jl <input.nc> [output.jld2]

Inputs are expected to follow Geodynamo.jl output structure with spectral variables:
  - velocity_toroidal_real/imag[spectral_mode, r]
  - velocity_poloidal_real/imag[spectral_mode, r]
  - magnetic_toroidal_real/imag[spectral_mode, r]
  - magnetic_poloidal_real/imag[spectral_mode, r]
  - l_values[spectral_mode], m_values[spectral_mode]

Output (JLD2):
  - l_values::Vector{Int}
  - m_values::Vector{Int}
  - Ek_lm::Vector{Float64}
  - Eb_lm::Vector{Float64}
  - Ek_l::Vector{Float64}
  - Eb_l::Vector{Float64}
  - Ek_m::Vector{Float64}
  - Eb_m::Vector{Float64}
  - metadata::Dict (if available): geometry, time, step
"""

using NetCDF
using Printf
using JLD2
using Statistics

function read_var(nc, name)
    varid = NetCDF.varid(nc, name)
    varid == -1 && error("Variable '$name' not found in file")
    return NetCDF.readvar(nc, name)
end

function load_spectral_data(nc)
    l_values = Int.(read_var(nc, "l_values"))
    m_values = Int.(read_var(nc, "m_values"))

    has_velocity = (NetCDF.varid(nc, "velocity_toroidal_real") != -1)
    has_magnetic = (NetCDF.varid(nc, "magnetic_toroidal_real") != -1)

    if !has_velocity && !has_magnetic
        error("No spectral velocity or magnetic variables found in file")
    end

    # Helper to load a pair of real/imag arrays if present, else zeros
    function load_pair(prefix)
        if NetCDF.varid(nc, "$(prefix)_real") != -1 && NetCDF.varid(nc, "$(prefix)_imag") != -1
            real = Float64.(read_var(nc, "$(prefix)_real"))
            imag = Float64.(read_var(nc, "$(prefix)_imag"))
        else
            real = zeros(Float64, length(l_values), 1)
            imag = zeros(Float64, length(m_values), 1)
        end
        return real, imag
    end

    vtor_r, vtor_i = load_pair("velocity_toroidal")
    vpol_r, vpol_i = load_pair("velocity_poloidal")
    mtor_r, mtor_i = load_pair("magnetic_toroidal")
    mpol_r, mpol_i = load_pair("magnetic_poloidal")

    return (
        l_values = l_values,
        m_values = m_values,
        vtor = (vtor_r, vtor_i),
        vpol = (vpol_r, vpol_i),
        mtor = (mtor_r, mtor_i),
        mpol = (mpol_r, mpol_i),
    )
end

function energy_per_mode(real::AbstractMatrix, imag::AbstractMatrix)
    # Sum over radius of |coeff|^2
    # Input shape: (nlm, nr)
    @assert size(real) == size(imag)
    nlm, nr = size(real)
    e = zeros(Float64, nlm)
    @inbounds for i in 1:nlm
        s = 0.0
        for r in 1:nr
            s += real[i,r]^2 + imag[i,r]^2
        end
        e[i] = s
    end
    return e
end

function compute_spectra(data)
    l = data.l_values
    m = data.m_values
    nlm = length(l)

    # Kinetic energy per (l,m): sum(|v_tor|^2 + |v_pol|^2) across radius
    Ek_lm = energy_per_mode(data.vtor[1], data.vtor[2]) .+ energy_per_mode(data.vpol[1], data.vpol[2])
    # Magnetic energy per (l,m)
    Eb_lm = energy_per_mode(data.mtor[1], data.mtor[2]) .+ energy_per_mode(data.mpol[1], data.mpol[2])

    # Aggregate by l and m
    lmax = maximum(l)
    mmax = maximum(m)
    Ek_l = zeros(Float64, lmax+1)
    Eb_l = zeros(Float64, lmax+1)
    Ek_m = zeros(Float64, mmax+1)
    Eb_m = zeros(Float64, mmax+1)

    @inbounds for i in 1:nlm
        ℓ = l[i]
        𝓶 = m[i]
        Ek_l[ℓ+1] += Ek_lm[i]
        Eb_l[ℓ+1] += Eb_lm[i]
        Ek_m[𝓶+1] += Ek_lm[i]
        Eb_m[𝓶+1] += Eb_lm[i]
    end

    return (Ek_lm=Ek_lm, Eb_lm=Eb_lm, Ek_l=Ek_l, Eb_l=Eb_l, Ek_m=Ek_m, Eb_m=Eb_m)
end

function main()
    if length(ARGS) < 1
        println("Usage: julia --project=. script/energy_spectra.jl <input.nc> [output.jld2]")
        return
    end
    input = ARGS[1]
    outpath = length(ARGS) >= 2 ? ARGS[2] : joinpath(dirname(input), "energy_spectra.jld2")
    outdir = dirname(outpath)
    isdir(outdir) || mkpath(outdir)

    nc = NetCDF.open(input, NC_NOWRITE)
    try
        data = load_spectral_data(nc)
        spectra = compute_spectra(data)

        # Try to collect basic metadata
        meta = Dict{String,Any}()
        try
            if NetCDF.varid(nc, "time") != -1
                meta["time"] = NetCDF.readvar(nc, "time")[1]
            end
            if NetCDF.varid(nc, "step") != -1
                meta["step"] = NetCDF.readvar(nc, "step")[1]
            end
            # Global attribute "geometry" if present
            try
                meta["geometry"] = NetCDF.getatt(nc, NetCDF.NC_GLOBAL, "geometry")
            catch
                # Ignore if missing
            end
        catch
            # Ignore metadata failures
        end

        # Save JLD2
        @save outpath data.l_values data.m_values spectra.Ek_lm spectra.Eb_lm spectra.Ek_l spectra.Eb_l spectra.Ek_m spectra.Eb_m meta
        println(@sprintf("Wrote spectra to %s", outpath))
    finally
        NetCDF.close(nc)
    end
end

isinteractive() || main()
