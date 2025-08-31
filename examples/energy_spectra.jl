#!/usr/bin/env julia

# Energy spectra (E_l) computed from spectral coefficients using SHTnsKit grids.
#
# Supports vector fields (velocity, magnetic) in toroidal/poloidal form and
# scalar fields (temperature_spectral, composition_spectral). For vectors, uses
# the common Parseval-consistent expressions (orthonormal Y_lm):
#   E_l(r) = 0.5 * sum_m [ (l(l+1)/r^2) (|T_lm|^2 + |P_lm|^2) + |∂(r P_lm)/∂r|^2 ]
# For scalars:
#   E_l(r) = 0.5 * sum_m |A_lm|^2
#
# These formulas match SHTnsKit’s orthonormal normalization.
#
# Usage examples:
#   # Velocity spectra at outer radius
#   julia --project examples/energy_spectra.jl out.nc velocity --r outer --out vel_outer.csv
#   # Magnetic spectra at nearest r=0.8 and print top 10
#   julia --project examples/energy_spectra.jl out.nc magnetic --r 0.8 --top 10
#   # Temperature scalar spectra integrated over r (shell-integrated)
#   julia --project examples/energy_spectra.jl out.nc temperature --integrate --out temp_integrated.csv
#
using NCDatasets
const HAVE_SHTNSKIT = try
    @eval using SHTnsKit
    true
catch
    false
end

mutable struct Opts
    rsel::Union{Nothing,String,Int,Float64}
    integrate::Bool
    out::Union{Nothing,String}
    top::Union{Nothing,Int}
    axis::Symbol  # :l or :m for scalar spectra
end

Opts() = Opts("outer", false, nothing, nothing, :l)

function parse_args()
    path = get(ARGS, 1, "")
    family = lowercase(get(ARGS, 2, "velocity"))  # velocity|magnetic|temperature|composition
    extras = ARGS[3:end]
    opts = Opts()
    i = 1
    while i <= length(extras)
        a = extras[i]
        if a == "--r" && i+1 <= length(extras)
            s = extras[i+1]
            if s in ("outer","inner")
                opts.rsel = s
            else
                try
                    v = parse(Float64, s)
                    opts.rsel = v
                catch
                    try
                        opts.rsel = parse(Int, s)
                    catch
                        error("--r expects 'outer'|'inner'|Int index|Float radius")
                    end
                end
            end
            i += 2
        elseif a == "--integrate"
            opts.integrate = true; i += 1
        elseif a == "--out" && i+1 <= length(extras)
            opts.out = extras[i+1]; i += 2
        elseif a == "--top" && i+1 <= length(extras)
            opts.top = parse(Int, extras[i+1]); i += 2
        elseif a == "--axis" && i+1 <= length(extras)
            ax = lowercase(extras[i+1])
            ax ∈ ("l","m") || error("--axis must be l or m")
            opts.axis = Symbol(ax); i += 2
        else
            error("Unknown option '$a'. Use: --r VAL --integrate --out FILE --top N")
        end
    end
    family ∈ ("velocity","magnetic","temperature","composition") || error("family must be velocity|magnetic|temperature|composition")
    return path, family, opts
end

pick_r_index(rvals::AbstractVector{<:Real}, rsel::Union{Nothing,String,Int,Float64}) =
    rsel === nothing ? length(rvals) :
    (rsel isa String && lowercase(rsel) == "outer") ? length(rvals) :
    (rsel isa String && lowercase(rsel) == "inner") ? 1 :
    (rsel isa Int) ? Int(rsel) :
    (rsel isa Real) ? findmin(abs.(rvals .- float(rsel)))[2] :  length(rvals)

function read_spectral(path::AbstractString, family::String)
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, "l_values") || error("l_values not found")
        haskey(ds, "m_values") || error("m_values not found")
        haskey(ds, "r") || error("r not found")
        lvals = vec(ds["l_values"][:])
        mvals = vec(ds["m_values"][:])
        rvals = vec(ds["r"][:])
        if family in ("velocity","magnetic")
            tor_r = "$(family)_toroidal_real"; tor_i = "$(family)_toroidal_imag"
            pol_r = "$(family)_poloidal_real"; pol_i = "$(family)_poloidal_imag"
            haskey(ds, tor_r) || error("$tor_r not found")
            haskey(ds, pol_r) || error("$pol_r not found")
            tor_real = Array(ds[tor_r][:])
            tor_imag = Array(ds[tor_i][:])
            pol_real = Array(ds[pol_r][:])
            pol_imag = Array(ds[pol_i][:])
            return (mode=:vector, l=lvals, m=mvals, r=rvals,
                    tor_real=tor_real, tor_imag=tor_imag,
                    pol_real=pol_real, pol_imag=pol_imag, ds=ds)
        else
            base = family == "temperature" ? "temperature_spectral" : "composition_spectral"
            sca_r = "$(base)_real"; sca_i = "$(base)_imag"
            haskey(ds, sca_r) || error("$sca_r not found")
            haskey(ds, sca_i) || error("$sca_i not found")
            sca_real = Array(ds[sca_r][:])
            sca_imag = Array(ds[sca_i][:])
            return (mode=:scalar, l=lvals, m=mvals, r=rvals, sca_real=sca_real, sca_imag=sca_imag, ds=ds)
        end
    catch
        close(ds); rethrow()
    end
end

function assemble_scalar_coeffs_matrix(l::Vector{Int}, m::Vector{Int},
                                       A_real::AbstractMatrix, A_imag::AbstractMatrix,
                                       r_idx::Int)
    lmax = maximum(l); mmax = maximum(m)
    C = zeros(ComplexF64, lmax+1, mmax+1)
    @inbounds for i in eachindex(l)
        ell = l[i]; mm = m[i]
        C[ell+1, mm+1] = complex(A_real[i, r_idx], A_imag[i, r_idx])
    end
    return C
end

function spectra_scalar_at_r(l::Vector{Int}, m::Vector{Int}, A_real::AbstractMatrix, A_imag::AbstractMatrix, r_idx::Int)
    if HAVE_SHTNSKIT && isdefined(SHTnsKit, :energy_scalar_l_spectrum)
        C = assemble_scalar_coeffs_matrix(l, m, A_real, A_imag, r_idx)
        # SHTnsKit returns per-l energies; multiply by 0.5 if needed to match our definition.
        El = SHTnsKit.energy_scalar_l_spectrum(C)
        return 0.5 .* El
    else
        lmax = maximum(l)
        E = zeros(Float64, lmax+1)
        for (i, ell) in enumerate(l)
            are = A_real[i, r_idx]; aim = A_imag[i, r_idx]
            E[ell+1] += 0.5 * (are*are + aim*aim)
        end
        return E
    end
end

function spectra_scalar_m_at_r(l::Vector{Int}, m::Vector{Int}, A_real::AbstractMatrix, A_imag::AbstractMatrix, r_idx::Int)
    if HAVE_SHTNSKIT && isdefined(SHTnsKit, :energy_scalar_m_spectrum)
        C = assemble_scalar_coeffs_matrix(l, m, A_real, A_imag, r_idx)
        Em = SHTnsKit.energy_scalar_m_spectrum(C)
        return 0.5 .* Em
    else
        mmax = maximum(m)
        E = zeros(Float64, mmax+1)
        for (i, mm) in enumerate(m)
            are = A_real[i, r_idx]; aim = A_imag[i, r_idx]
            E[mm+1] += 0.5 * (are*are + aim*aim)
        end
        return E
    end
end

function spectra_vector_at_r(l::Vector{Int}, m::Vector{Int},
                             T_real::AbstractMatrix, T_imag::AbstractMatrix,
                             P_real::AbstractMatrix, P_imag::AbstractMatrix,
                             rvals::AbstractVector, r_idx::Int)
    lmax = maximum(l)
    E = zeros(Float64, lmax+1)
    r = rvals[r_idx]
    nr = length(rvals)
    # finite differences for d(rP)/dr at r_idx per (l,m)
    function dr_of_rP_row(row::AbstractVector, rvals)
        if r_idx == 1
            return (rvals[2]*row[2] - rvals[1]*row[1]) / (rvals[2]-rvals[1])
        elseif r_idx == nr
            return (rvals[end]*row[end] - rvals[end-1]*row[end-1]) / (rvals[end]-rvals[end-1])
        else
            return (rvals[r_idx+1]*row[r_idx+1] - rvals[r_idx-1]*row[r_idx-1]) / (rvals[r_idx+1]-rvals[r_idx-1])
        end
    end
    rinv2 = 1.0 / (r*r)
    for (i, ell) in enumerate(l)
        # complex magnitudes
        Tre = T_real[i, r_idx]; Tim = T_imag[i, r_idx]
        Pre = P_real[i, r_idx]; Pim = P_imag[i, r_idx]
        # |T|^2 and |P|^2
        absT2 = Tre*Tre + Tim*Tim
        absP2 = Pre*Pre + Pim*Pim
        # real-valued d(rP)/dr using real part only; include imag as well
        d_rP_re = dr_of_rP_row(view(P_real, i, :), rvals)
        d_rP_im = dr_of_rP_row(view(P_imag, i, :), rvals)
        abs_d_rP2 = d_rP_re*d_rP_re + d_rP_im*d_rP_im
        E[ell+1] += 0.5 * ( ell*(ell+1) * rinv2 * (absT2 + absP2) + abs_d_rP2 )
    end
    return E
end

function maybe_write_csv(outpath::Union{Nothing,String}, l::Vector{Int}, E::Vector{Float64})
    outpath === nothing && return
    open(outpath, "w") do io
        println(io, "l,E_l")
        for ell in 0:maximum(l)
            println(io, "$(ell),$(E[ell+1])")
        end
    end
end

function main()
    path, family, opts = parse_args()
    data = read_spectral(path, family)
    l = data.l; rvals = data.r
    if opts.integrate
        # shell-integrated spectra across r (simple sum over r)
        if data.mode == :scalar
            if opts.axis == :l
                Esum = zeros(Float64, maximum(l)+1)
                for j in eachindex(rvals)
                    Esum .+= spectra_scalar_at_r(l, data.m, data.sca_real, data.sca_imag, j)
                end
                maybe_write_csv(opts.out, l, Esum)
                top = opts.top === nothing ? 0 : opts.top
                println("l, E_l (integrated over r)")
                for ell in 0:maximum(l)
                    println("$(ell), $(Esum[ell+1])")
                    if top>0 && ell+1>=top; break; end
                end
                return
            else
                mmax = maximum(data.m)
                Esum_m = zeros(Float64, mmax+1)
                for j in eachindex(rvals)
                    Esum_m .+= spectra_scalar_m_at_r(l, data.m, data.sca_real, data.sca_imag, j)
                end
                # Print m-spectrum
                println("m, E_m (integrated over r)")
                for mm in 0:mmax
                    println("$(mm), $(Esum_m[mm+1])")
                    if opts.top !== nothing && mm+1>=opts.top; break; end
                end
                return
            end
        else
            Esum = zeros(Float64, maximum(l)+1)
            for j in eachindex(rvals)
                Esum .+= spectra_vector_at_r(l, data.m, data.tor_real, data.tor_imag, data.pol_real, data.pol_imag, rvals, j)
            end
            maybe_write_csv(opts.out, l, Esum)
            top = opts.top === nothing ? 0 : opts.top
            println("l, E_l (integrated over r)")
            for ell in 0:maximum(l)
                println("$(ell), $(Esum[ell+1])")
                if top>0 && ell+1>=top; break; end
            end
            return
        end
    end
    # Single radius
    r_idx = pick_r_index(rvals, opts.rsel)
    if data.mode == :scalar
        if opts.axis == :l
            E = spectra_scalar_at_r(l, data.m, data.sca_real, data.sca_imag, r_idx)
            println("Scalar l-spectrum at r=$(rvals[r_idx])")
            maybe_write_csv(opts.out, l, E)
            top = opts.top === nothing ? 0 : opts.top
            println("l, E_l")
            for ell in 0:maximum(l)
                println("$(ell), $(E[ell+1])")
                if top>0 && ell+1>=top; break; end
            end
        else
            Em = spectra_scalar_m_at_r(l, data.m, data.sca_real, data.sca_imag, r_idx)
            mmax = length(Em)-1
            println("Scalar m-spectrum at r=$(rvals[r_idx])")
            println("m, E_m")
            for mm in 0:mmax
                println("$(mm), $(Em[mm+1])")
                if opts.top !== nothing && mm+1>=opts.top; break; end
            end
        end
    else
        E = spectra_vector_at_r(l, data.m, data.tor_real, data.tor_imag, data.pol_real, data.pol_imag, rvals, r_idx)
        println("Vector spectra at r=$(rvals[r_idx])")
        maybe_write_csv(opts.out, l, E)
        top = opts.top === nothing ? 0 : opts.top
        println("l, E_l")
        for ell in 0:maximum(l)
            println("$(ell), $(E[ell+1])")
            if top>0 && ell+1>=top; break; end
        end
    end
    close(data.ds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
