#!/usr/bin/env julia

# m=0 (axisymmetric) r-θ plot reconstructed from toroidal/poloidal spectral data.
#
# Reads spectral coefficients (e.g., velocity_toroidal_real/imag and velocity_poloidal_real/imag)
# along with l_values, m_values from a NetCDF output. Selects m=0 modes, performs a
# scalar spherical-harmonic synthesis Y_{l0}(θ) to transform toroidal/poloidal scalars
# into physical space over θ for all radii, then plots r–θ heatmaps.
#
# Note: This reconstructs the scalar toroidal/poloidal potentials for m=0. If you need
# vector components (u_r, u_θ, u_φ) from tor/pol, we can extend with vector synthesis,
# but this requires consistent normalization choices. This version uses standard
# normalization Y_{l0}(θ)=sqrt((2l+1)/(4π)) P_l(cosθ).
#
# Usage examples:
#   # Velocity toroidal m=0 (default normalization)
#   julia --project examples/plot_m0_from_torpol.jl out.nc velocity toroidal
#   # Magnetic poloidal m=0 with custom clim
#   julia --project examples/plot_m0_from_torpol.jl out.nc magnetic poloidal --clim -1 1
#   # Show both toroidal and poloidal (two subplots)
#   julia --project examples/plot_m0_from_torpol.jl out.nc velocity both --cmap balance
#   # Disable Ylm normalization (use raw P_l):
#   julia --project examples/plot_m0_from_torpol.jl out.nc velocity toroidal --no-norm

using NCDatasets

mutable struct Opts
    cmap::Union{Nothing,Symbol}
    clim::Union{Nothing,Tuple{Float64,Float64}}
    degrees::Bool
    nonorm::Bool
end

Opts() = Opts(nothing, nothing, true, false)

function parse_args()
    path = get(ARGS, 1, "")
    family = lowercase(get(ARGS, 2, "velocity"))   # "velocity" | "magnetic"
    which  = lowercase(get(ARGS, 3, "both"))       # "toroidal" | "poloidal" | "both"
    extras = ARGS[4:end]
    opts = Opts()
    i = 1
    while i <= length(extras)
        a = extras[i]
        if a == "--cmap" && i+1 <= length(extras)
            opts.cmap = Symbol(extras[i+1]); i += 2
        elseif a == "--clim" && i+2 <= length(extras)
            vmin = parse(Float64, extras[i+1]); vmax = parse(Float64, extras[i+2])
            opts.clim = (vmin, vmax); i += 3
        elseif a == "--degrees"
            opts.degrees = true; i += 1
        elseif a == "--radians"
            opts.degrees = false; i += 1
        elseif a == "--no-norm"
            opts.nonorm = true; i += 1
        else
            error("Unknown/malformed option '$a'. Supported: --cmap NAME --clim MIN MAX --degrees|--radians --no-norm")
        end
    end
    family ∈ ("velocity","magnetic") || error("family must be 'velocity' or 'magnetic'")
    which  ∈ ("toroidal","poloidal","both") || error("component must be 'toroidal','poloidal', or 'both'")
    return path, family, which, opts
end

function ensure_backend()
    local backend
    try
        @eval using GLMakie
        backend = :gl
    catch
        try
            @eval using CairoMakie
            backend = :cairo
        catch
            error("No Makie backend found. Install GLMakie or CairoMakie: import Pkg; Pkg.add(\"GLMakie\")")
        end
    end
    return backend
end

function read_spectral(path::AbstractString, family::String)
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, "theta") || error("theta not found")
        haskey(ds, "r") || error("r not found")
        haskey(ds, "l_values") || error("l_values not found (spectral metadata)")
        haskey(ds, "m_values") || error("m_values not found (spectral metadata)")
        tor_r = "$(family)_toroidal_real"; tor_i = "$(family)_toroidal_imag"
        pol_r = "$(family)_poloidal_real"; pol_i = "$(family)_poloidal_imag"
        haskey(ds, tor_r) || error("$tor_r not found in file")
        haskey(ds, tor_i) || error("$tor_i not found in file")
        haskey(ds, pol_r) || error("$pol_r not found in file")
        haskey(ds, pol_i) || error("$pol_i not found in file")

        theta = vec(ds["theta"][:])
        rvals = vec(ds["r"][:])
        lvals = vec(ds["l_values"][:])
        mvals = vec(ds["m_values"][:])
        tor_real = Array(ds[tor_r][:])
        tor_imag = Array(ds[tor_i][:])
        pol_real = Array(ds[pol_r][:])
        pol_imag = Array(ds[pol_i][:])
        # Expect dimensions (nlm, nr)
        size(tor_real,1) == length(lvals) || @warn "nlm mismatch for toroidal"
        size(pol_real,1) == length(lvals) || @warn "nlm mismatch for poloidal"
        return (theta=theta, r=rvals, l=lvals, m=mvals,
                tor_real=tor_real, tor_imag=tor_imag,
                pol_real=pol_real, pol_imag=pol_imag, ds=ds)
    catch
        close(ds); rethrow()
    end
end

# Compute Legendre P_l(x) for l=0..Lmax at all x via recurrence
function legendre_P_all(x::AbstractVector{<:Real}, Lmax::Int)
    nx = length(x)
    P = Array{Float64}(undef, Lmax+1, nx)
    @inbounds for j in 1:nx
        P[1,j] = 1.0       # l=0
        if Lmax >= 1
            P[2,j] = x[j]  # l=1
        end
    end
    @inbounds for l in 1:(Lmax-1)
        α = (2l + 1) / (l + 1)
        β = l / (l + 1)
        for j in 1:nx
            P[l+2, j] = α * x[j] * P[l+1, j] - β * P[l, j]
        end
    end
    return P  # P[l+1, j] corresponds to P_l(x[j])
end

function synthesize_m0(theta::AbstractVector, lvals::AbstractVector{<:Integer}, mvals::AbstractVector{<:Integer},
                       coeffs::AbstractMatrix; normalize::Bool=true)
    # coeffs is (nlm, nr) for either toroidal or poloidal real-part (complex OK but imag ~ 0 for m=0)
    # Returns matrix F(θ, r) of size (length(theta), nr)
    idxs = findall(==(0), mvals)
    isempty(idxs) && error("No m=0 coefficients found")
    # Determine max l among m=0 modes to size Legendre table
    Lmax = maximum(lvals[idxs])
    x = cos.(theta)
    P = legendre_P_all(x, Lmax)   # (Lmax+1) × nθ
    norm = normalize ? (l -> sqrt((2l+1)/(4π))) : (l -> 1.0)
    nθ = length(theta)
    nr = size(coeffs, 2)
    F = zeros(Float64, nθ, nr)
    @inbounds for (k, i) in enumerate(idxs)
        l = lvals[i]
        Pl = view(P, l+1, :)  # row for degree l
        for j in 1:nr
            c = coeffs[i, j]
            @simd for t in 1:nθ
                F[t, j] += norm(l) * Pl[t] * c
            end
        end
    end
    return F
end

function main()
    path, family, which, opts = parse_args()
    data = read_spectral(path, family)

    # Use real parts (m=0 should be real for real-valued fields)
    Tm0 = synthesize_m0(data.theta, data.l, data.m, data.tor_real; normalize=!opts.nonorm)
    Pm0 = synthesize_m0(data.theta, data.l, data.m, data.pol_real; normalize=!opts.nonorm)

    # Prepare plotting
    backend = ensure_backend()
    if backend == :gl
        using GLMakie
    else
        using CairoMakie
    end

    θplot = opts.degrees ? data.theta .* (180 / pi) : data.theta
    xlab = opts.degrees ? "θ (deg)" : "θ (rad)"

    # Helper to plot a single heatmap
    function plot_hm(figcell, Xθr, title_suffix)
        ax = Axis(figcell; xlabel=xlab, ylabel="r", title="$(family) $(title_suffix) m=0")
        hm_kwargs = (; interpolate=false)
        if opts.cmap !== nothing
            hm_kwargs = merge(hm_kwargs, (; colormap = opts.cmap))
        end
        if opts.clim !== nothing
            hm_kwargs = merge(hm_kwargs, (; colorrange = opts.clim))
        end
        h = heatmap!(ax, θplot, data.r, Xθr; hm_kwargs...)
        Colorbar(parent(figcell), h, label="value")
        return ax
    end

    if which == "both"
        fig = Figure(resolution=(1200, 650))
        plot_hm(fig[1,1], Tm0, "toroidal")
        plot_hm(fig[1,2], Pm0, "poloidal")
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_m0_rtheta_both.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    elseif which == "toroidal"
        fig = Figure(resolution=(900, 650))
        plot_hm(fig[1,1], Tm0, "toroidal")
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_toroidal_m0_rtheta.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    else
        fig = Figure(resolution=(900, 650))
        plot_hm(fig[1,1], Pm0, "poloidal")
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(family)_poloidal_m0_rtheta.png"
        save(outpng, fig); @info "Saved" outpng; display(fig)
    end

    close(data.ds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

