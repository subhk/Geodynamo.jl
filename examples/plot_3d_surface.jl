#!/usr/bin/env julia

# 3D surface rendering of a spherical field using GLMakie.
#
# - Reads `temperature[theta,phi,r]` from a NetCDF output produced by this repo
# - Builds a spherical mesh at a chosen radius and colors by the scalar field
# - Saves a PNG or opens an interactive window depending on backend
#
# Usage examples:
#   julia --project examples/plot_3d_surface.jl path/to/output.nc temperature outer
#   julia --project examples/plot_3d_surface.jl path/to/output.nc temperature 10    # r-index
#   julia --project examples/plot_3d_surface.jl path/to/output.nc temperature 0.95  # r-value (nearest)
#
# Notes:
# - Requires GLMakie (or CairoMakie for headless PNG). Install with:
#     import Pkg; Pkg.add(["GLMakie"])   # or Pkg.add("CairoMakie")
# - Uses existing dependency NCDatasets to read NetCDF files.

using NCDatasets

function parse_args()
    # Defaults: try to find a file in CWD; plot temperature; use outer radius
    path = get(ARGS, 1, "")
    varname = get(ARGS, 2, "temperature")
    rsel = get(ARGS, 3, "outer")  # "outer" | "inner" | Int index | Float radius
    return path, varname, rsel
end

function pick_radius_index(rvals::AbstractVector{<:Real}, rsel)
    if rsel isa AbstractString
        s = lowercase(rsel)
        if s == "outer"
            return length(rvals)
        elseif s == "inner"
            return 1
        else
            # try parse number
            try
                rv = parse(Float64, s)
                # nearest index
                return findmin(abs.(rvals .- rv))[2]
            catch
                error("Unrecognized radius selector: $rsel. Use 'outer', 'inner', index, or radius value.")
            end
        end
    elseif rsel isa Integer
        return Int(rsel)
    elseif rsel isa Real
        rv = float(rsel)
        return findmin(abs.(rvals .- rv))[2]
    else
        error("Unsupported radius selector type: $(typeof(rsel))")
    end
end

function read_field(path::AbstractString, varname::AbstractString)
    if isempty(path)
        error("No NetCDF path provided. Pass the file path as the first argument.")
    end
    isfile(path) || error("File not found: $path")
    ds = NCDataset(path)
    try
        haskey(ds, varname) || error("Variable '$varname' not found in $path")
        var = ds[varname]
        # Expect dims (theta,phi,r) with coordinates present
        haskey(ds, "theta") || error("Coordinate 'theta' not found in $path")
        haskey(ds, "phi")   || error("Coordinate 'phi' not found in $path")
        haskey(ds, "r")     || error("Coordinate 'r' not found in $path")
        theta = vec(ds["theta"][:])
        phi   = vec(ds["phi"][:])
        rvals = vec(ds["r"][:])
        return (ds=ds, var=var, theta=theta, phi=phi, rvals=rvals)
    catch
        close(ds)
        rethrow()
    end
end

function build_sphere_mesh(theta::AbstractVector, phi::AbstractVector, r::Real)
    # Returns (points::Vector{Point3f}, faces::Matrix{Int}) for Makie.mesh
    # Grid is theta in [0,π], phi in [0,2π)
    nθ = length(theta)
    nφ = length(phi)
    # Precompute sin/cos
    sθ = sin.(theta)
    cθ = cos.(theta)
    sφ = sin.(phi)
    cφ = cos.(phi)

    # Vertices
    # index(i,j) -> linear index with j varying fastest
    idx = (i,j) -> (i-1)*nφ + j
    points = Vector{Point3f}()
    sizehint!(points, nθ*nφ)
    for i in 1:nθ, j in 1:nφ
        x = r * sθ[i] * cφ[j]
        y = r * sθ[i] * sφ[j]
        z = r * cθ[i]
        push!(points, Point3f(x, y, z))
    end

    # Faces (two triangles per grid cell), wrap in phi
    faces = Vector{NTuple{3, Int}}()
    sizehint!(faces, 2*(nθ-1)*nφ)
    for i in 1:(nθ-1)
        for j in 1:nφ
            jn = (j % nφ) + 1
            v1 = idx(i, j)
            v2 = idx(i, jn)
            v3 = idx(i+1, j)
            v4 = idx(i+1, jn)
            # triangles (v1,v2,v3) and (v2,v4,v3)
            push!(faces, (v1, v2, v3))
            push!(faces, (v2, v4, v3))
        end
    end
    return points, faces
end

function field_at_radius(var, r_index::Int)
    # Expect var dims = (theta, phi, r)
    # Extract a (nθ, nφ) slice at r_index
    T = Array(var[:, :, r_index])
    return T
end

function ensure_backend()
    # Try GLMakie; fall back to CairoMakie (no interactivity)
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

function main()
    path, varname, rsel_raw = parse_args()
    data = read_field(path, varname)
    r_index = pick_radius_index(data.rvals, rsel_raw)
    @info "Using radius index" r_index value=data.rvals[r_index]

    Tθφ = field_at_radius(data.var, r_index)
    size(Tθφ, 1) == length(data.theta) || @warn "theta dimension mismatch; assuming var dims are (theta,phi,r)"
    size(Tθφ, 2) == length(data.phi)   || @warn "phi dimension mismatch; assuming var dims are (theta,phi,r)"

    points, faces = build_sphere_mesh(data.theta, data.phi, data.rvals[r_index])

    # Flatten colors to match vertex order (i varies slow, j varies fast)
    colors = vec(Tθφ)

    backend = ensure_backend()
    if backend == :gl
        using GLMakie
        fig = Figure(resolution=(900, 700))
        ax = Axis3(fig[1,1], aspect=:data, title="$(varname) at r=$(round(data.rvals[r_index], sigdigits=4))")
        m = mesh!(ax, points, faces; color=colors, shading=true, colormap=:viridis)
        Colorbar(fig[1,2], m, label=varname)
        ax.azimuth[] = 30
        ax.elevation[] = 20
        fig
        display(fig)
        # Optional: save a PNG next to the NetCDF
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(varname)_r$(r_index).png"
        try
            save(outpng, fig)
            @info "Saved" outpng
        catch err
            @warn "Failed to save PNG" exception=(err, catch_backtrace())
        end
        # keep window open in GLMakie if running non-interactively
        if !isinteractive()
            wait(display(fig))
        end
    else
        using CairoMakie
        fig = Figure(resolution=(900, 700))
        ax = Axis3(fig[1,1], aspect=:data, title="$(varname) at r=$(round(data.rvals[r_index], sigdigits=4))")
        m = mesh!(ax, points, faces; color=colors, shading=true, colormap=:viridis)
        Colorbar(fig[1,2], m, label=varname)
        ax.azimuth[] = 30
        ax.elevation[] = 20
        outpng = replace(basename(path), r"\.nc$" => "") * "_$(varname)_r$(r_index).png"
        save(outpng, fig)
        @info "Saved" outpng
    end

    close(data.ds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

