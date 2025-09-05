# ============================================================================
# NetCDF Boundary Condition Reader for Temperature and Compositional Fields
# ============================================================================

using NCDatasets
using Statistics

# ============================================================================
# Data Structures for Boundary Conditions
# ============================================================================

"""
    BoundaryData{T}

Structure to hold boundary condition data read from NetCDF files.
"""
struct BoundaryData{T<:AbstractFloat}
    # Spatial coordinates (if provided)
    theta::Union{Vector{T}, Nothing}      # Colatitude coordinates [rad]
    phi::Union{Vector{T}, Nothing}        # Longitude coordinates [rad]
    
    # Time coordinate (if time-dependent)
    time::Union{Vector{T}, Nothing}       # Time values
    
    # Boundary values
    values::Array{T}                      # Boundary values (nlat, nlon) or (nlat, nlon, ntime)
    
    # Metadata
    units::String                         # Physical units
    description::String                   # Field description
    file_path::String                     # Source file path
    
    # Validation info
    is_time_dependent::Bool               # Whether data varies in time
    nlat::Int                            # Number of latitude points
    nlon::Int                            # Number of longitude points
    ntime::Int                           # Number of time points (1 if time-independent)
end

"""
    BoundaryConditionSet{T}

Complete set of boundary conditions for inner and outer boundaries.
"""
struct BoundaryConditionSet{T<:AbstractFloat}
    inner_boundary::BoundaryData{T}       # Inner boundary (CMB) data
    outer_boundary::BoundaryData{T}       # Outer boundary (surface) data
    field_name::String                    # Field name (e.g., "temperature", "composition")
    creation_time::Float64               # When the boundary set was created
end

# ============================================================================
# NetCDF File Reading Functions
# ============================================================================

"""
    read_netcdf_boundary_data(file_path::String, field_name::String="", 
                             coord_names::Dict=default_coord_names()) -> BoundaryData{T}

Read boundary condition data from a NetCDF file.

# Arguments
- `file_path`: Path to NetCDF file
- `field_name`: Name of the field to read (auto-detected if empty)
- `coord_names`: Dictionary mapping coordinate types to variable names

# Expected NetCDF Structure
The NetCDF file should contain:
- Spatial coordinates: `lat`/`latitude`, `lon`/`longitude` (or `theta`, `phi`)
- Time coordinate (optional): `time`
- Data variable: field values with dimensions (lat, lon) or (lat, lon, time)

# Returns
BoundaryData structure containing the loaded boundary condition data
"""
function read_netcdf_boundary_data(file_path::String, field_name::String=""; 
                                  coord_names::Dict=default_coord_names(),
                                  precision::Type{T}=Float64) where T<:AbstractFloat
    
    if !isfile(file_path)
        throw(ArgumentError("NetCDF file not found: $file_path"))
    end
    
    NCDataset(file_path, "r") do ds
        # Auto-detect field name if not provided
        if isempty(field_name)
            field_name = detect_data_variable(ds)
        end
        
        if !haskey(ds, field_name)
            available_vars = join(keys(ds), ", ")
            throw(ArgumentError("Field '$field_name' not found in file. Available variables: $available_vars"))
        end
        
        # Read coordinate variables
        theta = read_coordinate(ds, coord_names["theta"], precision)
        phi = read_coordinate(ds, coord_names["phi"], precision)
        time_coord = read_coordinate(ds, coord_names["time"], precision)
        
        # Read the main data variable
        data_var = ds[field_name]
        values = Array{T}(data_var[:])
        
        # Get metadata
        units = get(data_var.attrib, "units", "")
        description = get(data_var.attrib, "long_name", get(data_var.attrib, "description", field_name))
        
        # Validate dimensions
        dims = size(values)
        is_time_dependent = length(dims) == 3
        
        nlat = dims[1]
        nlon = dims[2]
        ntime = is_time_dependent ? dims[3] : 1
        
        # Validate coordinate consistency
        validate_coordinates(theta, phi, time_coord, nlat, nlon, ntime, file_path)
        
        return BoundaryData{T}(
            theta, phi, time_coord, values, units, description, file_path,
            is_time_dependent, nlat, nlon, ntime
        )
    end
end

"""
    default_coord_names() -> Dict{String, Vector{String}}

Return default coordinate variable name mappings for NetCDF files.
"""
function default_coord_names()
    return Dict{String, Vector{String}}(
        "theta" => ["theta", "colatitude", "colat", "lat", "latitude"],
        "phi" => ["phi", "longitude", "long", "lon"],
        "time" => ["time", "t", "time_index"]
    )
end

"""
    detect_data_variable(ds::NCDataset) -> String

Auto-detect the main data variable in a NetCDF dataset.
"""
function detect_data_variable(ds::NCDataset)
    coord_names = ["lat", "latitude", "lon", "longitude", "theta", "phi", "time", "t"]
    
    # Look for variables that are not coordinates
    data_vars = String[]
    for (name, var) in ds
        if !(name in coord_names) && ndims(var) >= 2
            push!(data_vars, name)
        end
    end
    
    if isempty(data_vars)
        throw(ArgumentError("No suitable data variable found in NetCDF file"))
    elseif length(data_vars) == 1
        return data_vars[1]
    else
        # Prefer common field names
        preferred_names = ["temperature", "temp", "T", "composition", "comp", "C", "xi"]
        for pref_name in preferred_names
            for data_var in data_vars
                if occursin(pref_name, lowercase(data_var))
                    return data_var
                end
            end
        end
        # Return the first one if no preferred name found
        @warn "Multiple data variables found: $(join(data_vars, ", ")). Using: $(data_vars[1])"
        return data_vars[1]
    end
end

"""
    read_coordinate(ds::NCDataset, coord_names::Vector{String}, ::Type{T}) -> Union{Vector{T}, Nothing}

Read a coordinate variable from NetCDF dataset.
"""
function read_coordinate(ds::NCDataset, coord_names::Vector{String}, ::Type{T}) where T
    for name in coord_names
        if haskey(ds, name)
            coord_data = ds[name][:]
            return Vector{T}(coord_data)
        end
    end
    return nothing
end

"""
    validate_coordinates(theta, phi, time_coord, nlat, nlon, ntime, file_path)

Validate that coordinate dimensions are consistent with data dimensions.
"""
function validate_coordinates(theta, phi, time_coord, nlat, nlon, ntime, file_path)
    errors = String[]
    
    if theta !== nothing && length(theta) != nlat
        push!(errors, "Theta coordinate length ($(length(theta))) != data nlat ($nlat)")
    end
    
    if phi !== nothing && length(phi) != nlon
        push!(errors, "Phi coordinate length ($(length(phi))) != data nlon ($nlon)")
    end
    
    if time_coord !== nothing && ntime > 1 && length(time_coord) != ntime
        push!(errors, "Time coordinate length ($(length(time_coord))) != data ntime ($ntime)")
    end
    
    if !isempty(errors)
        error_msg = "Coordinate validation failed for file $file_path:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
end

# ============================================================================
# High-Level Boundary Condition Loading
# ============================================================================

"""
    load_temperature_boundaries(inner_file::String, outer_file::String; 
                               precision::Type{T}=Float64) -> BoundaryConditionSet{T}

Load temperature boundary conditions from separate inner and outer NetCDF files.

# Arguments
- `inner_file`: Path to NetCDF file containing inner boundary (CMB) temperature data
- `outer_file`: Path to NetCDF file containing outer boundary (surface) temperature data
- `precision`: Floating point precision for the data

# Returns
BoundaryConditionSet containing both inner and outer temperature boundary data

# Example
```julia
temp_bc = load_temperature_boundaries("cmb_temp.nc", "surface_temp.nc")
```
"""
function load_temperature_boundaries(inner_file::String, outer_file::String; 
                                   precision::Type{T}=Float64) where T<:AbstractFloat
    
    # Load inner boundary data
    @info "Loading inner boundary temperature from: $inner_file"
    inner_data = read_netcdf_boundary_data(inner_file, "", precision=precision)
    
    # Load outer boundary data
    @info "Loading outer boundary temperature from: $outer_file"
    outer_data = read_netcdf_boundary_data(outer_file, "", precision=precision)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "temperature")
    
    return BoundaryConditionSet{T}(
        inner_data, outer_data, "temperature", time()
    )
end

"""
    load_composition_boundaries(inner_file::String, outer_file::String; 
                              precision::Type{T}=Float64) -> BoundaryConditionSet{T}

Load compositional boundary conditions from separate inner and outer NetCDF files.

# Arguments
- `inner_file`: Path to NetCDF file containing inner boundary composition data
- `outer_file`: Path to NetCDF file containing outer boundary composition data
- `precision`: Floating point precision for the data

# Returns
BoundaryConditionSet containing both inner and outer compositional boundary data

# Example
```julia
comp_bc = load_composition_boundaries("cmb_composition.nc", "surface_composition.nc")
```
"""
function load_composition_boundaries(inner_file::String, outer_file::String; 
                                   precision::Type{T}=Float64) where T<:AbstractFloat
    
    # Load inner boundary data
    @info "Loading inner boundary composition from: $inner_file"
    inner_data = read_netcdf_boundary_data(inner_file, "", precision=precision)
    
    # Load outer boundary data  
    @info "Loading outer boundary composition from: $outer_file"
    outer_data = read_netcdf_boundary_data(outer_file, "", precision=precision)
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "composition")
    
    return BoundaryConditionSet{T}(
        inner_data, outer_data, "composition", time()
    )
end

"""
    validate_boundary_compatibility(inner::BoundaryData, outer::BoundaryData, field_name::String)

Validate that inner and outer boundary data are compatible.
"""
function validate_boundary_compatibility(inner::BoundaryData, outer::BoundaryData, field_name::String)
    errors = String[]
    
    # Check spatial resolution
    if inner.nlat != outer.nlat
        push!(errors, "Mismatched nlat: inner=$(inner.nlat), outer=$(outer.nlat)")
    end
    
    if inner.nlon != outer.nlon
        push!(errors, "Mismatched nlon: inner=$(inner.nlon), outer=$(outer.nlon)")
    end
    
    # Check time dependency
    if inner.is_time_dependent != outer.is_time_dependent
        push!(errors, "Time dependency mismatch: inner=$(inner.is_time_dependent), outer=$(outer.is_time_dependent)")
    end
    
    if inner.is_time_dependent && inner.ntime != outer.ntime
        push!(errors, "Mismatched ntime: inner=$(inner.ntime), outer=$(outer.ntime)")
    end
    
    # Check coordinate consistency (if both have coordinates)
    if inner.theta !== nothing && outer.theta !== nothing
        if !isapprox(inner.theta, outer.theta, rtol=1e-10)
            push!(errors, "Theta coordinates don't match between files")
        end
    end
    
    if inner.phi !== nothing && outer.phi !== nothing
        if !isapprox(inner.phi, outer.phi, rtol=1e-10)
            push!(errors, "Phi coordinates don't match between files")
        end
    end
    
    if !isempty(errors)
        error_msg = "$field_name boundary compatibility validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    @info "$field_name boundary files are compatible ($(inner.nlat)×$(inner.nlon), time_dep=$(inner.is_time_dependent))"
end

# ============================================================================
# Boundary Data Interpolation and Application
# ============================================================================

"""
    interpolate_boundary_to_grid(boundary_data::BoundaryData{T}, 
                               target_theta::Vector{T}, target_phi::Vector{T},
                               time_index::Int=1) -> Matrix{T}

Interpolate boundary data to a target grid.

# Arguments
- `boundary_data`: Source boundary data
- `target_theta`: Target colatitude coordinates [rad]  
- `target_phi`: Target longitude coordinates [rad]
- `time_index`: Time index for time-dependent data

# Returns
Interpolated boundary values on target grid (nlat_target, nlon_target)
"""
function interpolate_boundary_to_grid(boundary_data::BoundaryData{T}, 
                                    target_theta::Vector{T}, target_phi::Vector{T},
                                    time_index::Int=1) where T<:AbstractFloat
    
    if time_index < 1 || time_index > boundary_data.ntime
        throw(BoundsError("time_index=$time_index out of range [1, $(boundary_data.ntime)]"))
    end
    
    # Extract data for the specified time
    if boundary_data.is_time_dependent
        source_values = boundary_data.values[:, :, time_index]
    else
        source_values = boundary_data.values
    end
    
    # If no coordinates provided, assume data is already on correct grid
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        if size(source_values) == (length(target_theta), length(target_phi))
            return source_values
        else
            @warn "No coordinates in boundary data and size mismatch. Returning raw data."
            return source_values
        end
    end
    
    # Perform bilinear interpolation
    return bilinear_interpolation(boundary_data.theta, boundary_data.phi, source_values,
                                target_theta, target_phi)
end

"""
    bilinear_interpolation(src_theta, src_phi, src_values, target_theta, target_phi) -> Matrix{T}

Perform bilinear interpolation from source grid to target grid.
"""
function bilinear_interpolation(src_theta::Vector{T}, src_phi::Vector{T}, src_values::Matrix{T},
                               target_theta::Vector{T}, target_phi::Vector{T}) where T<:AbstractFloat
    # True bilinear interpolation with cached bracketing indices and weights
    nlat_target = length(target_theta)
    nlon_target = length(target_phi)
    result = zeros(T, nlat_target, nlon_target)

    (θ0, θ1, aθ, ϕ0, ϕ1, aϕ) = _get_bilinear_weights(src_theta, src_phi, target_theta, target_phi)

    @inbounds for i in 1:nlat_target
        i0 = θ0[i]; i1 = θ1[i]; α = aθ[i]
        for j in 1:nlon_target
            j0 = ϕ0[j]; j1 = ϕ1[j]; β = aϕ[j]
            f00 = src_values[i0, j0]
            f01 = src_values[i0, j1]
            f10 = src_values[i1, j0]
            f11 = src_values[i1, j1]
            # Bilinear blend
            result[i, j] = (1-α) * ((1-β)*f00 + β*f01) + α * ((1-β)*f10 + β*f11)
        end
    end
    return result
end

# Cache of bilinear weights for (src_theta, src_phi, target_theta, target_phi)
const _BILIN_CACHE = IdDict{Tuple{Any,Any,Any,Any}, Tuple{Vector{Int},Vector{Int},Vector{Float64},Vector{Int},Vector{Int},Vector{Float64}}}()

function _get_bilinear_weights(src_theta::Vector{T}, src_phi::Vector{T},
                               target_theta::Vector{T}, target_phi::Vector{T}) where T
    key = (src_theta, src_phi, target_theta, target_phi)
    weights = get(_BILIN_CACHE, key, nothing)
    if weights === nothing
        θ0, θ1, aθ = _build_axis_weights(src_theta, target_theta; periodic=false, period=zero(T))
        ϕ0, ϕ1, aϕ = _build_axis_weights(src_phi,   target_phi;  periodic=true,  period=2π)
        weights = (θ0, θ1, aθ, ϕ0, ϕ1, aϕ)
        _BILIN_CACHE[key] = weights
    end
    return weights
end

function _build_axis_weights(src::Vector{T}, tgt::Vector{T}; periodic::Bool, period) where T
    n = length(src)
    i0 = similar(tgt, Int)
    i1 = similar(tgt, Int)
    a  = similar(tgt, Float64)
    # Ensure monotonic increasing; if not, fall back to nearest
    inc = all(diff(src) .>= 0)
    if !inc
        @inbounds for k in eachindex(tgt)
            x = tgt[k]
            # nearest neighbor
            _, idx = findmin(abs.(src .- x))
            i0[k] = clamp(idx, 1, max(1, n-1))
            i1[k] = min(n, i0[k]+1)
            a[k]  = 0.0
        end
        return i0, i1, a
    end
    @inbounds for k in eachindex(tgt)
        x = tgt[k]
        if periodic
            # Wrap into [src[1], src[end]) assuming uniform 0..2π-like
            span = period
            # Map x near domain
            xw = x
            while xw < src[1]; xw += span; end
            while xw >= src[end]; xw -= span; end
            pos = searchsortedfirst(src, xw)
            if pos == 1
                i0[k] = n
                i1[k] = 1
                dx = (src[1] + span) - src[n]
                a[k] = dx ≈ 0 ? 0.0 : (xw - src[n]) / dx
            else
                i0[k] = pos - 1
                i1[k] = pos <= n ? pos : n
                dx = src[i1[k]] - src[i0[k]]
                a[k] = dx ≈ 0 ? 0.0 : (xw - src[i0[k]]) / dx
            end
        else
            # Clamp within bounds
            if x <= src[1]
                i0[k] = 1; i1[k] = 2; a[k] = 0.0
            elseif x >= src[end]
                i0[k] = n-1; i1[k] = n; a[k] = 1.0
            else
                pos = searchsortedfirst(src, x)
                i0[k] = pos - 1
                i1[k] = pos
                dx = src[i1[k]] - src[i0[k]]
                a[k] = dx ≈ 0 ? 0.0 : (x - src[i0[k]]) / dx
            end
        end
    end
    return i0, i1, a
end

"""
    get_boundary_statistics(boundary_data::BoundaryData) -> Dict{String, Any}

Get statistical information about boundary data.
"""
function get_boundary_statistics(boundary_data::BoundaryData)
    values = boundary_data.values
    
    stats = Dict{String, Any}(
        "min" => minimum(values),
        "max" => maximum(values), 
        "mean" => mean(values),
        "std" => std(values),
        "shape" => size(values),
        "units" => boundary_data.units,
        "description" => boundary_data.description,
        "time_dependent" => boundary_data.is_time_dependent,
        "ntime" => boundary_data.ntime,
        "file_path" => boundary_data.file_path
    )
    
    return stats
end

"""
    print_boundary_info(boundary_set::BoundaryConditionSet)

Print detailed information about a boundary condition set.
"""
function print_boundary_info(boundary_set::BoundaryConditionSet)
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║                  $(uppercase(boundary_set.field_name)) BOUNDARY CONDITIONS                  ║")
    println("╠══════════════════════════════════════════════════════════════╣")
    
    println("║ INNER BOUNDARY (CMB):                                        ║")
    print_boundary_data_info(boundary_set.inner_boundary, "║ ")
    
    println("║                                                              ║")
    println("║ OUTER BOUNDARY (SURFACE):                                    ║") 
    print_boundary_data_info(boundary_set.outer_boundary, "║ ")
    
    println("╚══════════════════════════════════════════════════════════════╝")
end

function print_boundary_data_info(boundary_data::BoundaryData, prefix::String)
    println("$(prefix)Grid: $(boundary_data.nlat) × $(boundary_data.nlon)                             ║")
    println("$(prefix)Time steps: $(boundary_data.ntime)                                      ║")
    println("$(prefix)Units: $(boundary_data.units)                                         ║")
    
    values = boundary_data.values
    println("$(prefix)Range: [$(round(minimum(values), digits=3)), $(round(maximum(values), digits=3))]                        ║")
    println("$(prefix)Mean: $(round(mean(values), digits=3))                                           ║")
    
    # Extract filename from path
    filename = basename(boundary_data.file_path)
    println("$(prefix)File: $(filename)                                      ║")
end

# ============================================================================
# Hybrid Boundary Conditions - Mix NetCDF and Programmatic Boundaries
# ============================================================================

"""
    load_single_temperature_boundary(file_path::String, boundary_type::Symbol; 
                                    precision::Type{T}=Float64) -> BoundaryData{T}

Load a single temperature boundary from NetCDF file.

# Arguments
- `file_path`: Path to NetCDF file
- `boundary_type`: Either `:inner` or `:outer` 
- `precision`: Floating point precision

# Returns
BoundaryData for the specified boundary

# Example
```julia
inner_boundary = load_single_temperature_boundary("cmb_temp.nc", :inner)
```
"""
function load_single_temperature_boundary(file_path::String, boundary_type::Symbol; 
                                        precision::Type{T}=Float64) where T<:AbstractFloat
    
    if !(boundary_type in [:inner, :outer])
        throw(ArgumentError("boundary_type must be :inner or :outer"))
    end
    
    @info "Loading $(boundary_type) temperature boundary from: $file_path"
    boundary_data = read_netcdf_boundary_data(file_path, "", precision=precision)
    
    return boundary_data
end

"""
    load_single_composition_boundary(file_path::String, boundary_type::Symbol; 
                                    precision::Type{T}=Float64) -> BoundaryData{T}

Load a single compositional boundary from NetCDF file.

# Arguments
- `file_path`: Path to NetCDF file
- `boundary_type`: Either `:inner` or `:outer`
- `precision`: Floating point precision

# Returns
BoundaryData for the specified boundary

# Example
```julia
outer_boundary = load_single_composition_boundary("surface_comp.nc", :outer)
```
"""
function load_single_composition_boundary(file_path::String, boundary_type::Symbol; 
                                        precision::Type{T}=Float64) where T<:AbstractFloat
    
    if !(boundary_type in [:inner, :outer])
        throw(ArgumentError("boundary_type must be :inner or :outer"))
    end
    
    @info "Loading $(boundary_type) composition boundary from: $file_path"
    boundary_data = read_netcdf_boundary_data(file_path, "", precision=precision)
    
    return boundary_data
end

"""
    create_programmatic_boundary(pattern::Symbol, config::SHTnsKitConfig;
                                amplitude::T=T(1.0), parameters::Dict=Dict()) -> BoundaryData{T}

Create boundary conditions programmatically using predefined patterns.

# Arguments
- `pattern`: Pattern type (:uniform, :y11, :plume, :hemisphere, :custom)
- `config`: SHTns configuration for grid information
- `amplitude`: Pattern amplitude
- `parameters`: Additional pattern-specific parameters

# Available Patterns
- `:uniform`: Uniform value (amplitude)
- `:y11`: Y₁₁ spherical harmonic pattern  
- `:plume`: Gaussian plume pattern
- `:hemisphere`: Hemispherical pattern
- `:dipole`: Dipolar pattern
- `:quadrupole`: Quadrupolar pattern
- `:checkerboard`: Alternating pattern
- `:custom`: User-defined function in parameters["function"]

# Example
```julia
# Create Y₁₁ pattern for inner boundary
inner_boundary = create_programmatic_boundary(:y11, config, amplitude=200.0)

# Create custom pattern
custom_func(theta, phi) = sin(theta) * cos(2*phi)  # Custom function
outer_boundary = create_programmatic_boundary(:custom, config, 
                                            amplitude=50.0, 
                                            parameters=Dict("function" => custom_func))
```
"""
function create_programmatic_boundary(pattern::Symbol, config::SHTnsKitConfig;
                                    amplitude::T=T(1.0),
                                    parameters::Dict=Dict(),
                                    units::String="dimensionless",
                                    description::String="Programmatically generated boundary",
                                    time_dependent::Bool=false,
                                    ntime::Int=1) where T<:AbstractFloat
    
    nlat, nlon = config.nlat, config.nlon
    
    # Create coordinate grids
    theta, phi = create_shtns_coordinate_grids(config)
    
    # Initialize boundary values array
    if time_dependent
        values = zeros(T, nlat, nlon, ntime)
    else
        values = zeros(T, nlat, nlon)
        ntime = 1
    end
    
    # Generate pattern based on type
    for time_idx in 1:ntime
        if time_dependent
            # Time-dependent patterns can use time_idx for evolution
            time_factor = get(parameters, "time_factor", 0.0) * (time_idx - 1)
        else
            time_factor = 0.0
        end
        
        for (i, th) in enumerate(theta)
            for (j, ph) in enumerate(phi)
                value = generate_boundary_pattern(pattern, th, ph, amplitude, parameters, time_factor)
                
                if time_dependent
                    values[i, j, time_idx] = value
                else
                    values[i, j] = value
                end
            end
        end
    end
    
    # Create time coordinate if time-dependent
    time_coord = time_dependent ? collect(T, range(0, 1, length=ntime)) : nothing
    
    return BoundaryData{T}(
        theta, phi, time_coord, values, units, description, "programmatic",
        time_dependent, nlat, nlon, ntime
    )
end

"""
    generate_boundary_pattern(pattern::Symbol, theta, phi, amplitude, parameters, time_factor)

Generate specific boundary pattern at given coordinates.
"""
function generate_boundary_pattern(pattern::Symbol, theta::T, phi::T, amplitude::T, 
                                 parameters::Dict, time_factor::T) where T<:AbstractFloat
    
    if pattern == :uniform
        return amplitude
        
    elseif pattern == :y11
        # Y₁₁ spherical harmonic (real part)
        return amplitude * sin(theta) * cos(phi + time_factor)
        
    elseif pattern == :y20
        # Y₂₀ spherical harmonic (zonal)
        return amplitude * 0.5 * (3*cos(theta)^2 - 1)
        
    elseif pattern == :y22
        # Y₂₂ spherical harmonic
        return amplitude * sin(theta)^2 * cos(2*phi + time_factor)
        
    elseif pattern == :plume
        # Gaussian plume pattern
        center_theta = get(parameters, "center_theta", π/2)
        center_phi = get(parameters, "center_phi", 0.0)
        width = get(parameters, "width", π/4)
        
        # Angular distance from plume center
        angular_dist = acos(cos(theta)*cos(center_theta) + 
                           sin(theta)*sin(center_theta)*cos(phi - center_phi - time_factor))
        
        return amplitude * exp(-(angular_dist/width)^2)
        
    elseif pattern == :hemisphere
        # Hemispherical pattern
        hemisphere_axis = get(parameters, "axis", "z")  # x, y, or z axis
        
        if hemisphere_axis == "z"
            return amplitude * max(0.0, cos(theta))
        elseif hemisphere_axis == "x"  
            return amplitude * max(0.0, sin(theta) * cos(phi + time_factor))
        elseif hemisphere_axis == "y"
            return amplitude * max(0.0, sin(theta) * sin(phi + time_factor))
        end
        
    elseif pattern == :dipole
        # Dipolar pattern (like Y₁₀)
        return amplitude * cos(theta)
        
    elseif pattern == :quadrupole
        # Quadrupolar pattern
        return amplitude * sin(theta)^2 * cos(2*phi + time_factor)
        
    elseif pattern == :checkerboard
        # Checkerboard pattern
        nblocks_theta = get(parameters, "nblocks_theta", 4)
        nblocks_phi = get(parameters, "nblocks_phi", 8)
        
        block_theta = floor(Int, theta / π * nblocks_theta)
        block_phi = floor(Int, (phi + time_factor) / (2π) * nblocks_phi)
        
        return amplitude * ((-1)^(block_theta + block_phi))
        
    elseif pattern == :custom
        # User-defined custom function
        if haskey(parameters, "function")
            custom_func = parameters["function"]
            return amplitude * custom_func(theta, phi + time_factor)
        else
            throw(ArgumentError("Custom pattern requires 'function' parameter"))
        end
        
    else
        throw(ArgumentError("Unknown pattern: $pattern"))
    end
end

"""
    create_hybrid_temperature_boundaries(inner_spec, outer_spec, config::SHTnsKitConfig; precision::Type{T}=Float64)

Create temperature boundary set with mix of NetCDF and programmatic boundaries.

# Arguments
- `inner_spec`: Inner boundary specification (NetCDF file path or (:pattern, parameters...))
- `outer_spec`: Outer boundary specification (NetCDF file path or (:pattern, parameters...))
- `config`: SHTns configuration
- `precision`: Floating point precision

# Examples
```julia
# NetCDF inner, programmatic outer
temp_bc = create_hybrid_temperature_boundaries("cmb_temp.nc", 
                                              (:uniform, 300.0), config)

# Programmatic inner, NetCDF outer  
temp_bc = create_hybrid_temperature_boundaries((:y11, 4000.0, Dict("amplitude" => 200.0)),
                                              "surface_temp.nc", config)

# Both programmatic
temp_bc = create_hybrid_temperature_boundaries((:plume, 4200.0, Dict("width" => π/6)),
                                              (:uniform, 300.0), config)
```
"""
function create_hybrid_temperature_boundaries(inner_spec, outer_spec, config::SHTnsKitConfig; 
                                            precision::Type{T}=Float64) where T<:AbstractFloat
    
    # Load/create inner boundary
    if isa(inner_spec, String)
        # NetCDF file
        inner_data = load_single_temperature_boundary(inner_spec, :inner, precision=precision)
    elseif isa(inner_spec, Tuple)
        # Programmatic specification
        pattern, amplitude = inner_spec[1], inner_spec[2]
        parameters = length(inner_spec) > 2 ? inner_spec[3] : Dict()
        inner_data = create_programmatic_boundary(pattern, config, 
                                                amplitude=T(amplitude), 
                                                parameters=parameters,
                                                units="K",
                                                description="Inner boundary temperature")
    else
        throw(ArgumentError("inner_spec must be NetCDF file path (String) or pattern tuple"))
    end
    
    # Load/create outer boundary
    if isa(outer_spec, String)
        # NetCDF file
        outer_data = load_single_temperature_boundary(outer_spec, :outer, precision=precision)
    elseif isa(outer_spec, Tuple)
        # Programmatic specification
        pattern, amplitude = outer_spec[1], outer_spec[2]
        parameters = length(outer_spec) > 2 ? outer_spec[3] : Dict()
        outer_data = create_programmatic_boundary(pattern, config,
                                                amplitude=T(amplitude),
                                                parameters=parameters, 
                                                units="K",
                                                description="Outer boundary temperature")
    else
        throw(ArgumentError("outer_spec must be NetCDF file path (String) or pattern tuple"))
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "temperature")
    
    return BoundaryConditionSet{T}(
        inner_data, outer_data, "temperature", time()
    )
end

"""
    create_hybrid_composition_boundaries(inner_spec, outer_spec, config::SHTnsKitConfig; precision::Type{T}=Float64)

Create compositional boundary set with mix of NetCDF and programmatic boundaries.

# Arguments
- `inner_spec`: Inner boundary specification (NetCDF file path or (:pattern, parameters...))
- `outer_spec`: Outer boundary specification (NetCDF file path or (:pattern, parameters...))
- `config`: SHTns configuration
- `precision`: Floating point precision

# Examples
```julia
# NetCDF inner, programmatic outer (uniform zero composition at surface)
comp_bc = create_hybrid_composition_boundaries("cmb_composition.nc", 
                                              (:uniform, 0.0), config)

# Programmatic plume pattern inner, NetCDF outer
comp_bc = create_hybrid_composition_boundaries((:plume, 0.8, Dict("width" => π/8)),
                                              "surface_composition.nc", config)
```
"""
function create_hybrid_composition_boundaries(inner_spec, outer_spec, config::SHTnsKitConfig;
                                            precision::Type{T}=Float64) where T<:AbstractFloat
    
    # Load/create inner boundary
    if isa(inner_spec, String)
        inner_data = load_single_composition_boundary(inner_spec, :inner, precision=precision)
    elseif isa(inner_spec, Tuple)
        pattern, amplitude = inner_spec[1], inner_spec[2]
        parameters = length(inner_spec) > 2 ? inner_spec[3] : Dict()
        inner_data = create_programmatic_boundary(pattern, config,
                                                amplitude=T(amplitude),
                                                parameters=parameters,
                                                units="dimensionless",
                                                description="Inner boundary composition")
    else
        throw(ArgumentError("inner_spec must be NetCDF file path (String) or pattern tuple"))
    end
    
    # Load/create outer boundary  
    if isa(outer_spec, String)
        outer_data = load_single_composition_boundary(outer_spec, :outer, precision=precision)
    elseif isa(outer_spec, Tuple)
        pattern, amplitude = outer_spec[1], outer_spec[2]
        parameters = length(outer_spec) > 2 ? outer_spec[3] : Dict()
        outer_data = create_programmatic_boundary(pattern, config,
                                                amplitude=T(amplitude),
                                                parameters=parameters,
                                                units="dimensionless", 
                                                description="Outer boundary composition")
    else
        throw(ArgumentError("outer_spec must be NetCDF file path (String) or pattern tuple"))
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "composition")
    
    return BoundaryConditionSet{T}(
        inner_data, outer_data, "composition", time()
    )
end

"""
    create_time_dependent_programmatic_boundary(pattern::Symbol, config::SHTnsKitConfig, 
                                               time_span::Tuple{T,T}, ntime::Int;
                                               amplitude::T=T(1.0), parameters::Dict=Dict()) where T

Create time-dependent programmatic boundary conditions.

# Arguments  
- `pattern`: Base pattern type
- `config`: SHTns configuration
- `time_span`: (start_time, end_time) tuple
- `ntime`: Number of time steps
- `amplitude`: Pattern amplitude
- `parameters`: Pattern parameters including time evolution

# Time Evolution Parameters
- `"time_factor"`: Rate of pattern rotation/evolution
- `"amplitude_evolution"`: Function for time-varying amplitude
- `"pattern_evolution"`: Function for time-varying pattern parameters

# Example
```julia
# Rotating plume pattern
rotating_plume = create_time_dependent_programmatic_boundary(
    :plume, config, (0.0, 1.0), 100,
    amplitude=500.0,
    parameters=Dict(
        "width" => π/6,
        "center_theta" => π/3,
        "center_phi" => 0.0,
        "time_factor" => 2π  # One full rotation over time span
    )
)
```
"""
function create_time_dependent_programmatic_boundary(pattern::Symbol, config::SHTnsKitConfig,
                                                   time_span::Tuple{T,T}, ntime::Int;
                                                   amplitude::T=T(1.0), 
                                                   parameters::Dict=Dict(),
                                                   units::String="dimensionless",
                                                   description::String="Time-dependent programmatic boundary") where T<:AbstractFloat
    
    # Create time-dependent boundary
    time_dependent_params = copy(parameters)
    time_dependent_params["time_factor"] = get(parameters, "time_factor", 0.0)
    
    return create_programmatic_boundary(pattern, config,
                                      amplitude=amplitude,
                                      parameters=time_dependent_params,
                                      units=units,
                                      description=description,
                                      time_dependent=true,
                                      ntime=ntime)
end

# Export functions
export BoundaryData, BoundaryConditionSet
export read_netcdf_boundary_data, load_temperature_boundaries, load_composition_boundaries
export interpolate_boundary_to_grid, get_boundary_statistics, print_boundary_info
export validate_boundary_compatibility

# Export hybrid boundary functions
export load_single_temperature_boundary, load_single_composition_boundary
export create_programmatic_boundary, create_hybrid_temperature_boundaries, create_hybrid_composition_boundaries
export create_time_dependent_programmatic_boundary
