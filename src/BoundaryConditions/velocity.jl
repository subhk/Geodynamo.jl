# ============================================================================
# Velocity Boundary Conditions
# ============================================================================

using SHTnsKit
using PencilArrays
using PencilFFTs
using Statistics

"""
    load_velocity_boundary_conditions!(velocity_field, boundary_specs::Dict)

Load velocity boundary conditions from various sources.

# Arguments
- `velocity_field`: SHTnsVelocityField structure
- `boundary_specs`: Dictionary specifying boundary sources

# Examples
```julia
# No-slip boundaries at both surfaces
boundary_specs = Dict(
    :inner => (:no_slip, 0.0),
    :outer => (:no_slip, 0.0)
)

# Stress-free boundaries
boundary_specs = Dict(
    :inner => (:stress_free, 0.0),
    :outer => (:stress_free, 0.0)
)

# NetCDF file for inner, no-slip for outer
boundary_specs = Dict(
    :inner => "cmb_velocity.nc",
    :outer => (:no_slip, 0.0)
)
```
"""
function load_velocity_boundary_conditions!(velocity_field, boundary_specs::Dict)
    
    if get_rank() == 0
        @info "Loading velocity boundary conditions..."
    end
    
    # Determine boundary types
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    if inner_spec === nothing || outer_spec === nothing
        throw(ArgumentError("Both :inner and :outer boundary specifications required"))
    end
    
    # Load or generate boundary data
    if isa(inner_spec, String) && isa(outer_spec, String)
        # Both from NetCDF files
        boundary_set = load_velocity_boundaries_from_files(inner_spec, outer_spec, velocity_field.config)
    elseif isa(inner_spec, String) && isa(outer_spec, Tuple)
        # Inner from file, outer programmatic
        boundary_set = create_hybrid_velocity_boundaries(inner_spec, outer_spec, velocity_field.config)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, String)
        # Inner programmatic, outer from file
        boundary_set = create_hybrid_velocity_boundaries(outer_spec, inner_spec, velocity_field.config, swap_boundaries=true)
    elseif isa(inner_spec, Tuple) && isa(outer_spec, Tuple)
        # Both programmatic
        boundary_set = create_programmatic_velocity_boundaries(inner_spec, outer_spec, velocity_field.config)
    else
        throw(ArgumentError("Invalid boundary specification format"))
    end
    
    # Store boundary conditions in field
    velocity_field.boundary_condition_set = boundary_set
    velocity_field.boundary_time_index[] = 1
    
    # Create interpolation cache
    velocity_field.boundary_interpolation_cache = create_velocity_interpolation_cache(boundary_set, velocity_field.config)
    
    # Apply initial boundary conditions
    apply_velocity_boundary_conditions!(velocity_field)
    
    if get_rank() == 0
        print_boundary_info(boundary_set)
        @info "Velocity boundary conditions loaded successfully"
    end
    
    return velocity_field
end

"""
    load_velocity_boundaries_from_files(inner_file::String, outer_file::String, config)

Load velocity boundary conditions from NetCDF files.
"""
function load_velocity_boundaries_from_files(inner_file::String, outer_file::String, config)
    
    # Validate files exist
    for file in [inner_file, outer_file]
        if !isfile(file)
            throw(ArgumentError("Velocity boundary file not found: $file"))
        end
    end
    
    # Read boundary data
    inner_data = read_netcdf_boundary_data(inner_file, precision=config.T)
    outer_data = read_netcdf_boundary_data(outer_file, precision=config.T)
    
    # Update field type for velocity
    inner_data.field_type = "velocity"
    outer_data.field_type = "velocity"
    
    # Validate vector field dimensions (should have 3 components: r, θ, φ)
    if inner_data.ncomponents != 3 || outer_data.ncomponents != 3
        throw(ArgumentError("Velocity boundary conditions require 3 components (r, θ, φ)"))
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_hybrid_velocity_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)

Create hybrid velocity boundaries (one from file, one programmatic).
"""
function create_hybrid_velocity_boundaries(file_spec::String, prog_spec::Tuple, config; swap_boundaries=false)
    
    # Load file-based boundary
    file_data = read_netcdf_boundary_data(file_spec, precision=config.T)
    file_data.field_type = "velocity"
    
    # Create programmatic boundary
    pattern, amplitude = prog_spec[1], prog_spec[2]
    parameters = length(prog_spec) >= 3 ? prog_spec[3] : Dict()
    
    prog_data = create_programmatic_velocity_boundary(
        pattern, config, amplitude; parameters=parameters
    )
    
    # Ensure same grid resolution
    if file_data.nlat != config.nlat || file_data.nlon != config.nlon
        # Interpolate file data to config grid
        theta_target = collect(range(0, π, length=config.nlat))
        phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
        
        interpolated_values = interpolate_boundary_to_grid(file_data, theta_target, phi_target, 1)
        
        file_data = create_boundary_data(
            interpolated_values, "velocity";
            theta=theta_target, phi=phi_target, time=nothing,
            units=file_data.units, description=file_data.description,
            file_path=file_data.file_path
        )
    end
    
    # Assign boundaries based on swap_boundaries flag
    if swap_boundaries
        inner_data, outer_data = prog_data, file_data
    else
        inner_data, outer_data = file_data, prog_data
    end
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_velocity_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)

Create fully programmatic velocity boundaries.
"""
function create_programmatic_velocity_boundaries(inner_spec::Tuple, outer_spec::Tuple, config)
    
    # Create inner boundary
    inner_pattern, inner_amplitude = inner_spec[1], inner_spec[2]
    inner_parameters = length(inner_spec) >= 3 ? inner_spec[3] : Dict()
    
    inner_data = create_programmatic_velocity_boundary(
        inner_pattern, config, inner_amplitude; parameters=inner_parameters
    )
    
    # Create outer boundary
    outer_pattern, outer_amplitude = outer_spec[1], outer_spec[2]
    outer_parameters = length(outer_spec) >= 3 ? outer_spec[3] : Dict()
    
    outer_data = create_programmatic_velocity_boundary(
        outer_pattern, config, outer_amplitude; parameters=outer_parameters
    )
    
    # Validate compatibility
    validate_boundary_compatibility(inner_data, outer_data, "velocity")
    
    # Create boundary condition set
    boundary_set = BoundaryConditionSet(
        inner_data, outer_data, "velocity", VELOCITY, time()
    )
    
    return boundary_set
end

"""
    create_programmatic_velocity_boundary(pattern::Symbol, config, amplitude::Real=1.0; 
                                        parameters::Dict=Dict())

Create programmatically generated velocity boundary conditions.

# Available patterns:
- `:no_slip` - Zero velocity at boundary (amplitude ignored)
- `:stress_free` - Zero stress at boundary (amplitude ignored)  
- `:uniform_rotation` - Uniform rotation with angular velocity amplitude
- `:differential_rotation` - Differential rotation pattern
- `:zonal_flow` - Zonal (east-west) flow pattern
- `:meridional_flow` - Meridional (north-south) flow pattern
- `:custom` - User-defined velocity function
"""
function create_programmatic_velocity_boundary(pattern::Symbol, config, amplitude::Real=1.0;
                                             parameters::Dict=Dict())
    
    # Create coordinate grids
    nlat, nlon = config.nlat, config.nlon
    theta = collect(range(0, π, length=nlat))
    phi = collect(range(0, 2π, length=nlon+1)[1:end-1])
    
    # Initialize velocity components array [nlat, nlon, 3] for (v_r, v_θ, v_φ)
    values = zeros(config.T, nlat, nlon, 3)
    
    # Generate velocity pattern
    if pattern == :no_slip
        # All velocity components are zero (already initialized)
        description = "No-slip boundary condition (zero velocity)"
        
    elseif pattern == :stress_free
        # Only radial component is zero for stress-free conditions
        # Tangential components determined by stress-free conditions
        # For simplicity, set all components to zero here
        # (proper stress-free implementation requires solver integration)
        description = "Stress-free boundary condition"
        
    elseif pattern == :uniform_rotation
        # Uniform rotation with angular velocity = amplitude
        omega = amplitude
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0  
                values[i, j, 3] = omega * sin(θ)  # v_φ = ω sin(θ)
            end
        end
        description = "Uniform rotation (ω = $omega rad/s)"
        
    elseif pattern == :differential_rotation
        # Differential rotation: ω(θ) = ω₀ sin²(θ)
        omega0 = amplitude
        for (i, θ) in enumerate(theta)
            omega_theta = omega0 * sin(θ)^2
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0
                values[i, j, 3] = omega_theta * sin(θ)  # v_φ = ω(θ) sin(θ)
            end
        end
        description = "Differential rotation (ω₀ = $omega0 rad/s)"
        
    elseif pattern == :zonal_flow
        # East-west flow pattern: v_φ = amplitude * sin(n*θ)
        n = get(parameters, "wavenumber", 1)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = 0.0  # v_θ = 0
                values[i, j, 3] = amplitude * sin(n * θ)  # v_φ
            end
        end
        description = "Zonal flow (n = $n, amplitude = $amplitude)"
        
    elseif pattern == :meridional_flow
        # North-south flow pattern: v_θ = amplitude * cos(m*φ)
        m = get(parameters, "wavenumber", 1)
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                values[i, j, 1] = 0.0  # v_r = 0
                values[i, j, 2] = amplitude * cos(m * φ)  # v_θ
                values[i, j, 3] = 0.0  # v_φ = 0
            end
        end
        description = "Meridional flow (m = $m, amplitude = $amplitude)"
        
    elseif pattern == :custom
        # User-defined velocity function
        if !haskey(parameters, "function")
            throw(ArgumentError("Custom velocity pattern requires 'function' in parameters"))
        end
        
        user_func = parameters["function"]
        for (i, θ) in enumerate(theta)
            for (j, φ) in enumerate(phi)
                v_r, v_theta, v_phi = user_func(θ, φ)
                values[i, j, 1] = amplitude * v_r
                values[i, j, 2] = amplitude * v_theta
                values[i, j, 3] = amplitude * v_phi
            end
        end
        description = "Custom velocity pattern (amplitude = $amplitude)"
        
    else
        throw(ArgumentError("Unknown velocity pattern: $pattern"))
    end
    
    # Create BoundaryData structure
    return create_boundary_data(
        values, "velocity";
        theta=theta, phi=phi, time=nothing,
        units="m/s",
        description=description,
        file_path="programmatic"
    )
end

"""
    create_velocity_interpolation_cache(boundary_set::BoundaryConditionSet, config)

Create interpolation cache for velocity boundaries.
"""
function create_velocity_interpolation_cache(boundary_set::BoundaryConditionSet, config)
    
    cache = Dict{String, Any}()
    
    # Create target grid (simulation grid)
    theta_target = collect(range(0, π, length=config.nlat))
    phi_target = collect(range(0, 2π, length=config.nlon+1)[1:end-1])
    
    # Create interpolation caches
    cache["inner"] = create_interpolation_cache(boundary_set.inner_boundary, theta_target, phi_target)
    cache["outer"] = create_interpolation_cache(boundary_set.outer_boundary, theta_target, phi_target)
    
    return cache
end

"""
    apply_velocity_boundary_conditions!(velocity_field, time_index::Int=1)

Apply velocity boundary conditions to the field.
"""
function apply_velocity_boundary_conditions!(velocity_field, time_index::Int=1)
    
    if velocity_field.boundary_condition_set === nothing
        @warn "No boundary conditions loaded for velocity field"
        return velocity_field
    end
    
    boundary_set = velocity_field.boundary_condition_set
    cache = velocity_field.boundary_interpolation_cache
    
    # Interpolate boundary data to simulation grid
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Transform to spectral space for each velocity component
    # Toroidal component
    inner_toroidal = physical_to_spectral_boundary(inner_physical[:, :, 1], velocity_field.config)  # v_r component
    outer_toroidal = physical_to_spectral_boundary(outer_physical[:, :, 1], velocity_field.config)
    
    # Poloidal component (combine v_θ and v_φ components)
    inner_poloidal = physical_to_spectral_boundary(
        sqrt.(inner_physical[:, :, 2].^2 + inner_physical[:, :, 3].^2), velocity_field.config
    )
    outer_poloidal = physical_to_spectral_boundary(
        sqrt.(outer_physical[:, :, 2].^2 + outer_physical[:, :, 3].^2), velocity_field.config
    )
    
    # Apply to boundary arrays
    velocity_field.toroidal.boundary_values[1, :] = inner_toroidal  # Inner boundary
    velocity_field.toroidal.boundary_values[2, :] = outer_toroidal  # Outer boundary
    velocity_field.poloidal.boundary_values[1, :] = inner_poloidal
    velocity_field.poloidal.boundary_values[2, :] = outer_poloidal
    
    # Update time index
    velocity_field.boundary_time_index[] = time_index
    
    return velocity_field
end

"""
    update_time_dependent_velocity_boundaries!(velocity_field, current_time::Float64)

Update time-dependent velocity boundary conditions.
"""
function update_time_dependent_velocity_boundaries!(velocity_field, current_time::Float64)
    
    if velocity_field.boundary_condition_set === nothing
        return velocity_field
    end
    
    boundary_set = velocity_field.boundary_condition_set
    
    # Check if boundaries are time-dependent
    if !boundary_set.inner_boundary.is_time_dependent && !boundary_set.outer_boundary.is_time_dependent
        return velocity_field  # Nothing to update
    end
    
    # Find time index for current time
    time_index = find_boundary_time_index(boundary_set, current_time)
    
    # Only update if time index has changed
    if time_index != velocity_field.boundary_time_index[]
        apply_velocity_boundary_conditions!(velocity_field, time_index)
        
        if get_rank() == 0
            @info "Updated velocity boundaries to time index $time_index (t=$current_time)"
        end
    end
    
    return velocity_field
end

"""
    get_current_velocity_boundaries(velocity_field)

Get current velocity boundary conditions.
"""
function get_current_velocity_boundaries(velocity_field)
    
    if velocity_field.boundary_condition_set === nothing
        return Dict(:error => "No boundary conditions loaded")
    end
    
    boundary_set = velocity_field.boundary_condition_set
    time_index = velocity_field.boundary_time_index[]
    cache = velocity_field.boundary_interpolation_cache
    
    # Get current boundary data
    inner_physical = interpolate_with_cache(boundary_set.inner_boundary, cache["inner"], time_index)
    outer_physical = interpolate_with_cache(boundary_set.outer_boundary, cache["outer"], time_index)
    
    # Get spectral coefficients
    inner_toroidal_spectral = velocity_field.toroidal.boundary_values[1, :]
    outer_toroidal_spectral = velocity_field.toroidal.boundary_values[2, :]
    inner_poloidal_spectral = velocity_field.poloidal.boundary_values[1, :]
    outer_poloidal_spectral = velocity_field.poloidal.boundary_values[2, :]
    
    return Dict(
        :inner_physical => inner_physical,
        :outer_physical => outer_physical,
        :inner_toroidal_spectral => inner_toroidal_spectral,
        :outer_toroidal_spectral => outer_toroidal_spectral,
        :inner_poloidal_spectral => inner_poloidal_spectral,
        :outer_poloidal_spectral => outer_poloidal_spectral,
        :time_index => time_index,
        :metadata => Dict(
            "field_name" => boundary_set.field_name,
            "source" => "file_based",
            "inner_file" => boundary_set.inner_boundary.file_path,
            "outer_file" => boundary_set.outer_boundary.file_path,
            "creation_time" => boundary_set.creation_time,
            "components" => ["v_r", "v_theta", "v_phi"]
        )
    )
end

"""
    set_programmatic_velocity_boundaries!(velocity_field, inner_spec::Tuple, outer_spec::Tuple)

Set programmatic velocity boundary conditions.
"""
function set_programmatic_velocity_boundaries!(velocity_field, inner_spec::Tuple, outer_spec::Tuple)
    
    boundary_specs = Dict(:inner => inner_spec, :outer => outer_spec)
    return load_velocity_boundary_conditions!(velocity_field, boundary_specs)
end

"""
    validate_velocity_boundary_files(boundary_specs::Dict, config)

Validate velocity boundary condition files.
"""
function validate_velocity_boundary_files(boundary_specs::Dict, config)
    
    inner_spec = get(boundary_specs, :inner, nothing)
    outer_spec = get(boundary_specs, :outer, nothing)
    
    errors = String[]
    
    # Validate file specifications
    if isa(inner_spec, String)
        try
            validate_netcdf_boundary_file(inner_spec, ["velocity", "u", "v", "w"])
            # Check vector components
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            if inner_data.ncomponents != 3
                push!(errors, "Inner velocity file must have 3 components (v_r, v_theta, v_phi)")
            end
        catch e
            push!(errors, "Inner boundary file error: $e")
        end
    end
    
    if isa(outer_spec, String)
        try
            validate_netcdf_boundary_file(outer_spec, ["velocity", "u", "v", "w"])
            # Check vector components
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            if outer_data.ncomponents != 3
                push!(errors, "Outer velocity file must have 3 components (v_r, v_theta, v_phi)")
            end
        catch e
            push!(errors, "Outer boundary file error: $e")
        end
    end
    
    # If both are files, check compatibility
    if isa(inner_spec, String) && isa(outer_spec, String)
        try
            inner_data = read_netcdf_boundary_data(inner_spec, precision=config.T)
            outer_data = read_netcdf_boundary_data(outer_spec, precision=config.T)
            validate_boundary_compatibility(inner_data, outer_data, "velocity")
        catch e
            push!(errors, "Boundary compatibility error: $e")
        end
    end
    
    if !isempty(errors)
        error_msg = "Velocity boundary validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

export load_velocity_boundary_conditions!, set_programmatic_velocity_boundaries!
export update_time_dependent_velocity_boundaries!, get_current_velocity_boundaries
export validate_velocity_boundary_files, create_programmatic_velocity_boundary