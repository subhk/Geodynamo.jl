# ============================================================================
# Grid Interpolation Utilities for Boundary Conditions
# ============================================================================

using LinearAlgebra

"""
    interpolate_boundary_to_grid(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                target_phi::Vector{T}, time_index::Int=1) where T

Interpolate boundary data to a target grid using bilinear interpolation.
"""
function interpolate_boundary_to_grid(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                    target_phi::Vector{T}, time_index::Int=1) where T
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        throw(ArgumentError("Source boundary data must have coordinate information for interpolation"))
    end
    
    # Get source coordinates and data
    src_theta = boundary_data.theta
    src_phi = boundary_data.phi
    
    # Extract data for the specified time index
    if boundary_data.is_time_dependent
        if time_index > boundary_data.ntime
            time_index = boundary_data.ntime  # Clamp to available range
        end
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values[:, :, time_index]
        else
            src_data = boundary_data.values[:, :, time_index, :]
        end
    else
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values
        else
            src_data = boundary_data.values
        end
    end
    
    # Initialize output array
    if boundary_data.ncomponents == 1
        interpolated = zeros(T, length(target_theta), length(target_phi))
    else
        interpolated = zeros(T, length(target_theta), length(target_phi), boundary_data.ncomponents)
    end
    
    # Perform bilinear interpolation
    for (i, theta_t) in enumerate(target_theta)
        for (j, phi_t) in enumerate(target_phi)
            
            # Find surrounding points in source grid
            theta_idx = find_grid_indices(src_theta, theta_t)
            phi_idx = find_grid_indices(src_phi, phi_t)
            
            # Get interpolation weights
            theta_weights = get_interpolation_weights(src_theta, theta_t, theta_idx)
            phi_weights = get_interpolation_weights(src_phi, phi_t, phi_idx)
            
            # Perform bilinear interpolation
            if boundary_data.ncomponents == 1
                interpolated[i, j] = bilinear_interpolate(
                    src_data, theta_idx, phi_idx, theta_weights, phi_weights
                )
            else
                for comp in 1:boundary_data.ncomponents
                    interpolated[i, j, comp] = bilinear_interpolate(
                        src_data[:, :, comp], theta_idx, phi_idx, theta_weights, phi_weights
                    )
                end
            end
        end
    end
    
    return interpolated
end

"""
    find_grid_indices(coords::Vector{T}, target::T) where T

Find the two surrounding indices in a coordinate array for interpolation.
"""
function find_grid_indices(coords::Vector{T}, target::T) where T
    n = length(coords)
    
    # Handle edge cases
    if target <= coords[1]
        return (1, 1)
    elseif target >= coords[end]
        return (n, n)
    end
    
    # Binary search for surrounding indices
    low, high = 1, n
    while high - low > 1
        mid = (low + high) ÷ 2
        if coords[mid] <= target
            low = mid
        else
            high = mid
        end
    end
    
    return (low, high)
end

"""
    get_interpolation_weights(coords::Vector{T}, target::T, indices::Tuple{Int, Int}) where T

Calculate interpolation weights for linear interpolation.
"""
function get_interpolation_weights(coords::Vector{T}, target::T, indices::Tuple{Int, Int}) where T
    i1, i2 = indices
    
    if i1 == i2
        return (1.0, 0.0)
    end
    
    dx = coords[i2] - coords[i1]
    w2 = (target - coords[i1]) / dx
    w1 = 1.0 - w2
    
    return (w1, w2)
end

"""
    bilinear_interpolate(data::Matrix{T}, theta_idx::Tuple{Int, Int}, phi_idx::Tuple{Int, Int},
                        theta_weights::Tuple{T, T}, phi_weights::Tuple{T, T}) where T

Perform bilinear interpolation on a 2D data array.
"""
function bilinear_interpolate(data::Matrix{T}, theta_idx::Tuple{Int, Int}, phi_idx::Tuple{Int, Int},
                            theta_weights::Tuple{T, T}, phi_weights::Tuple{T, T}) where T
    
    i1, i2 = theta_idx
    j1, j2 = phi_idx
    wt1, wt2 = theta_weights
    wp1, wp2 = phi_weights
    
    # Get the four surrounding points
    f11 = data[i1, j1]
    f12 = data[i1, j2]
    f21 = data[i2, j1]
    f22 = data[i2, j2]
    
    # Bilinear interpolation
    result = wt1 * wp1 * f11 + wt1 * wp2 * f12 + wt2 * wp1 * f21 + wt2 * wp2 * f22
    
    return result
end

"""
    create_interpolation_cache(boundary_data::BoundaryData, target_theta::Vector{T}, 
                              target_phi::Vector{T}) where T

Create interpolation cache for efficient repeated interpolations.
"""
function create_interpolation_cache(boundary_data::BoundaryData, target_theta::Vector{T}, 
                                  target_phi::Vector{T}) where T
    
    cache = Dict{String, Any}()
    
    if boundary_data.theta === nothing || boundary_data.phi === nothing
        return cache  # No interpolation needed
    end
    
    src_theta = boundary_data.theta
    src_phi = boundary_data.phi
    
    # Pre-compute interpolation indices and weights
    theta_indices = Vector{Tuple{Int, Int}}(undef, length(target_theta))
    theta_weights = Vector{Tuple{T, T}}(undef, length(target_theta))
    phi_indices = Vector{Tuple{Int, Int}}(undef, length(target_phi))
    phi_weights = Vector{Tuple{T, T}}(undef, length(target_phi))
    
    for (i, theta_t) in enumerate(target_theta)
        theta_indices[i] = find_grid_indices(src_theta, theta_t)
        theta_weights[i] = get_interpolation_weights(src_theta, theta_t, theta_indices[i])
    end
    
    for (j, phi_t) in enumerate(target_phi)
        phi_indices[j] = find_grid_indices(src_phi, phi_t)
        phi_weights[j] = get_interpolation_weights(src_phi, phi_t, phi_indices[j])
    end
    
    cache["theta_indices"] = theta_indices
    cache["theta_weights"] = theta_weights
    cache["phi_indices"] = phi_indices
    cache["phi_weights"] = phi_weights
    cache["target_shape"] = (length(target_theta), length(target_phi))
    
    return cache
end

"""
    interpolate_with_cache(boundary_data::BoundaryData, cache::Dict, time_index::Int=1)

Perform interpolation using pre-computed cache for efficiency.
"""
function interpolate_with_cache(boundary_data::BoundaryData, cache::Dict, time_index::Int=1)
    
    if isempty(cache)
        # No interpolation needed, return data as-is
        if boundary_data.is_time_dependent
            if boundary_data.ncomponents == 1
                return boundary_data.values[:, :, time_index]
            else
                return boundary_data.values[:, :, time_index, :]
            end
        else
            return boundary_data.values
        end
    end
    
    # Extract cached interpolation data
    theta_indices = cache["theta_indices"]
    theta_weights = cache["theta_weights"]
    phi_indices = cache["phi_indices"]
    phi_weights = cache["phi_weights"]
    nlat_tgt, nlon_tgt = cache["target_shape"]
    
    # Extract source data for the specified time index
    if boundary_data.is_time_dependent
        if time_index > boundary_data.ntime
            time_index = boundary_data.ntime
        end
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values[:, :, time_index]
        else
            src_data = boundary_data.values[:, :, time_index, :]
        end
    else
        if boundary_data.ncomponents == 1
            src_data = boundary_data.values
        else
            src_data = boundary_data.values
        end
    end
    
    # Initialize output array
    if boundary_data.ncomponents == 1
        interpolated = zeros(eltype(boundary_data.values), nlat_tgt, nlon_tgt)
    else
        interpolated = zeros(eltype(boundary_data.values), nlat_tgt, nlon_tgt, boundary_data.ncomponents)
    end
    
    # Perform cached interpolation
    for i in 1:nlat_tgt
        for j in 1:nlon_tgt
            if boundary_data.ncomponents == 1
                interpolated[i, j] = bilinear_interpolate(
                    src_data, theta_indices[i], phi_indices[j], 
                    theta_weights[i], phi_weights[j]
                )
            else
                for comp in 1:boundary_data.ncomponents
                    interpolated[i, j, comp] = bilinear_interpolate(
                        src_data[:, :, comp], theta_indices[i], phi_indices[j],
                        theta_weights[i], phi_weights[j]
                    )
                end
            end
        end
    end
    
    return interpolated
end

"""
    validate_interpolation_grids(src_theta::Vector, src_phi::Vector, 
                                tgt_theta::Vector, tgt_phi::Vector)

Validate that source and target grids are compatible for interpolation.
"""
function validate_interpolation_grids(src_theta::Vector, src_phi::Vector, 
                                     tgt_theta::Vector, tgt_phi::Vector)
    
    errors = String[]
    
    # Check coordinate ranges
    if minimum(tgt_theta) < minimum(src_theta) || maximum(tgt_theta) > maximum(src_theta)
        push!(errors, "Target theta range exceeds source range")
    end
    
    if minimum(tgt_phi) < minimum(src_phi) || maximum(tgt_phi) > maximum(src_phi)
        push!(errors, "Target phi range exceeds source range")
    end
    
    # Check for monotonicity
    if !issorted(src_theta)
        push!(errors, "Source theta coordinates are not monotonic")
    end
    
    if !issorted(src_phi)
        push!(errors, "Source phi coordinates are not monotonic")
    end
    
    if !issorted(tgt_theta)
        push!(errors, "Target theta coordinates are not monotonic")
    end
    
    if !issorted(tgt_phi)
        push!(errors, "Target phi coordinates are not monotonic")
    end
    
    if !isempty(errors)
        error_msg = "Grid interpolation validation failed:\n" * join(errors, "\n")
        throw(ArgumentError(error_msg))
    end
    
    return true
end

export interpolate_boundary_to_grid, create_interpolation_cache, interpolate_with_cache
export validate_interpolation_grids