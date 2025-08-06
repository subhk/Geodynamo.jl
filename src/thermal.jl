# module Temperature
# using PencilArrays
# using ..Parameters
# using ..VariableTypes
# using ..SHTnsSetup
# using ..SHTnsTransforms
# using ..Timestepping
# using ..LinearOps
    
struct SHTnsTemperatureField{T}
    # Physical space temperature
    temperature::SHTnsPhysicalField{T}
    gradient::SHTnsVectorField{T}
    
    # Spectral representation
    spectral::SHTnsSpectralField{T}
    
    # Nonlinear terms (advection)
    nonlinear::SHTnsSpectralField{T}
    
    # Sources and boundary conditions
    internal_sources::Vector{T}
    boundary_values::Matrix{T}
end

function create_shtns_temperature_field(::Type{T}, config::SHTnsConfig, 
                                        domain::RadialDomain, 
                                        pencils, pencil_spec) where T
                                        
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Temperature field
    temperature = create_shtns_physical_field(T, config, domain, pencil_θ, pencil_φ, pencil_r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral representation
    spectral  = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nonlinear = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Sources and boundary conditions
    internal_sources = zeros(T, domain.N)
    boundary_values  = zeros(T, 2, config.nlm)  # ICB and CMB values
    
    return SHTnsTemperatureField{T}(temperature, gradient, spectral, nonlinear,
                                    internal_sources, boundary_values)
end


function compute_temperature_nonlinear!(temp_field::SHTnsTemperatureField{T}, 
                                        vel_fields, 
                                        transpose_plans=nothing) where T

    # Use local computation flags to minimize communication
    needs_transpose = transpose_plans !== nothing

    # Convert spectral temperature to physical space
    shtns_spectral_to_physical!(temp_field.spectral, temp_field.temperature)
    
    # Compute temperature gradient using SHTns
    compute_temperature_gradient!(temp_field)
    
    # Compute advection -u·∇T locally with fused operations
    if vel_fields !== nothing
        compute_temperature_advection!(temp_field, vel_fields)
    end
    
    # Add internal heat sources
    add_internal_sources!(temp_field)
    
    # Transform to spectral space
    shtns_physical_to_spectral!(temp_field.temperature, temp_field.nonlinear)
end


# Alternative implementation using SHTns built-in gradient operations
function compute_temperature_gradient!(temp_field::SHTnsTemperatureField{T}) where T
    # Use SHTns built-in gradient operations for maximum efficiency
    # This leverages optimized SHTns routines
    
    config = temp_field.spectral.config
    sht   = config.sht
    nlm   = config.nlm
    
    # Get local data views
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    grad_r    = parent(temp_field.gradient.r_component.data)
    grad_θ    = parent(temp_field.gradient.θ_component.data)
    grad_φ    = parent(temp_field.gradient.φ_component.data)
    
    # Get local ranges
    r_range  = get_local_range(temp_field.spectral.pencil, 3)
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    
    # Pre-allocate arrays
    coeffs = zeros(ComplexF64, nlm)
    
    # Process radial levels
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Gather coefficients
            fill!(coeffs, zero(ComplexF64))
            
            @simd for lm_idx in lm_range
                if lm_idx <= nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    coeffs[lm_idx] = complex(spec_real[local_lm, 1, local_r],
                                            spec_imag[local_lm, 1, local_r])
                end
            end
            
            # Collective communication if needed
            if length(lm_range) < nlm
                coeffs = MPI.Allreduce(coeffs, MPI.SUM, get_comm())
            end
            
            # Compute horizontal gradients using SHTns
            compute_horizontal_gradient_batch!(sht, coeffs, grad_θ, grad_φ, 
                                              local_r, config)
        end
    end
    
    # Compute radial gradient using optimized finite differences
    compute_radial_gradient!(temp_field)
end


function compute_radial_gradient!(temp_field::SHTnsTemperatureField{T}, 
                                           domain::RadialDomain) where T
    # Compute radial gradient using high-order finite difference matrices
    
    # Create high-order radial derivative matrix
    dr_matrix = create_derivative_matrix(1, domain)  # First derivative
    
    # Apply to temperature field in physical space
    for i_theta in 1:temp_field.temperature.nlat, j_phi in 1:temp_field.temperature.nlon
        if (i_theta <= size(temp_field.temperature.data_r, 1) && 
            j_phi <= size(temp_field.temperature.data_r, 2))
            
            # Extract radial profile
            temp_profile = temp_field.temperature.data_r[i_theta, j_phi, :]
            
            # Apply derivative matrix
            grad_profile = apply_banded_matrix_vector(dr_matrix, temp_profile)
            
            # Store result
            if length(grad_profile) <= size(temp_field.gradient.r_component.data_r, 3)
                temp_field.gradient.r_component.data_r[i_theta, j_phi, 1:length(grad_profile)] = grad_profile
            end
        end
    end
end


function apply_gradient_boundary_conditions!(temp_field::SHTnsTemperatureField{T}, 
                                           domain::RadialDomain) where T
    # Apply boundary conditions to temperature gradients
    # This is important for proper heat flux calculations
    
    N = domain.N
    
    # Inner boundary (r = ri)
    for i_theta in 1:temp_field.gradient.r_component.nlat, j_phi in 1:temp_field.gradient.r_component.nlon
        if (i_theta <= size(temp_field.gradient.r_component.data_r, 1) && 
            j_phi <= size(temp_field.gradient.r_component.data_r, 2))
            
            # Apply boundary condition based on thermal boundary condition type
            # For fixed temperature: gradient determined by derivative
            # For fixed heat flux: gradient is prescribed
            
            if i_tmp_bc == 1  # Fixed temperature BC
                # Gradient computed from finite differences (already done)
                # No additional modification needed
            elseif i_tmp_bc == 2  # Fixed heat flux BC
                # Set radial gradient to prescribed value
                prescribed_heat_flux = get_prescribed_heat_flux(i_theta, j_phi)  # Would come from BC
                temp_field.gradient.r_component.data_r[i_theta, j_phi, 1] = prescribed_heat_flux
                temp_field.gradient.r_component.data_r[i_theta, j_phi, N] = prescribed_heat_flux
            end
        end
    end
end


# Utility functions
function create_radial_derivative_matrix()
    # Create radial derivative matrix for Chebyshev grid
    # This would use the proper radial domain information
    N = i_N
    bandwidth = i_KL
    
    # Placeholder - would create proper finite difference matrix
    data = zeros(2*bandwidth + 1, N)
    
    # Fill with finite difference coefficients
    for i in 1:N
        if i > 1 && i < N
            # Centered difference
            data[bandwidth, i+1] = 0.5
            data[bandwidth+2, i-1] = -0.5
        elseif i == 1
            # Forward difference
            data[bandwidth+1, i] = -1.0
            data[bandwidth, i+1] = 1.0
        else
            # Backward difference
            data[bandwidth+2, i-1] = -1.0
            data[bandwidth+1, i] = 1.0
        end
    end
    
    return BandedMatrix(data, bandwidth, N)
end


function apply_banded_matrix_vector(matrix::BandedMatrix{T}, vector::Vector{T}) where T
    # Apply banded matrix to vector
    return apply_derivative_matrix(matrix, vector)
end

function get_radius_at_level(r_idx::Int)
    # Get radius at radial grid point
    # Placeholder - would come from radial domain
    return 0.5 + 0.5 * cos(π * (r_idx - 1) / (i_N - 1))
end

function get_prescribed_heat_flux(i_theta::Int, j_phi::Int)
    # Get prescribed heat flux for boundary conditions
    # Placeholder - would come from boundary condition specification
    return 0.0
end


function compute_temperature_advection!(temp_field::SHTnsTemperatureField{T}, vel_fields) where T
    # Compute -u · ∇T
    vel = vel_fields.velocity
    grad = temp_field.gradient
    
    for r_idx in temp_field.temperature.local_radial_range
        if r_idx <= size(temp_field.temperature.data_r, 3)
            for j_phi in 1:temp_field.temperature.nlon, i_theta in 1:temp_field.temperature.nlat
                if (i_theta <= size(temp_field.temperature.data_r, 1) && 
                    j_phi <= size(temp_field.temperature.data_r, 2) &&
                    i_theta <= size(vel.r_component.data_r, 1) && 
                    j_phi <= size(vel.r_component.data_r, 2) &&
                    i_theta <= size(grad.r_component.data_r, 1) && 
                    j_phi <= size(grad.r_component.data_r, 2))
                    
                    u_r = vel.r_component.data_r[i_theta, j_phi, r_idx]
                    u_θ = vel.θ_component.data_r[i_theta, j_phi, r_idx]
                    u_φ = vel.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    dT_dr     = grad.r_component.data_r[i_theta, j_phi, r_idx]
                    dT_dtheta = grad.θ_component.data_r[i_theta, j_phi, r_idx]
                    dT_dphi   = grad.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    advection = -(u_r * dT_dr + u_θ * dT_dtheta + u_φ * dT_dphi)
                    temp_field.temperature.data_r[i_theta, j_phi, r_idx] = advection
                end
            end
        end
    end
end


function add_internal_sources_spectral!(temp_field::SHTnsTemperatureField{T}) where T
    # Work directly in spectral space for efficiency
    
    # Get local data
    nl_real = parent(temp_field.nonlinear.data_real)
    nl_imag = parent(temp_field.nonlinear.data_imag)
    
    # Find l=m=0 mode locally
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    r_range = get_local_range(temp_field.spectral.pencil, 3)
    
    # Check if l=m=0 is in local range
    for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            l = temp_field.spectral.config.l_values[lm_idx]
            m = temp_field.spectral.config.m_values[lm_idx]
            
            if l == 0 && m == 0
                local_lm = lm_idx - first(lm_range) + 1
                
                # Add sources only to l=m=0 mode
                @inbounds @simd for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if r_idx <= length(temp_field.internal_sources) && 
                       local_r <= size(nl_real, 3)
                        nl_real[local_lm, 1, local_r] += temp_field.internal_sources[r_idx]
                    end
                end
                break  # Only one l=m=0 mode
            end
        end
    end
end
    
#export SHTnsTemperatureField, create_shtns_temperature_field, compute_temperature_nonlinear!

#end
