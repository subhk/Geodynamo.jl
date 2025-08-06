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
                                        vel_fields) where T
    # Convert spectral temperature to physical space
    shtns_spectral_to_physical!(temp_field.spectral, temp_field.temperature)
    
    # Compute temperature gradient using SHTns
    compute_temperature_gradient!(temp_field)
    
    # Compute advection: -u · ∇T
    compute_temperature_advection!(temp_field, vel_fields)
    
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
    sht = config.sht
    
    # Process each radial level
    for r_idx in temp_field.spectral.local_radial_range
        if r_idx <= size(temp_field.spectral.data_real, 3)
            
            # Extract temperature coefficients at this radial level
            T_coeffs = extract_spectral_coefficients(temp_field.spectral, r_idx)
            
            # Use SHTns to compute horizontal gradient
            compute_horizontal_gradient_shtns!(sht, T_coeffs, temp_field.gradient, r_idx, config)
            
            # Compute radial gradient using finite differences
            compute_radial_gradient_at_level!(temp_field, r_idx)
        end
    end
end

function compute_horizontal_gradient_shtns!(sht, T_coeffs::Vector{ComplexF64}, 
                                          gradient::SHTnsVectorField{T}, 
                                          r_idx::Int, config::SHTnsConfig) where T
    # Compute horizontal gradient components using SHTns
    
    # Get radius for geometric factors
    r = get_radius_at_level(r_idx)
    r_inv = 1.0 / max(r, 1e-10)
    
    # Compute ∂T/∂θ using SHTns
    dT_dtheta_phys = synthesis_dtheta(sht, T_coeffs)
    
    # Compute ∂T/∂φ using SHTns  
    dT_dphi_phys = synthesis_dphi(sht, T_coeffs)
    
    # Apply geometric factors and store
    theta_grid = config.theta_grid
    nlat, nlon = size(dT_dtheta_phys)
    
    for i_theta in 1:nlat, j_phi in 1:nlon
        if (i_theta <= size(gradient.θ_component.data_r, 1) && 
            j_phi <= size(gradient.θ_component.data_r, 2) &&
            r_idx <= size(gradient.θ_component.data_r, 3))
            
            theta = theta_grid[i_theta]
            sin_theta = sin(theta)
            sin_theta_inv = 1.0 / max(sin_theta, 1e-10)
            
            # Store gradient components with proper geometric factors
            gradient.θ_component.data_r[i_theta, j_phi, r_idx] = 
                r_inv * real(dT_dtheta_phys[i_theta, j_phi])
            
            gradient.φ_component.data_r[i_theta, j_phi, r_idx] = 
                r_inv * sin_theta_inv * real(dT_dphi_phys[i_theta, j_phi])
        end
    end
end

function compute_radial_gradient_at_level!(temp_field::SHTnsTemperatureField{T}, r_idx::Int) where T
    # Compute radial gradient at a specific radial level using finite differences
    
    # Get temperature values at neighboring radial points
    N = size(temp_field.temperature.data_r, 3)
    
    for i_theta in 1:temp_field.temperature.nlat, j_phi in 1:temp_field.temperature.nlon
        if (i_theta <= size(temp_field.temperature.data_r, 1) && 
            j_phi <= size(temp_field.temperature.data_r, 2))
            
            # Extract radial profile at this (θ,φ) point
            radial_profile = temp_field.temperature.data_r[i_theta, j_phi, :]
            
            # Compute radial derivative using finite differences
            if r_idx == 1
                # Forward difference at inner boundary
                if N >= 2
                    dT_dr = radial_profile[2] - radial_profile[1]
                else
                    dT_dr = zero(T)
                end
            elseif r_idx == N
                # Backward difference at outer boundary
                dT_dr = radial_profile[N] - radial_profile[N-1]
            else
                # Centered difference in interior
                dT_dr = 0.5 * (radial_profile[r_idx+1] - radial_profile[r_idx-1])
            end
            
            # Store radial gradient
            if r_idx <= size(temp_field.gradient.r_component.data_r, 3)
                temp_field.gradient.r_component.data_r[i_theta, j_phi, r_idx] = dT_dr
            end
        end
    end
end

# High-accuracy implementation with proper radial grid
function compute_temperature_gradient_high_accuracy!(temp_field::SHTnsTemperatureField{T}, 
                                                   domain::RadialDomain) where T
    # High-accuracy implementation using proper radial grid and derivative matrices
    
    config = temp_field.spectral.config
    
    # Create derivative fields
    create_gradient_derivative_fields!(temp_field, config)
    
    # Compute angular derivatives using SHTns (spectral accuracy)
    compute_angular_gradients_spectral!(temp_field, config)
    
    # Compute radial derivatives using high-order finite differences
    compute_radial_gradient_high_order!(temp_field, domain)
    
    # Apply boundary conditions to gradients
    apply_gradient_boundary_conditions!(temp_field, domain)
    
    # Convert to physical space with proper geometric factors
    finalize_gradient_computation!(temp_field, domain, config)
end

function create_gradient_derivative_fields!(temp_field::SHTnsTemperatureField{T}, 
                                          config::SHTnsConfig) where T
    # Ensure gradient field components are properly allocated
    # This would initialize the gradient vector field if needed
    
    # Check dimensions and allocate if necessary
    nlat, nlon = config.nlat, config.nlon
    nr = size(temp_field.spectral.data_real, 3)
    
    # Ensure gradient fields have correct dimensions
    # (Implementation would verify and resize as needed)
end

function compute_angular_gradients_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                           config::SHTnsConfig) where T
    # Compute angular gradients using SHTns spectral derivatives
    
    sht = config.sht
    
    # Create temporary spectral fields for derivatives
    dT_dtheta_spec = similar(temp_field.spectral)
    dT_dphi_spec = similar(temp_field.spectral)
    
    # Compute spectral derivatives
    @views for lm_idx in 1:temp_field.spectral.nlm
        for r_idx in temp_field.spectral.local_radial_range
            if r_idx <= size(temp_field.spectral.data_real, 3)
                
                # Extract coefficients for this radial level
                T_coeffs = extract_spectral_coefficients(temp_field.spectral, r_idx)
                
                # Compute angular derivatives
                dT_dtheta_coeffs = compute_theta_derivative_coeffs(sht, T_coeffs, config, lm_idx)
                dT_dphi_coeffs = compute_phi_derivative_coeffs(sht, T_coeffs, config, lm_idx)
                
                # Store spectral derivatives
                dT_dtheta_spec.data_real[lm_idx, 1, r_idx] = real(dT_dtheta_coeffs[lm_idx])
                dT_dtheta_spec.data_imag[lm_idx, 1, r_idx] = imag(dT_dtheta_coeffs[lm_idx])
                dT_dphi_spec.data_real[lm_idx, 1, r_idx] = real(dT_dphi_coeffs[lm_idx])
                dT_dphi_spec.data_imag[lm_idx, 1, r_idx] = imag(dT_dphi_coeffs[lm_idx])
            end
        end
    end
    
    # Convert to physical space
    shtns_spectral_to_physical!(dT_dtheta_spec, temp_field.gradient.θ_component)
    shtns_spectral_to_physical!(dT_dphi_spec, temp_field.gradient.φ_component)
end

function compute_theta_derivative_coeffs(sht, T_coeffs::Vector{ComplexF64}, 
                                       config::SHTnsConfig, lm_idx::Int)
    # Compute θ derivative coefficients using SHTns
    
    # Use SHTns built-in theta derivative
    dT_dtheta_phys = synthesis_dtheta(sht, T_coeffs)
    
    # Convert back to spectral coefficients
    return analysis(sht, dT_dtheta_phys)
end

function compute_phi_derivative_coeffs(sht, T_coeffs::Vector{ComplexF64}, 
                                     config::SHTnsConfig, lm_idx::Int)
    # Compute φ derivative coefficients using SHTns
    
    # Use SHTns built-in phi derivative
    dT_dphi_phys = synthesis_dphi(sht, T_coeffs)
    
    # Convert back to spectral coefficients
    return analysis(sht, dT_dphi_phys)
end

function compute_radial_gradient_high_order!(temp_field::SHTnsTemperatureField{T}, 
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

function finalize_gradient_computation!(temp_field::SHTnsTemperatureField{T}, 
                                      domain::RadialDomain, config::SHTnsConfig) where T
    # Apply final geometric factors and ensure consistency
    
    theta_grid = config.theta_grid
    
    for r_idx in temp_field.gradient.r_component.local_radial_range
        if r_idx <= size(temp_field.gradient.r_component.data_r, 3)
            
            # Get radius and geometric factors
            r = domain.r[r_idx, 4]  # r^1
            r_inv = 1.0 / max(r, 1e-10)
            
            for i_theta in 1:temp_field.gradient.r_component.nlat
                theta = theta_grid[i_theta]
                sin_theta = sin(theta)
                sin_theta_inv = 1.0 / max(sin_theta, 1e-10)
                
                for j_phi in 1:temp_field.gradient.r_component.nlon
                    if (i_theta <= size(temp_field.gradient.r_component.data_r, 1) && 
                        j_phi <= size(temp_field.gradient.r_component.data_r, 2))
                        
                        # Apply geometric factors to get proper gradient components
                        # ∇T = (∂T/∂r, (1/r)∂T/∂θ, (1/(r sin θ))∂T/∂φ)
                        
                        # Radial component (already correct)
                        # temp_field.gradient.r_component.data_r[i_theta, j_phi, r_idx] unchanged
                        
                        # θ component: multiply by 1/r
                        temp_field.gradient.θ_component.data_r[i_theta, j_phi, r_idx] *= r_inv
                        
                        # φ component: multiply by 1/(r sin θ)
                        temp_field.gradient.φ_component.data_r[i_theta, j_phi, r_idx] *= (r_inv * sin_theta_inv)
                    end
                end
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

function apply_derivative_matrix(matrix::BandedMatrix{T}, vector::Vector{T}) where T
    # Apply banded derivative matrix to vector
    result = zeros(T, length(vector))
    N = matrix.size
    bandwidth = matrix.bandwidth
    
    for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= size(matrix.data, 1) && j <= length(vector)
                result[i] += matrix.data[band_row, j] * vector[j]
            end
        end
    end
    
    return result
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
                    
                    dT_dr = grad.r_component.data_r[i_theta, j_phi, r_idx]
                    dT_dtheta = grad.θ_component.data_r[i_theta, j_phi, r_idx]
                    dT_dphi = grad.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    advection = -(u_r * dT_dr + u_θ * dT_dtheta + u_φ * dT_dphi)
                    temp_field.temperature.data_r[i_theta, j_phi, r_idx] = advection
                end
            end
        end
    end
end

function add_internal_sources_local!(temp_field::SHTnsTemperatureField{T}) where T
    # Add internal heat sources to l=m=0 mode
    spec_real = parent(temp_field.spectral.data_real)
    
    # Find l=m=0 mode in local data
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    
    for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            l = temp_field.spectral.config.l_values[lm_idx]
            m = temp_field.spectral.config.m_values[lm_idx]
            
            if l == 0 && m == 0
                local_lm = lm_idx - first(lm_range) + 1
                r_range = get_local_range(temp_field.spectral.pencil, 3)
                
                for r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if r_idx <= length(temp_field.internal_sources)
                        spec_real[local_lm, 1, local_r] += temp_field.internal_sources[r_idx]
                    end
                end
                break
            end
        end
    end
end
    
#export SHTnsTemperatureField, create_shtns_temperature_field, compute_temperature_nonlinear!

#end
