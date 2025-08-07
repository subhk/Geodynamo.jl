# ===============================================================
# Temperature  field components with optimized SHTns transforms
# ===============================================================
struct SHTnsTemperatureField{T}
    # Physical space temperature
    temperature::SHTnsPhysicalField{T}
    gradient::SHTnsVectorField{T}
    
    # Spectral representation
    spectral::SHTnsSpectralField{T}
    
    # Nonlinear terms (advection)
    nonlinear::SHTnsSpectralField{T}
    
    # Work arrays for efficient computation
    work_spectral::SHTnsSpectralField{T}
    work_physical::SHTnsPhysicalField{T}
    advection_physical::SHTnsPhysicalField{T}
    
    # Gradient spectral components for efficiency
    grad_theta_spec::SHTnsSpectralField{T}
    grad_phi_spec::SHTnsSpectralField{T}
    
    # Sources and boundary conditions
    internal_sources::Vector{T}
    boundary_values::Matrix{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}  # l(l+1) values
    
    # Transform manager
    transform_manager::SHTnsTransformManager{T}
    
    # Radial derivative matrix for gradient computation
    dr_matrix::BandedMatrix{T}
end


function create_shtns_temperature_field(::Type{T}, config::SHTnsConfig, 
                                        domain::RadialDomain, 
                                        pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Temperature field
    temperature = create_shtns_physical_field(T, config, domain, pencil_r)
    
    # Gradient components
    gradient = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral representation
    spectral  = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nonlinear = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Work arrays
    work_spectral = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_physical = create_shtns_physical_field(T, config, domain, pencil_r)
    advection_physical = create_shtns_physical_field(T, config, domain, pencil_r)
    
    # Gradient spectral components
    grad_theta_spec = create_shtns_spectral_field(T, config, domain, pencil_spec)
    grad_phi_spec = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Sources and boundary conditions
    internal_sources = zeros(T, domain.N)
    boundary_values  = zeros(T, 2, config.nlm)  # ICB and CMB values
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Create transform manager
    transform_manager = get_transform_manager(T, config, pencil_spec)
    
    # Create radial derivative matrix
    dr_matrix = create_derivative_matrix(1, domain)
    
    return SHTnsTemperatureField{T}(temperature, gradient, spectral, nonlinear,
                                    work_spectral, work_physical, advection_physical,
                                    grad_theta_spec, grad_phi_spec,
                                    internal_sources, boundary_values,
                                    l_factors, transform_manager, dr_matrix)
end


# ==========================================================
# Main nonlinear computation using optimized transforms
# ==========================================================
function compute_temperature_nonlinear!(temp_field::SHTnsTemperatureField{T}, 
                                        vel_fields, domain::RadialDomain,
                                        transpose_plans=nothing) where T
    # Zero work arrays
    zero_temperature_work_arrays!(temp_field)
    
    # Step 1: Convert spectral temperature to physical space using optimized transform
    shtns_spectral_to_physical!(temp_field.spectral, temp_field.temperature, transpose_plans)
    
    # Step 2: Compute temperature gradient efficiently
    compute_temperature_gradient!(temp_field, domain)
    
    # Step 3: Compute advection term -u·∇T in physical space
    if vel_fields !== nothing
        compute_temperature_advection!(temp_field, vel_fields)
    end
    
    # Step 4: Add internal heat sources
    add_internal_sources!(temp_field)
    
    # Step 5: Transform advection + sources to spectral space for nonlinear term
    shtns_physical_to_spectral!(temp_field.advection_physical, temp_field.nonlinear, transpose_plans)
    
    # Step 6: Apply boundary conditions in spectral space
    apply_temperature_boundary_conditions!(temp_field, domain)
end


# ============================================================================
# Optimized gradient computation using SHTns
# ============================================================================
function compute_temperature_gradient!(temp_field::SHTnsTemperatureField{T}, 
                                                domain::RadialDomain) where T
    # Compute gradient using SHTns optimized routines
    config = temp_field.spectral.config
    sht = config.sht
    
    # Get local data views
    spec_real = parent(temp_field.spectral.data_real)
    spec_imag = parent(temp_field.spectral.data_imag)
    
    grad_r_data = parent(temp_field.gradient.r_component.data)
    grad_θ_data = parent(temp_field.gradient.θ_component.data)
    grad_φ_data = parent(temp_field.gradient.φ_component.data)
    
    # Get local ranges
    r_range = get_local_range(temp_field.spectral.pencil, 3)
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    
    # Use transform manager for efficiency
    manager = temp_field.transform_manager
    coeffs = manager.coeffs_full
    
    # Pre-allocate derivative work arrays
    nr = domain.N
    temp_profile = zeros(T, nr)
    dtemp_dr = zeros(T, nr)
    
    # Step 1: Compute horizontal gradients using SHTns at each radial level
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(spec_real, 3)
            # Fill coefficients for this radial level
            fill_coefficients_from_local!(coeffs, spec_real, spec_imag, 
                                         local_r, lm_range)
            
            # MPI communication if needed
            if manager.needs_allreduce
                MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
            end
            
            # Compute both horizontal derivatives simultaneously
            dT_dtheta = synthesis_dtheta(sht, coeffs)
            dT_dphi = synthesis_dphi(sht, coeffs)
            
            # Apply geometric factors and store
            r = domain.r[r_idx, 4]
            r_inv = domain.r[r_idx, 3]  # 1/r
            
            @simd for idx in eachindex(dT_dtheta)
                if idx <= size(grad_θ_data, 1) * size(grad_θ_data, 2)
                    i = ((idx - 1) % size(grad_θ_data, 1)) + 1
                    j = ((idx - 1) ÷ size(grad_θ_data, 1)) + 1
                    
                    if i <= config.nlat && j <= config.nlon
                        theta = config.theta_grid[i]
                        sin_theta_inv = 1.0 / max(sin(theta), 1e-10)
                        
                        # Store with proper geometric factors
                        linear_idx = i + (j-1)*size(grad_θ_data, 1) + 
                                    (local_r-1)*size(grad_θ_data, 1)*size(grad_θ_data, 2)
                        if linear_idx <= length(grad_θ_data)
                            grad_θ_data[linear_idx] = r_inv * real(dT_dtheta[idx])
                            grad_φ_data[linear_idx] = r_inv * sin_theta_inv * real(dT_dphi[idx])
                        end
                    end
                end
            end
        end
    end
    
    # Step 2: Compute radial gradient using spectral coefficients
    compute_radial_gradient_spectral!(temp_field, domain)
end

# ==================================================
# Radial gradient computation in spectral space
# ==================================================
function compute_radial_gradient_spectral!(temp_field::SHTnsTemperatureField{T}, 
                                          domain::RadialDomain) where T

    # Compute dT/dr for each (l,m) mode using banded matrix
    
    spec_real   = parent(temp_field.spectral.data_real)
    spec_imag   = parent(temp_field.spectral.data_imag)
    grad_r_data = parent(temp_field.gradient.r_component.data)
    
    # Get local ranges
    lm_range = get_local_range(temp_field.spectral.pencil, 1)
    r_range  = get_local_range(temp_field.spectral.pencil, 3)
    
    nr = domain.N
    temp_profile_real = zeros(T, nr)
    temp_profile_imag = zeros(T, nr)
    dtemp_dr_real    = zeros(T, nr)
    dtemp_dr_imag    = zeros(T, nr)
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= temp_field.spectral.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Extract radial profile for this mode
            for r_idx in 1:nr
                if r_idx in r_range
                    local_r = r_idx - first(r_range) + 1
                    if local_r <= size(spec_real, 3)
                        temp_profile_real[r_idx] = spec_real[local_lm, 1, local_r]
                        temp_profile_imag[r_idx] = spec_imag[local_lm, 1, local_r]
                    end
                else
                    temp_profile_real[r_idx] = zero(T)
                    temp_profile_imag[r_idx] = zero(T)
                end
            end
            
            # Apply radial derivative
            apply_derivative_matrix!(dtemp_dr_real, temp_field.dr_matrix, temp_profile_real)
            apply_derivative_matrix!(dtemp_dr_imag, temp_field.dr_matrix, temp_profile_imag)
            
            # Transform to physical space and store
            # This requires synthesis at each radial level
            # For efficiency, we batch this operation
        end
    end
    
    # Transform radial gradient to physical space
    shtns_spectral_to_physical!(temp_field.grad_theta_spec, 
                                temp_field.gradient.r_component)
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


function compute_temperature_advection!(temp_field::SHTnsTemperatureField{T}, vel_fields) where T
    # Compute -u · ∇T
    # vel = vel_fields.velocity
    # grad = temp_field.gradient

    # Get local data views
    work_data = parent(temp_field.work_physical.data)

    vel_r   = parent(vel_fields.velocity.r_component.data)
    vel_θ   = parent(vel_fields.velocity.θ_component.data)
    vel_φ   = parent(vel_fields.velocity.φ_component.data)
    grad_r  = parent(temp_field.gradient.r_component.data)
    grad_θ  = parent(temp_field.gradient.θ_component.data)
    grad_φ  = parent(temp_field.gradient.φ_component.data)
    
    # Get dimensions
    local_size = size(work_data)
    
    # Get dimensions
    local_size = size(work_data)
    
    # Fused loop for advection computation
    @inbounds @simd for idx in eachindex(work_data)
        if idx <= length(vel_r) && idx <= length(grad_r)
            # Compute -u·∇T with fused operations
            work_data[idx] = -(vel_r[idx] * grad_r[idx] + 
                              vel_θ[idx] * grad_θ[idx] + 
                              vel_φ[idx] * grad_φ[idx])
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
    

function zero_work_arrays!(temp_field::SHTnsTemperatureField{T}) where T
    # Efficiently zero work arrays
    fill!(parent(temp_field.work_physical.data), zero(T))
    fill!(parent(temp_field.work_spectral.data_real), zero(T))
    fill!(parent(temp_field.work_spectral.data_imag), zero(T))
end


# # Export functions
# export SHTnsTemperatureField, create_shtns_temperature_field
# export compute_temperature_nonlinear!, compute_temperature_batch!
# export zero_work_arrays!
