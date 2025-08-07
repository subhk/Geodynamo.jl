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


# ==================================================
# Optimized gradient computation using SHTns
# ==================================================
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


# ================================
# Fused advection computation
# ================================
function compute_temperature_advection!(temp_field::SHTnsTemperatureField{T}, 
                                             vel_fields) where T
    # Compute -u·∇T with fused operations for efficiency
    
    # Get local data views
    u_r = parent(vel_fields.velocity.r_component.data)
    u_θ = parent(vel_fields.velocity.θ_component.data)
    u_φ = parent(vel_fields.velocity.φ_component.data)
    
    grad_r = parent(temp_field.gradient.r_component.data)
    grad_θ = parent(temp_field.gradient.θ_component.data)
    grad_φ = parent(temp_field.gradient.φ_component.data)
    
    advection = parent(temp_field.advection_physical.data)
    
    # Fused computation of -u·∇T
    @inbounds @simd for idx in eachindex(advection)
        if idx <= length(u_r) && idx <= length(grad_r)
            advection[idx] = -(u_r[idx] * grad_r[idx] + 
                              u_θ[idx] * grad_θ[idx] + 
                              u_φ[idx] * grad_φ[idx])
        end
    end
end


# =======================================
# Optimized internal source addition
# =======================================
function add_internal_sources_optimized!(temp_field::SHTnsTemperatureField{T}) where T
    # Add internal heat sources efficiently
    # Sources are typically axisymmetric (l,m=0 modes)
    
    advection = parent(temp_field.advection_physical.data)
    
    # Get local ranges
    r_range = get_local_range(temp_field.advection_physical.pencil, 3)
    
    # Add volumetric heating (if present)
    if !all(iszero, temp_field.internal_sources)
        nlat = temp_field.advection_physical.nlat
        nlon = temp_field.advection_physical.nlon
        
        @inbounds for r_idx in r_range
            if r_idx <= length(temp_field.internal_sources)
                local_r = r_idx - first(r_range) + 1
                source_value = temp_field.internal_sources[r_idx]
                
                # Add uniformly across the sphere at this radius
                @simd for j in 1:nlon
                    for i in 1:nlat
                        linear_idx = i + (j-1)*nlat + (local_r-1)*nlat*nlon
                        if linear_idx <= length(advection)
                            advection[linear_idx] += source_value
                        end
                    end
                end
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
