# ============================================================================
# Physics Modules with SHTns
# ============================================================================


# Velocity field components with SHTns
struct SHTnsVelocityFields{T}
    # Physical space velocities
    velocity::SHTnsVectorField{T}
    vorticity::SHTnsVectorField{T}
    
    # Spectral representation (toroidal-poloidal)
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
    
    # Vorticity in spectral space (for efficient curl computation)
    vort_toroidal::SHTnsSpectralField{T}
    vort_poloidal::SHTnsSpectralField{T}
    
    # Nonlinear terms
    nl_toroidal::SHTnsSpectralField{T}
    nl_poloidal::SHTnsSpectralField{T}
    
    # Work arrays for efficient computation
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
    work_physical::SHTnsVectorField{T}
    advection_physical::SHTnsVectorField{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}          # l(l+1) values
    coriolis_factors::Matrix{Float64}   # Pre-computed Coriolis terms
    
    # Radial derivative matrices
    dr_matrix::BandedMatrix{T}          # First derivative
    d2r_matrix::BandedMatrix{T}         # Second derivative
    laplacian_matrix::BandedMatrix{T}   # Radial Laplacian operator
    
    # Transform manager for efficient transforms
    transform_manager::SHTnsTransformManager{T}
end


function create_shtns_velocity_fields(::Type{T}, config::SHTnsConfig, 
                                      domain::RadialDomain, 
                                      pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Create vector fields
    velocity  = create_shtns_vector_field(T, config, domain, pencils)
    vorticity = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral fields
    toroidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    poloidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    pressure    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, domain, pencil_spec)

    work_physical = create_shtns_vector_field(T, config, domain, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(Float64, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end
    
    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal, 
                                  nl_toroidal, nl_poloidal, pressure,
                                  work_tor, work_pol, work_physical,
                                  l_factors, coriolis_factors)
end


# =============================
# Main nonlinear computation
# =============================
function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field) where T

    # Convert spectral toroidal/poloidal to physical velocity
    shtns_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
    
    # Compute vorticity from velocity
    compute_vorticity_shtns!(fields.velocity, fields.vorticity)
    
    # Compute u × ω (advection term)
    compute_advection_term!(fields)
    
    # Add Coriolis force: -2Ω × u
    add_coriolis_force!(fields)
    
    # Add buoyancy forces
    add_thermal_buoyancy!(fields, temp_field)
    if comp_field !== nothing
        add_compositional_buoyancy!(fields, comp_field)
    end
    
    # Add Lorentz force: (∇ × B) × B
    if mag_field !== nothing
        #add_lorentz_force!(fields, mag_field)
        add_lorentz_force_vectorSH!(fields, mag_field)
    end
    
    # Transform nonlinear terms back to spectral space
    shtns_vector_analysis!(fields.velocity, fields.nl_toroidal, fields.nl_poloidal)
end

# function compute_vorticity!(velocity::SHTnsVectorField{T}, vorticity::SHTnsVectorField{T}) where T
#     # Compute vorticity = ∇ × u using SHTns
#     # This would use SHTns curl operations
    
#     # Simplified implementation - would use proper SHTns curl
#     config = velocity.r_component.config
    
#     # For now, copy structure (placeholder)
#     for r_idx in velocity.r_component.local_radial_range
#         # Compute curl components using SHTns operators
#         # ω_r = (1/sin θ)(∂v_φ/∂θ - ∂(sin θ v_θ)/∂φ)
#         # ω_θ = (1/sin θ)(∂v_r/∂φ) - ∂v_φ/∂r
#         # ω_φ = ∂v_θ/∂r - (1/r)(∂v_r/∂θ)
        
#         # This requires spectral differentiation - placeholder
#     end
# end


# Alternative implementation using vector spherical harmonics
function compute_vorticity_vector_sh!(velocity::SHTnsVectorField{T}, 
                                     vorticity::SHTnsVectorField{T}) where T
    # Compute vorticity using vector spherical harmonic operations
    # This approach works directly with toroidal-poloidal decomposition
    
    # First, we need to get the toroidal-poloidal representation of velocity
    # This requires access to the velocity's spectral representation
    # For this example, assume we have access to toroidal and poloidal components
    
    config = velocity.r_component.config
    nlm = config.nlm
    
    # Create temporary toroidal-poloidal fields for velocity
    vel_toroidal = create_temp_spectral_field(T, config, velocity.r_component)
    vel_poloidal = create_temp_spectral_field(T, config, velocity.r_component)
    
    # Convert velocity to toroidal-poloidal (this would be done by vector analysis)
    shtns_vector_analysis!(velocity, vel_toroidal, vel_poloidal)
    
    # Create vorticity toroidal-poloidal fields
    vort_toroidal = similar(vel_toroidal)
    vort_poloidal = similar(vel_poloidal)
    
    # Compute curl using vector spherical harmonic relations
    compute_curl_vector_sh!(vel_toroidal, vel_poloidal, vort_toroidal, vort_poloidal, config)
    
    # Convert back to physical vector components
    shtns_vector_synthesis!(vort_toroidal, vort_poloidal, vorticity)
end

function compute_curl_vector_sh!(vel_toroidal::SHTnsSpectralField{T}, 
                                vel_poloidal::SHTnsSpectralField{T},
                                vort_toroidal::SHTnsSpectralField{T}, 
                                vort_poloidal::SHTnsSpectralField{T},
                                config::SHTnsConfig) where T
    # Compute curl using vector spherical harmonic relationships
    # For vector field v = ∇ × (T r̂) + ∇ × ∇ × (P r̂)
    # The curl ∇ × v has specific relationships with T and P
    
    @views for lm_idx in 1:vel_toroidal.nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        
        # Vector spherical harmonic curl relationships
        l_factor = Float64(l * (l + 1))
        
        for r_idx in vel_toroidal.local_radial_range
            if r_idx <= size(vel_toroidal.data_real, 3)
                
                # Get velocity toroidal and poloidal coefficients
                T_vel_real = vel_toroidal.data_real[lm_idx, 1, r_idx]
                T_vel_imag = vel_toroidal.data_imag[lm_idx, 1, r_idx]
                P_vel_real = vel_poloidal.data_real[lm_idx, 1, r_idx]
                P_vel_imag = vel_poloidal.data_imag[lm_idx, 1, r_idx]
                
                # Curl relationships (simplified - full implementation needs radial derivatives)
                # These would involve radial derivatives of T and P and coupling between modes
                
                # For curl of toroidal field: contributes to poloidal vorticity
                vort_poloidal.data_real[lm_idx, 1, r_idx] = l_factor * T_vel_real
                vort_poloidal.data_imag[lm_idx, 1, r_idx] = l_factor * T_vel_imag
                
                # For curl of poloidal field: contributes to toroidal vorticity
                vort_toroidal.data_real[lm_idx, 1, r_idx] = -l_factor * P_vel_real
                vort_toroidal.data_imag[lm_idx, 1, r_idx] = -l_factor * P_vel_imag
            end
        end
    end
end

# =========================================
# Vorticity computation in spectral space
# =========================================
function compute_vorticity_spectral!(fields::SHTnsVelocityFields{T}) where T
    # Compute vorticity directly in spectral space using l(l+1) factors
    # ω = ∇ × u, which in toroidal-poloidal space has simple relationships
    
    # Get local data views
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    
    vort_tor_real = parent(fields.work_tor.data_real)
    vort_tor_imag = parent(fields.work_tor.data_imag)
    vort_pol_real = parent(fields.work_pol.data_real)
    vort_pol_imag = parent(fields.work_pol.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range = get_local_range(fields.toroidal.pencil, 3)
    
    # Apply curl relationships in spectral space
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Vorticity from velocity in spectral space
                    # ω_tor = l(l+1) * u_pol
                    # ω_pol = -l(l+1) * u_tor
                    vort_tor_real[local_lm, 1, local_r] = l_factor * pol_real[local_lm, 1, local_r]
                    vort_tor_imag[local_lm, 1, local_r] = l_factor * pol_imag[local_lm, 1, local_r]
                    vort_pol_real[local_lm, 1, local_r] = -l_factor * tor_real[local_lm, 1, local_r]
                    vort_pol_imag[local_lm, 1, local_r] = -l_factor * tor_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end


function compute_radial_vorticity_component!(velocity::SHTnsVectorField{T}, 
                                           vorticity::SHTnsVectorField{T}, 
                                           r_idx::Int, config::SHTnsConfig) where T
    # Compute radial component of vorticity
    # ω_r = (1/(r sin θ)) * [∂u_φ/∂θ - ∂(sin θ u_θ)/∂φ]
    
    sht = config.sht
    
    # Extract u_θ and u_φ at this radial level
    u_theta = zeros(ComplexF64, config.nlat, config.nlon)
    u_phi = zeros(ComplexF64, config.nlat, config.nlon)
    
    for j_phi in 1:config.nlon, i_theta in 1:config.nlat
        if (i_theta <= size(velocity.θ_component.data_r, 1) && 
            j_phi <= size(velocity.θ_component.data_r, 2))
            
            u_theta[i_theta, j_phi] = complex(velocity.θ_component.data_r[i_theta, j_phi, r_idx])
            u_phi[i_theta, j_phi] = complex(velocity.φ_component.data_r[i_theta, j_phi, r_idx])
        end
    end
    
    # Analyze to get spectral coefficients
    u_theta_coeffs = analysis(sht, u_theta)
    u_phi_coeffs = analysis(sht, u_phi)
    
    # Compute derivatives
    du_phi_dtheta = synthesis_dtheta(sht, u_phi_coeffs)
    du_theta_dphi = synthesis_dphi(sht, u_theta_coeffs)
    
    # Compute radial vorticity component
    r = 1.0  # Would get from radial domain
    r_inv = 1.0 / max(r, 1e-10)
    
    for j_phi in 1:config.nlon, i_theta in 1:config.nlat
        if (i_theta <= size(vorticity.r_component.data_r, 1) && 
            j_phi <= size(vorticity.r_component.data_r, 2))
            
            theta = config.theta_grid[i_theta]
            sin_theta = sin(theta)
            sin_theta_inv = 1.0 / max(sin_theta, 1e-10)
            
            omega_r = r_inv * sin_theta_inv * 
                     (real(du_phi_dtheta[i_theta, j_phi]) - 
                      sin_theta * real(du_theta_dphi[i_theta, j_phi]))
            
            vorticity.r_component.data_r[i_theta, j_phi, r_idx] = omega_r
        end
    end
end


function compute_advection_term!(fields::SHTnsVelocityFields{T}) where T
    # Compute u × ω in physical space
    vel = fields.velocity
    vort = fields.vorticity
    
    # u × ω = (u_θ ω_φ - u_φ ω_θ, u_φ ω_r - u_r ω_φ, u_r ω_θ - u_θ ω_r)

    for r_idx in vel.r_component.local_radial_range
        if r_idx <= size(vel.r_component.data_r, 3)
            for j_phi in 1:vel.r_component.nlon, i_theta in 1:vel.r_component.nlat
                if i_theta <= size(vel.r_component.data_r, 1) && j_phi <= size(vel.r_component.data_r, 2)
                    u_r = vel.r_component.data_r[i_theta, j_phi, r_idx]
                    u_θ = vel.θ_component.data_r[i_theta, j_phi, r_idx]
                    u_φ = vel.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    ω_r = vort.r_component.data_r[i_theta, j_phi, r_idx]
                    ω_θ = vort.θ_component.data_r[i_theta, j_phi, r_idx]
                    ω_φ = vort.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    # Store advection in velocity field temporarily
                    vel.r_component.data_r[i_theta, j_phi, r_idx] = d_Ro * (u_θ * ω_φ - u_φ * ω_θ)
                    vel.θ_component.data_r[i_theta, j_phi, r_idx] = d_Ro * (u_φ * ω_r - u_r * ω_φ)
                    vel.φ_component.data_r[i_theta, j_phi, r_idx] = d_Ro * (u_r * ω_θ - u_θ * ω_r)
                end
            end
        end
    end
end


function add_coriolis_force_local!(fields::SHTnsVelocityFields{T}) where T
    # Coriolis force: -2Ω × u assuming rotation about z-axis
    # In spherical coordinates: Ω = Ω(cos θ ê_r - sin θ ê_θ)

    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    config = fields.velocity.r_component.config
    theta_grid = config.theta_grid
    
    # Get local indices
    local_size = size(vel_r)
    
    for k in 1:local_size[3], j in 1:local_size[2], i in 1:local_size[1]
        if i <= length(theta_grid)
            theta = theta_grid[i]
            cos_theta = cos(theta)
            sin_theta = sin(theta)
            
            u_r = vel_r[i, j, k]
            u_θ = vel_θ[i, j, k]
            u_φ = vel_φ[i, j, k]
            
            vel_r[i, j, k] += -2.0 * (-sin_theta * u_φ)
            vel_θ[i, j, k] += -2.0 * (cos_theta  * u_φ)
            vel_φ[i, j, k] += -2.0 * (-cos_theta * u_θ + sin_theta * u_r)
        end
    end
end


function add_thermal_buoyancy!(work_r::AbstractArray{T,3}, 
                                scalar_field, factor::Float64) where T
    # Get the scalar field data (temperature or composition)
    if isa(scalar_field, SHTnsPhysicalField)
        scalar_data = parent(scalar_field.data)
    else
        # If it's already in physical space from temperature module
        scalar_data = parent(scalar_field.temperature.data)
    end
    
    # Vectorized addition of buoyancy
    @inbounds @simd for idx in eachindex(work_r)
        if idx <= length(scalar_data)
            work_r[idx] += factor * scalar_data[idx]
        end
    end
end


function add_lorentz_force_spectral!(fields::SHTnsVelocityFields{T}, mag_field) where T
    # Compute Lorentz force in spectral space for efficiency
    # F = (∇ × B) × B / Pm
    
    # First compute current density j = ∇ × B in spectral space
    compute_magnetic_curl_spectral!(mag_field, fields.work_tor, fields.work_pol, fields.l_factors)
    
    # Transform j to physical space
    shtns_vector_synthesis!(fields.work_tor, fields.work_pol, fields.work_physical)
    
    # Also need B in physical space (should already be there from mag_field computation)
    # If not, transform it
    if !is_in_physical_space(mag_field.magnetic)
        shtns_vector_synthesis!(mag_field.toroidal, mag_field.poloidal, mag_field.magnetic)
    end
    
    # Compute j × B in physical space
    compute_cross_product_jB!(fields.work_physical, mag_field.magnetic, fields.work_physical)
    
    # Scale by 1/Pm
    scale_field!(fields.work_physical, 1.0 / d_Pm)
    
    # Add to existing forces
    add_vector_fields!(fields.work_physical, fields.work_physical)
end


function compute_magnetic_curl_spectral!(mag_field, j_tor::SHTnsSpectralField{T}, 
                                        j_pol::SHTnsSpectralField{T}, l_factors) where T
    # Compute j = ∇ × B in spectral space
    
    mag_tor_real = parent(mag_field.toroidal.data_real)
    mag_tor_imag = parent(mag_field.toroidal.data_imag)
    mag_pol_real = parent(mag_field.poloidal.data_real)
    mag_pol_imag = parent(mag_field.poloidal.data_imag)
    
    j_tor_real = parent(j_tor.data_real)
    j_tor_imag = parent(j_tor.data_imag)
    j_pol_real = parent(j_pol.data_real)
    j_pol_imag = parent(j_pol.data_imag)
    
    lm_range = get_local_range(j_tor.pencil, 1)
    r_range  = get_local_range(j_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(mag_tor_real, 3)
                    # Curl in spectral space
                    j_tor_real[local_lm, 1, local_r] = l_factor * mag_pol_real[local_lm, 1, local_r]
                    j_tor_imag[local_lm, 1, local_r] = l_factor * mag_pol_imag[local_lm, 1, local_r]
                    j_pol_real[local_lm, 1, local_r] = -l_factor * mag_tor_real[local_lm, 1, local_r]
                    j_pol_imag[local_lm, 1, local_r] = -l_factor * mag_tor_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end



# ============================================================================
# Velocity Module using Optimized SHTns Transforms
# ============================================================================

using LinearAlgebra
using MPI

struct SHTnsVelocityFields{T}
    # Physical space velocities
    velocity::SHTnsVectorField{T}
    vorticity::SHTnsVectorField{T}
    
    # Spectral representation (toroidal-poloidal)
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
    
    # Nonlinear terms
    nl_toroidal::SHTnsSpectralField{T}
    nl_poloidal::SHTnsSpectralField{T}
    
    # Pressure (for pressure correction)
    pressure::SHTnsSpectralField{T}
    
    # Work arrays for efficient computation
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
    work_physical::SHTnsVectorField{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}  # l(l+1) values
    coriolis_factors::Matrix{Float64}  # Pre-computed Coriolis terms
    
    # Transform manager for efficient transforms
    transform_manager::SHTnsTransformManager{T}
end

function create_shtns_velocity_fields(::Type{T}, config::SHTnsConfig, 
                                      domain::RadialDomain, 
                                      pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Create vector fields
    velocity  = create_shtns_vector_field(T, config, domain, pencils)
    vorticity = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral fields
    toroidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    poloidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    pressure    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, domain, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(Float64, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end
    
    # Create transform manager
    transform_manager = get_transform_manager(T, config, pencil_spec)
    
    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal, 
                                  nl_toroidal, nl_poloidal, pressure,
                                  work_tor, work_pol, work_physical,
                                  l_factors, coriolis_factors, transform_manager)
end

# ============================================================================
# Main nonlinear computation using optimized transforms
# ============================================================================

function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field) where T
    # Zero work arrays once
    zero_velocity_work_arrays!(fields)
    
    # Step 1: Use optimized vector synthesis from shtns_transforms.jl
    shtns_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
    
    # Step 2: Compute vorticity using spectral curl
    compute_vorticity_spectral!(fields)
    
    # Step 3: Transform vorticity to physical space
    shtns_vector_synthesis!(fields.work_tor, fields.work_pol, fields.vorticity)
    
    # Step 4: Compute all nonlinear terms in physical space
    compute_all_nonlinear_terms_fused!(fields, temp_field, comp_field, mag_field)
    
    # Step 5: Use optimized vector analysis to go back to spectral
    shtns_vector_analysis!(fields.work_physical, fields.nl_toroidal, fields.nl_poloidal)
end

# ============================================================================
# Vorticity computation in spectral space
# ============================================================================

function compute_vorticity_spectral!(fields::SHTnsVelocityFields{T}) where T
    # Compute vorticity directly in spectral space using l(l+1) factors
    # ω = ∇ × u, which in toroidal-poloidal space has simple relationships
    
    # Get local data views
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    
    vort_tor_real = parent(fields.work_tor.data_real)
    vort_tor_imag = parent(fields.work_tor.data_imag)
    vort_pol_real = parent(fields.work_pol.data_real)
    vort_pol_imag = parent(fields.work_pol.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range = get_local_range(fields.toroidal.pencil, 3)
    
    # Apply curl relationships in spectral space
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Vorticity from velocity in spectral space
                    # ω_tor = l(l+1) * u_pol
                    # ω_pol = -l(l+1) * u_tor
                    vort_tor_real[local_lm, 1, local_r] = l_factor * pol_real[local_lm, 1, local_r]
                    vort_tor_imag[local_lm, 1, local_r] = l_factor * pol_imag[local_lm, 1, local_r]
                    vort_pol_real[local_lm, 1, local_r] = -l_factor * tor_real[local_lm, 1, local_r]
                    vort_pol_imag[local_lm, 1, local_r] = -l_factor * tor_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end

# ============================================================================
# Fused nonlinear term computation in physical space
# ============================================================================

function compute_all_nonlinear_terms_fused!(fields::SHTnsVelocityFields{T},
                                           temp_field, comp_field, mag_field) where T
    # Get all data views
    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    vort_r = parent(fields.vorticity.r_component.data)
    vort_θ = parent(fields.vorticity.θ_component.data)
    vort_φ = parent(fields.vorticity.φ_component.data)
    
    work_r = parent(fields.work_physical.r_component.data)
    work_θ = parent(fields.work_physical.θ_component.data)
    work_φ = parent(fields.work_physical.φ_component.data)
    
    # Get dimensions and ranges
    local_size = size(vel_r)
    nlat = fields.velocity.r_component.config.nlat
    
    # Main fused computation loop
    @inbounds for k in 1:local_size[3]
        for j in 1:local_size[2]
            # Get pre-computed Coriolis factors for this latitude
            if j <= nlat
                sin_theta = fields.coriolis_factors[1, j]
                cos_theta = fields.coriolis_factors[2, j]
            else
                sin_theta = 0.0
                cos_theta = 1.0
            end
            
            @simd for i in 1:local_size[1]
                if i <= length(vel_r) ÷ (local_size[2] * local_size[3])
                    idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                    
                    if idx <= length(vel_r)
                        # Load velocity and vorticity components
                        u_r = vel_r[idx]
                        u_θ = vel_θ[idx]
                        u_φ = vel_φ[idx]
                        
                        ω_r = vort_r[idx]
                        ω_θ = vort_θ[idx]
                        ω_φ = vort_φ[idx]
                        
                        # Advection: u × ω (scaled by Rossby number)
                        adv_r = d_Ro * (u_θ * ω_φ - u_φ * ω_θ)
                        adv_θ = d_Ro * (u_φ * ω_r - u_r * ω_φ)
                        adv_φ = d_Ro * (u_r * ω_θ - u_θ * ω_r)
                        
                        # Coriolis: -2Ω × u
                        cor_r = -2.0 * (-sin_theta * u_φ)
                        cor_θ = -2.0 * (cos_theta * u_φ)
                        cor_φ = -2.0 * (-cos_theta * u_θ + sin_theta * u_r)
                        
                        # Store combined result
                        work_r[idx] = adv_r + cor_r
                        work_θ[idx] = adv_θ + cor_θ
                        work_φ[idx] = adv_φ + cor_φ
                    end
                end
            end
        end
    end
    
    # Add buoyancy forces
    if temp_field !== nothing
        add_buoyancy_optimized!(work_r, temp_field, d_Ra * d_Pr)
    end
    
    if comp_field !== nothing
        add_buoyancy_optimized!(work_r, comp_field, d_Ra * d_Pr)  # Should use Ra_C
    end
    
    # Add Lorentz force if magnetic field present
    if mag_field !== nothing
        add_lorentz_force_spectral!(fields, mag_field)
    end
end

# =====================
# Addition of forces
# =====================
function add_thermal_buoyancy!(work_r::AbstractArray{T,3}, 
                                scalar_field, factor::Float64) where T
    # Get the scalar field data (temperature or composition)
    if isa(scalar_field, SHTnsPhysicalField)
        scalar_data = parent(scalar_field.data)
    else
        # If it's already in physical space from temperature module
        scalar_data = parent(scalar_field.temperature.data)
    end
    
    # Vectorized addition of buoyancy
    @inbounds @simd for idx in eachindex(work_r)
        if idx <= length(scalar_data)
            work_r[idx] += factor * scalar_data[idx]
        end
    end
end


function add_lorentz_force!(fields::SHTnsVelocityFields{T}, mag_field) where T
    # Compute Lorentz force in spectral space for efficiency
    # F = (∇ × B) × B / Pm
    
    # First compute current density j = ∇ × B in spectral space
    compute_magnetic_curl_spectral!(mag_field, fields.work_tor, fields.work_pol, fields.l_factors)
    
    # Transform j to physical space
    shtns_vector_synthesis!(fields.work_tor, fields.work_pol, fields.work_physical)
    
    # Also need B in physical space (should already be there from mag_field computation)
    # If not, transform it
    if !is_in_physical_space(mag_field.magnetic)
        shtns_vector_synthesis!(mag_field.toroidal, mag_field.poloidal, mag_field.magnetic)
    end
    
    # Compute j × B in physical space
    compute_cross_product_jB!(fields.work_physical, mag_field.magnetic, fields.work_physical)
    
    # Scale by 1/Pm
    scale_field!(fields.work_physical, 1.0 / d_Pm)
    
    # Add to existing forces
    add_vector_fields!(fields.work_physical, fields.work_physical)
end


function compute_magnetic_curl_spectral!(mag_field, j_tor::SHTnsSpectralField{T}, 
                                        j_pol::SHTnsSpectralField{T}, l_factors) where T
    # Compute j = ∇ × B in spectral space
    
    mag_tor_real = parent(mag_field.toroidal.data_real)
    mag_tor_imag = parent(mag_field.toroidal.data_imag)
    mag_pol_real = parent(mag_field.poloidal.data_real)
    mag_pol_imag = parent(mag_field.poloidal.data_imag)
    
    j_tor_real = parent(j_tor.data_real)
    j_tor_imag = parent(j_tor.data_imag)
    j_pol_real = parent(j_pol.data_real)
    j_pol_imag = parent(j_pol.data_imag)
    
    lm_range = get_local_range(j_tor.pencil, 1)
    r_range = get_local_range(j_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(mag_tor_real, 3)
                    # Curl in spectral space
                    j_tor_real[local_lm, 1, local_r] = l_factor * mag_pol_real[local_lm, 1, local_r]
                    j_tor_imag[local_lm, 1, local_r] = l_factor * mag_pol_imag[local_lm, 1, local_r]
                    j_pol_real[local_lm, 1, local_r] = -l_factor * mag_tor_real[local_lm, 1, local_r]
                    j_pol_imag[local_lm, 1, local_r] = -l_factor * mag_tor_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end


function compute_cross_product_jB!(j_field::SHTnsVectorField{T}, 
                                  B_field::SHTnsVectorField{T},
                                  output::SHTnsVectorField{T}) where T
    # Compute j × B in physical space
    
    j_r = parent(j_field.r_component.data)
    j_θ = parent(j_field.θ_component.data)
    j_φ = parent(j_field.φ_component.data)
    
    B_r = parent(B_field.r_component.data)
    B_θ = parent(B_field.θ_component.data)
    B_φ = parent(B_field.φ_component.data)
    
    out_r = parent(output.r_component.data)
    out_θ = parent(output.θ_component.data)
    out_φ = parent(output.φ_component.data)
    
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            # Cross product components
            out_r[idx] = j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx]
            out_θ[idx] = j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx]
            out_φ[idx] = j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx]
        end
    end
end


# ====================
# Utility functions
# ====================
function zero_velocity_work_arrays!(fields::SHTnsVelocityFields{T}) where T
    # Efficiently zero all work arrays
    fill!(parent(fields.work_tor.data_real), zero(T))
    fill!(parent(fields.work_tor.data_imag), zero(T))
    fill!(parent(fields.work_pol.data_real), zero(T))
    fill!(parent(fields.work_pol.data_imag), zero(T))
    
    fill!(parent(fields.work_physical.r_component.data), zero(T))
    fill!(parent(fields.work_physical.θ_component.data), zero(T))
    fill!(parent(fields.work_physical.φ_component.data), zero(T))
    
    fill!(parent(fields.advection_physical.r_component.data), zero(T))
    fill!(parent(fields.advection_physical.θ_component.data), zero(T))
    fill!(parent(fields.advection_physical.φ_component.data), zero(T))
    
    fill!(parent(fields.vort_toroidal.data_real), zero(T))
    fill!(parent(fields.vort_toroidal.data_imag), zero(T))
    fill!(parent(fields.vort_poloidal.data_real), zero(T))
    fill!(parent(fields.vort_poloidal.data_imag), zero(T))
end


function scale_field!(field::SHTnsVectorField{T}, factor::Float64) where T
    # Scale all components of a vector field
    parent(field.r_component.data) .*= factor
    parent(field.θ_component.data) .*= factor
    parent(field.φ_component.data) .*= factor
end


function add_vector_fields!(dest::SHTnsVectorField{T}, source::SHTnsVectorField{T}) where T
    # Add source to destination
    parent(dest.r_component.data) .+= parent(source.r_component.data)
    parent(dest.θ_component.data) .+= parent(source.θ_component.data)
    parent(dest.φ_component.data) .+= parent(source.φ_component.data)
end



# =====================================================
# Diagnostic functions using transform infrastructure
# =====================================================
function compute_kinetic_energy(fields::SHTnsVelocityFields{T}) where T
    # Efficient kinetic energy computation in spectral space
    
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    
    local_energy = zero(Float64)
    
    # Use l(l+1) weighting for proper spectral integration
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range  = get_local_range(fields.toroidal.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Properly weighted spectral energy
                    weight = 1.0 / max(l_factor, 1.0)
                    local_energy += weight * (tor_real[local_lm, 1, local_r]^2 + 
                                             tor_imag[local_lm, 1, local_r]^2 + 
                                             pol_real[local_lm, 1, local_r]^2 + 
                                             pol_imag[local_lm, 1, local_r]^2)
                end
            end
        end
    end
    
    # Global sum
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end


function compute_enstrophy(fields::SHTnsVelocityFields{T}) where T
    # Compute enstrophy from vorticity spectral coefficients
    
    # First ensure vorticity is computed
    compute_vorticity_spectral!(fields)
    
    vort_tor_real = parent(fields.work_tor.data_real)
    vort_tor_imag = parent(fields.work_tor.data_imag)
    vort_pol_real = parent(fields.work_pol.data_real)
    vort_pol_imag = parent(fields.work_pol.data_imag)
    
    local_enstrophy = zero(Float64)
    
    lm_range = get_local_range(fields.work_tor.pencil, 1)
    r_range = get_local_range(fields.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.work_tor.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(vort_tor_real, 3)
                    weight = 1.0 / max(l_factor, 1.0)
                    local_enstrophy += weight * (vort_tor_real[local_lm, 1, local_r]^2 + 
                                                vort_tor_imag[local_lm, 1, local_r]^2 + 
                                                vort_pol_real[local_lm, 1, local_r]^2 + 
                                                vort_pol_imag[local_lm, 1, local_r]^2)
                end
            end
        end
    end
    
    # Global sum
    return 0.5 * MPI.Allreduce(local_enstrophy, MPI.SUM, get_comm())
end

# # Export functions
# export SHTnsVelocityFields, create_shtns_velocity_fields
# export compute_velocity_nonlinear!, compute_velocity_nonlinear_batched!
# export compute_kinetic_energy, compute_enstrophy
# export zero_velocity_work_arrays!