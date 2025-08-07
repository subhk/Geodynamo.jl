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
    vort_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    vort_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, domain, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, domain, pencils)
    advection_physical = create_shtns_vector_field(T, config, domain, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Pre-compute Coriolis factors (sin(θ) and cos(θ))
    coriolis_factors = zeros(Float64, 2, config.nlat)
    for i in 1:config.nlat
        coriolis_factors[1, i] = sin(config.theta_grid[i])
        coriolis_factors[2, i] = cos(config.theta_grid[i])
    end
    
    # Create radial derivative matrices
    dr_matrix = create_derivative_matrix(1, domain)
    d2r_matrix = create_derivative_matrix(2, domain)
    laplacian_matrix = create_radial_laplacian(domain)
    
    # Create transform manager
    transform_manager = get_transform_manager(T, config, pencil_spec)
    
    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal,
                                  vort_toroidal, vort_poloidal,
                                  nl_toroidal, nl_poloidal,
                                  work_tor, work_pol, work_physical,
                                  advection_physical,
                                  l_factors, coriolis_factors,
                                  dr_matrix, d2r_matrix, laplacian_matrix,
                                  transform_manager)
end


# =============================
# Main nonlinear computation
# =============================
function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field,
                                    domain::RadialDomain) where T
    # Zero work arrays once
    zero_velocity_work_arrays!(fields)
    
    # Step 1: Use optimized vector synthesis from shtns_transforms.jl
    shtns_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
    
    # Step 2: Compute vorticity in spectral space with full derivatives
    compute_vorticity_spectral_full!(fields, domain)
    
    # Step 3: Transform vorticity to physical space
    shtns_vector_synthesis!(fields.vort_toroidal, fields.vort_poloidal, fields.vorticity)
    
    # Step 4: Compute all nonlinear terms in physical space with optimizations
    compute_all_nonlinear_terms!(fields, temp_field, comp_field, mag_field, domain)
    
    # Step 5: Use optimized vector analysis to go back to spectral
    shtns_vector_analysis!(fields.advection_physical, fields.nl_toroidal, fields.nl_poloidal)
end



# =================================================
# Vorticity computation with radial derivatives
# =================================================
function compute_vorticity_spectral_full!(fields::SHTnsVelocityFields{T}, 
                                         domain::RadialDomain) where T
    # Compute vorticity ω = ∇ × u in spectral space with full radial derivatives
    # For toroidal-poloidal decomposition:
    # ω_tor = [l(l+1)/r² - d²/dr² - 2/r d/dr] u_pol
    # ω_pol = -l(l+1)/r² u_tor
    
    # Get local data views
    u_tor_real = parent(fields.toroidal.data_real)
    u_tor_imag = parent(fields.toroidal.data_imag)
    u_pol_real = parent(fields.poloidal.data_real)
    u_pol_imag = parent(fields.poloidal.data_imag)
    
    ω_tor_real = parent(fields.vort_toroidal.data_real)
    ω_tor_imag = parent(fields.vort_toroidal.data_imag)
    ω_pol_real = parent(fields.vort_poloidal.data_real)
    ω_pol_imag = parent(fields.vort_poloidal.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range  = get_local_range(fields.toroidal.pencil, 3)
    
    nr = domain.N
    
    # Process each (l,m) mode
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            # Extract radial profiles
            pol_profile_real = extract_local_radial_profile(u_pol_real, local_lm, nr, r_range)
            pol_profile_imag = extract_local_radial_profile(u_pol_imag, local_lm, nr, r_range)
            tor_profile_real = extract_local_radial_profile(u_tor_real, local_lm, nr, r_range)
            tor_profile_imag = extract_local_radial_profile(u_tor_imag, local_lm, nr, r_range)
            
            # Compute radial derivatives for poloidal component
            dpol_dr_real   = apply_derivative_local(fields.dr_matrix, pol_profile_real)
            dpol_dr_imag   = apply_derivative_local(fields.dr_matrix, pol_profile_imag)
            d2pol_dr2_real = apply_derivative_local(fields.d2r_matrix, pol_profile_real)
            d2pol_dr2_imag = apply_derivative_local(fields.d2r_matrix, pol_profile_imag)
            
            # Compute vorticity components
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(ω_tor_real, 3)
                    r_inv = domain.r[r_idx, 3]   # 1/r
                    r_inv2 = domain.r[r_idx, 2]  # 1/r²
                    
                    # Toroidal vorticity from poloidal velocity (with full derivatives)
                    ω_tor_real[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_real[r_idx] 
                                                        - d2pol_dr2_real[r_idx] 
                                                        - 2.0 * r_inv * dpol_dr_real[r_idx])
                    ω_tor_imag[local_lm, 1, local_r] = (l_factor * r_inv2 * pol_profile_imag[r_idx] 
                                                        - d2pol_dr2_imag[r_idx] 
                                                        - 2.0 * r_inv * dpol_dr_imag[r_idx])
                    
                    # Poloidal vorticity from toroidal velocity
                    ω_pol_real[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_real[r_idx]
                    ω_pol_imag[local_lm, 1, local_r] = -l_factor * r_inv2 * tor_profile_imag[r_idx]
                end
            end
        end
    end
end


# ==========================================
# nonlinear term computation
# ==========================================
function compute_all_nonlinear_terms!(fields::SHTnsVelocityFields{T},
                                               temp_field, comp_field, mag_field,
                                               domain::RadialDomain) where T
    # Compute all forces in a single optimized loop
    
    # Get all data views
    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    vort_r = parent(fields.vorticity.r_component.data)
    vort_θ = parent(fields.vorticity.θ_component.data)
    vort_φ = parent(fields.vorticity.φ_component.data)
    
    adv_r = parent(fields.advection_physical.r_component.data)
    adv_θ = parent(fields.advection_physical.θ_component.data)
    adv_φ = parent(fields.advection_physical.φ_component.data)
    
    # Get dimensions
    local_size = size(vel_r)
    nlat = fields.velocity.r_component.config.nlat
    nlon = fields.velocity.r_component.config.nlon
    
    # Main fused computation loop with cache blocking
    @inbounds for k in 1:local_size[3]
        # Get radius for this level
        r_idx = k + first(get_local_range(fields.velocity.r_component.pencil, 3)) - 1
        if r_idx <= domain.N
            r = domain.r[r_idx, 4]
            r_inv = domain.r[r_idx, 3]
        else
            r = 1.0
            r_inv = 1.0
        end
        
        for j in 1:local_size[2]
            # Get pre-computed Coriolis factors for this latitude
            theta_idx = min(j, nlat)
            sin_theta = fields.coriolis_factors[1, theta_idx]
            cos_theta = fields.coriolis_factors[2, theta_idx]
            
            @simd for i in 1:local_size[1]
                linear_idx = i + (j-1)*local_size[1] + (k-1)*local_size[1]*local_size[2]
                
                if linear_idx <= length(vel_r)
                    # Load velocity and vorticity components
                    u_r = vel_r[linear_idx]
                    u_θ = vel_θ[linear_idx]
                    u_φ = vel_φ[linear_idx]
                    
                    ω_r = vort_r[linear_idx]
                    ω_θ = vort_θ[linear_idx]
                    ω_φ = vort_φ[linear_idx]
                    
                    # Advection: u × ω (scaled by Rossby number)
                    adv_r_val = d_Ro * (u_θ * ω_φ - u_φ * ω_θ)
                    adv_θ_val = d_Ro * (u_φ * ω_r - u_r * ω_φ)
                    adv_φ_val = d_Ro * (u_r * ω_θ - u_θ * ω_r)
                    
                    # Coriolis: -2Ω × u
                    cor_r = -2.0 * (-sin_theta * u_φ)
                    cor_θ = -2.0 * (cos_theta * u_φ)
                    cor_φ = -2.0 * (-cos_theta * u_θ + sin_theta * u_r)
                    
                    # Store combined result
                    adv_r[linear_idx] = adv_r_val + cor_r
                    adv_θ[linear_idx] = adv_θ_val + cor_θ
                    adv_φ[linear_idx] = adv_φ_val + cor_φ
                end
            end
        end
    end
    
    # Add buoyancy forces with proper scaling
    if temp_field !== nothing
        add_buoyancy_force!(adv_r, temp_field, d_Ra * d_Pr, domain)
    end
    
    if comp_field !== nothing
        add_buoyancy_force!(adv_r, comp_field, d_Ra_C * d_Sc, domain)
    end
    
    # Add Lorentz force if magnetic field present
    if mag_field !== nothing
        add_lorentz_force!(fields, mag_field, domain)
    end
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