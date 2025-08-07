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
        add_thermal_buoyancy_force!(adv_r, temp_field, d_Ra * d_Pr, domain)
    end
    
    if comp_field !== nothing
        add_buoyancy_force!(adv_r, comp_field, d_Ra_C * d_Sc, domain)
    end
    
    # Add Lorentz force if magnetic field present
    if mag_field !== nothing
        add_lorentz_force!(fields, mag_field, domain)
    end
end


# =====================================
# Thermal buoyancy force addition
# =====================================
function add_thermal_buoyancy_force!(force_r::AbstractArray{T,3}, 
                                      scalar_field, factor::Float64,
                                      domain::RadialDomain) where T
    # Add buoyancy force with proper radial scaling
    
    # Get scalar field data
    if isa(scalar_field, SHTnsPhysicalField)
        scalar_data = parent(scalar_field.data)
    else
        scalar_data = parent(scalar_field.temperature.data)
    end
    
    # Get local radial range
    r_range = get_local_range(scalar_field.pencil, 3)
    
    # Vectorized addition with radial dependence
    @inbounds @simd for idx in eachindex(force_r)
        if idx <= length(scalar_data)
            # Get radial index for this point
            k = ((idx - 1) ÷ (size(force_r, 1) * size(force_r, 2))) + 1
            r_idx = k + first(r_range) - 1
            
            if r_idx <= domain.N
                # Include radial dependence for spherical geometry
                r = domain.r[r_idx, 4]
                gravity_factor = r^2  # Gravity scales as r² in spherical geometry
                force_r[idx] += factor * gravity_factor * scalar_data[idx]
            else
                force_r[idx] += factor * scalar_data[idx]
            end
        end
    end
end


# ===============================
# Lorentz force computation
# ===============================
function add_lorentz_force!(fields::SHTnsVelocityFields{T}, 
                                     mag_field::SHTnsMagneticFields{T},
                                     domain::RadialDomain) where T
    # Compute Lorentz force F = (∇ × B) × B / Pm more efficiently
    
    # Step 1: Current density j = ∇ × B is already computed in mag_field
    # We can reuse it from the magnetic field computation
    
    # Step 2: Compute j × B in physical space
    j_r = parent(mag_field.current.r_component.data)
    j_θ = parent(mag_field.current.θ_component.data)
    j_φ = parent(mag_field.current.φ_component.data)
    
    B_r = parent(mag_field.magnetic.r_component.data)
    B_θ = parent(mag_field.magnetic.θ_component.data)
    B_φ = parent(mag_field.magnetic.φ_component.data)
    
    adv_r = parent(fields.advection_physical.r_component.data)
    adv_θ = parent(fields.advection_physical.θ_component.data)
    adv_φ = parent(fields.advection_physical.φ_component.data)
    
    # Fused loop for j × B / Pm
    Pm_inv = 1.0 / d_Pm
    
    @inbounds @simd for idx in eachindex(j_r)
        if idx <= length(B_r)
            # Add Lorentz force to existing forces
            adv_r[idx] += Pm_inv * (j_θ[idx] * B_φ[idx] - j_φ[idx] * B_θ[idx])
            adv_θ[idx] += Pm_inv * (j_φ[idx] * B_r[idx] - j_r[idx] * B_φ[idx])
            adv_φ[idx] += Pm_inv * (j_r[idx] * B_θ[idx] - j_θ[idx] * B_r[idx])
        end
    end
end


# =====================================
# Boundary conditions for velocity
# =====================================
function apply_velocity_boundary_conditions!(fields::SHTnsVelocityFields{T}, 
                                           domain::RadialDomain) where T
    # Apply no-slip or stress-free boundary conditions
    
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    
    lm_range = get_local_range(fields.poloidal.pencil, 1)
    r_range = get_local_range(fields.poloidal.pencil, 3)
    
    # No-penetration: poloidal field vanishes at boundaries
    if 1 in r_range
        local_r = 1 - first(r_range) + 1
        @inbounds for lm_idx in lm_range
            if lm_idx <= fields.poloidal.nlm
                local_lm = lm_idx - first(lm_range) + 1
                pol_real[local_lm, 1, local_r] = 0.0
                pol_imag[local_lm, 1, local_r] = 0.0
            end
        end
    end
    
    if domain.N in r_range
        local_r = domain.N - first(r_range) + 1
        @inbounds for lm_idx in lm_range
            if lm_idx <= fields.poloidal.nlm
                local_lm = lm_idx - first(lm_range) + 1
                pol_real[local_lm, 1, local_r] = 0.0
                pol_imag[local_lm, 1, local_r] = 0.0
            end
        end
    end
    
    # For no-slip: toroidal field also vanishes
    # For stress-free: d(rT)/dr = 0 at boundaries
    if i_vel_bc == 1  # No-slip
        if 1 in r_range
            local_r = 1 - first(r_range) + 1
            @inbounds for lm_idx in lm_range
                if lm_idx <= fields.toroidal.nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    tor_real[local_lm, 1, local_r] = 0.0
                    tor_imag[local_lm, 1, local_r] = 0.0
                end
            end
        end
        
        if domain.N in r_range
            local_r = domain.N - first(r_range) + 1
            @inbounds for lm_idx in lm_range
                if lm_idx <= fields.toroidal.nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    tor_real[local_lm, 1, local_r] = 0.0
                    tor_imag[local_lm, 1, local_r] = 0.0
                end
            end
        end
    elseif i_vel_bc == 2  # Stress-free
        apply_stress_free_bc!(fields, domain)
    end
end


function apply_stress_free_bc!(fields::SHTnsVelocityFields{T}, 
                               domain::RadialDomain) where T
    # Apply stress-free boundary conditions: d(rT)/dr = 0
    # This requires modifying the toroidal field near boundaries
    
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    nr = domain.N
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            # Extract radial profile
            tor_profile_real = extract_local_radial_profile(tor_real, local_lm, nr,
                                                           get_local_range(fields.toroidal.pencil, 3))
            
            # Apply stress-free correction
            apply_stress_free_correction!(tor_profile_real, domain)
            
            # Store corrected profile
            store_local_radial_profile!(tor_real, tor_profile_real, local_lm,
                                       get_local_range(fields.toroidal.pencil, 3))
            
            # Repeat for imaginary part
            if any(x -> abs(x) > 1e-12, view(tor_imag, local_lm, 1, :))
                tor_profile_imag = extract_local_radial_profile(tor_imag, local_lm, nr,
                                                               get_local_range(fields.toroidal.pencil, 3))
                apply_stress_free_correction!(tor_profile_imag, domain)
                store_local_radial_profile!(tor_imag, tor_profile_imag, local_lm,
                                           get_local_range(fields.toroidal.pencil, 3))
            end
        end
    end
end


function apply_stress_free_correction!(profile::Vector{T}, domain::RadialDomain) where T
    # Modify profile to satisfy d(rT)/dr = 0 at boundaries
    # This uses linear extrapolation near boundaries
    
    N = domain.N
    
    # Inner boundary: d(rT)/dr = 0 at r=ri
    r1 = domain.r[1, 4]
    r2 = domain.r[2, 4]
    profile[1] = profile[2] * r2 / r1  # Linear extrapolation
    
    # Outer boundary: d(rT)/dr = 0 at r=ro
    rN = domain.r[N, 4]
    rN1 = domain.r[N-1, 4]
    profile[N] = profile[N-1] * rN1 / rN  # Linear extrapolation
end


# ===========================================
# Helper functions for radial operations
# ===========================================
function extract_local_radial_profile(data::AbstractArray{T,3}, local_lm::Int, 
                                     nr::Int, r_range) where T
    profile = zeros(T, nr)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= nr
            profile[r_idx] = data[local_lm, 1, local_r]
        end
    end
    
    return profile
end


function store_local_radial_profile!(data::AbstractArray{T,3}, profile::Vector{T},
                                    local_lm::Int, r_range) where T
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        if local_r <= size(data, 3) && r_idx <= length(profile)
            data[local_lm, 1, local_r] = profile[r_idx]
        end
    end
end


function apply_derivative_local(matrix::BandedMatrix{T}, field::Vector{T}) where T
    # Apply banded derivative matrix
    N = matrix.size
    bandwidth = matrix.bandwidth
    result = zeros(T, N)
    
    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                result[i] += matrix.data[band_row, j] * field[j]
            end
        end
    end
    
    return result
end


function solve_helmholtz_equation(laplacian::BandedMatrix{T}, source::Vector{T},
                                 l_factor::Float64, domain::RadialDomain) where T
    # Solve (∇²_r - l(l+1)/r²) u = source
    # This is a simplified solver - in practice would use more sophisticated methods
    
    N = domain.N
    solution = zeros(T, N)
    
    # Build full operator for this l value
    operator = zeros(T, N, N)
    bandwidth = laplacian.bandwidth
    
    @inbounds for j in 1:N
        for i in max(1, j - bandwidth):min(N, j + bandwidth)
            band_row = bandwidth + 1 + i - j
            if 1 <= band_row <= 2*bandwidth + 1
                operator[i, j] = laplacian.data[band_row, j]
                if i == j
                    # Add -l(l+1)/r² term to diagonal
                    r_inv2 = domain.r[i, 2]
                    operator[i, j] -= l_factor * r_inv2
                end
            end
        end
    end
    
    # Apply boundary conditions (solution vanishes at boundaries)
    operator[1, :] .= 0.0
    operator[1, 1] = 1.0
    operator[N, :] .= 0.0
    operator[N, N] = 1.0
    source[1] = 0.0
    source[N] = 0.0
    
    # Solve linear system (would use iterative solver in practice)
    solution = operator \ source
    
    return solution
end


# =====================================================
# Diagnostic functions using transform infrastructure
# =====================================================
function compute_kinetic_energy(fields::SHTnsVelocityFields{T}, domain::RadialDomain) where T
    # Compute kinetic energy with proper radial integration
    
    tor_real = parent(fields.toroidal.data_real)
    tor_imag = parent(fields.toroidal.data_imag)
    pol_real = parent(fields.poloidal.data_real)
    pol_imag = parent(fields.poloidal.data_imag)
    
    local_energy = zero(Float64)
    
    lm_range = get_local_range(fields.toroidal.pencil, 1)
    r_range = get_local_range(fields.toroidal.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(l_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    # Include radial weight for spherical integration
                    r = domain.r[r_idx, 4]
                    r_weight = r^2 * domain.integration_weights[r_idx]
                    
                    local_energy += weight * r_weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        pol_real[local_lm, 1, local_r]^2 + 
                        pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Global sum
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
end


function compute_reynolds_stress(fields::SHTnsVelocityFields{T}) where T
    # Compute Reynolds stress tensor <u_i u_j>
    # This requires transforming to physical space and computing products
    
    vel_r = parent(fields.velocity.r_component.data)
    vel_θ = parent(fields.velocity.θ_component.data)
    vel_φ = parent(fields.velocity.φ_component.data)
    
    # Compute all 6 independent components
    R_rr = mean(vel_r .* vel_r)
    R_θθ = mean(vel_θ .* vel_θ)
    R_φφ = mean(vel_φ .* vel_φ)
    R_rθ = mean(vel_r .* vel_θ)
    R_rφ = mean(vel_r .* vel_φ)
    R_θφ = mean(vel_θ .* vel_φ)
    
    # Global averages
    R_rr = MPI.Allreduce(R_rr, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_θθ = MPI.Allreduce(R_θθ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_φφ = MPI.Allreduce(R_φφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_rθ = MPI.Allreduce(R_rθ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_rφ = MPI.Allreduce(R_rφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    R_θφ = MPI.Allreduce(R_θφ, MPI.SUM, get_comm()) / MPI.Comm_size(get_comm())
    
    return (R_rr, R_θθ, R_φφ, R_rθ, R_rφ, R_θφ)
end



function compute_enstrophy(fields::SHTnsVelocityFields{T}, domain::RadialDomain) where T
    # Compute enstrophy from vorticity spectral coefficients
    
    vort_tor_real = parent(fields.vort_toroidal.data_real)
    vort_tor_imag = parent(fields.vort_toroidal.data_imag)
    vort_pol_real = parent(fields.vort_poloidal.data_real)
    vort_pol_imag = parent(fields.vort_poloidal.data_imag)
    
    local_enstrophy = zero(Float64)
    
    lm_range = get_local_range(fields.vort_toroidal.pencil, 1)
    r_range = get_local_range(fields.vort_toroidal.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= fields.vort_toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = fields.l_factors[lm_idx]
            weight = 1.0 / max(l_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(vort_tor_real, 3)
                    r = domain.r[r_idx, 4]
                    r_weight = r^2 * domain.integration_weights[r_idx]
                    
                    local_enstrophy += weight * r_weight * (
                        vort_tor_real[local_lm, 1, local_r]^2 + 
                        vort_tor_imag[local_lm, 1, local_r]^2 + 
                        vort_pol_real[local_lm, 1, local_r]^2 + 
                        vort_pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    return 0.5 * MPI.Allreduce(local_enstrophy, MPI.SUM, get_comm())
end


# # Export functions
# export SHTnsVelocityFields, create_shtns_velocity_fields
# export compute_velocity_nonlinear!, compute_velocity_nonlinear_batched!
# export compute_kinetic_energy, compute_enstrophy
# export zero_velocity_work_arrays!