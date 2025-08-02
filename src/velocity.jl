# ============================================================================
# Physics Modules with SHTns
# ============================================================================

# module Velocity
#     using PencilArrays
#     using ..Parameters
#     using ..VariableTypes
#     using ..SHTnsSetup
#     using ..SHTnsTransforms
#     using ..Timestepping
#     using ..LinearOps
#     using ..PencilSetup
    

# Velocity field components with SHTns
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
end

function create_shtns_velocity_fields(::Type{T}, config::SHTnsConfig, 
                                        domain::RadialDomain, 
                                        pencils, pencil_spec) where T

    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Create vector field
    velocity  = create_shtns_vector_field(T, config, domain, pencils)
    vorticity = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral fields
    toroidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    poloidal    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    pressure    = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal, 
                                    nl_toroidal, nl_poloidal, pressure)
end

function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field) where T
    # Convert spectral toroidal/poloidal to physical velocity
    shtns_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
    
    # Compute vorticity from velocity
    compute_vorticity_shtns_curl!(fields.velocity, fields.vorticity)
    
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

# Optimized version using SHTns built-in curl operations
function compute_vorticity_shtns_curl!(velocity::SHTnsVectorField{T}, 
                                      vorticity::SHTnsVectorField{T}) where T
    # Use SHTns built-in curl operations if available
    # This is the most efficient approach
    
    config = velocity.r_component.config
    sht = config.sht
    
    for r_idx in velocity.r_component.local_radial_range
        if r_idx <= size(velocity.r_component.data_r, 3)
            
            # Extract velocity field at this radial level
            v_theta = zeros(ComplexF64, config.nlat, config.nlon)
            v_phi = zeros(ComplexF64, config.nlat, config.nlon)
            
            for j_phi in 1:config.nlon, i_theta in 1:config.nlat
                if (i_theta <= size(velocity.θ_component.data_r, 1) && 
                    j_phi <= size(velocity.θ_component.data_r, 2))
                    
                    v_theta[i_theta, j_phi] = complex(velocity.θ_component.data_r[i_theta, j_phi, r_idx])
                    v_phi[i_theta, j_phi] = complex(velocity.φ_component.data_r[i_theta, j_phi, r_idx])
                end
            end
            
            # Use SHTns vector analysis to get toroidal-poloidal
            tor_coeffs, pol_coeffs = vector_analysis(sht, v_theta, v_phi)
            
            # Apply curl operation in spectral space
            # This would use proper vector spherical harmonic curl relations
            curl_tor_coeffs = similar(tor_coeffs)
            curl_pol_coeffs = similar(pol_coeffs)
            
            for lm_idx in 1:length(tor_coeffs)
                l = config.l_values[lm_idx]
                l_factor = Float64(l * (l + 1))
                
                # Simplified curl (full implementation needs radial derivatives)
                curl_pol_coeffs[lm_idx] = l_factor * tor_coeffs[lm_idx]
                curl_tor_coeffs[lm_idx] = -l_factor * pol_coeffs[lm_idx]
            end
            
            # Convert back to physical space
            omega_theta, omega_phi = vector_synthesis(sht, curl_tor_coeffs, curl_pol_coeffs)
            
            # Store vorticity components
            for j_phi in 1:config.nlon, i_theta in 1:config.nlat
                if (i_theta <= size(vorticity.θ_component.data_r, 1) && 
                    j_phi <= size(vorticity.θ_component.data_r, 2))
                    
                    vorticity.θ_component.data_r[i_theta, j_phi, r_idx] = real(omega_theta[i_theta, j_phi])
                    vorticity.φ_component.data_r[i_theta, j_phi, r_idx] = real(omega_phi[i_theta, j_phi])
                end
            end
            
            # Radial component would need additional computation
            compute_radial_vorticity_component!(velocity, vorticity, r_idx, config)
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

function add_coriolis_force!(fields::SHTnsVelocityFields{T}) where T
    # Coriolis force: -2Ω × u assuming rotation about z-axis
    # In spherical coordinates: Ω = Ω(cos θ ê_r - sin θ ê_θ)
    
    vel = fields.velocity
    config = vel.r_component.config
    
    for r_idx in vel.r_component.local_radial_range
        if r_idx <= size(vel.r_component.data_r, 3)
            for j_phi in 1:vel.r_component.nlon, i_theta in 1:vel.r_component.nlat
                if i_theta <= size(vel.r_component.data_r, 1) && j_phi <= size(vel.r_component.data_r, 2)
                    # Get θ coordinate from SHTns grid
                    theta = config.theta_grid[i_theta]
                    cos_theta = cos(theta)
                    sin_theta = sin(theta)
                    
                    u_r = vel.r_component.data_r[i_theta, j_phi, r_idx]
                    u_θ = vel.θ_component.data_r[i_theta, j_phi, r_idx]
                    u_φ = vel.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    # Coriolis force components
                    coriolis_r = -2.0 * (-sin_theta * u_φ)
                    coriolis_θ = -2.0 * (cos_theta * u_φ)
                    coriolis_φ = -2.0 * (-cos_theta * u_θ + sin_theta * u_r)
                    
                    vel.r_component.data_r[i_theta, j_phi, r_idx] += coriolis_r
                    vel.θ_component.data_r[i_theta, j_phi, r_idx] += coriolis_θ
                    vel.φ_component.data_r[i_theta, j_phi, r_idx] += coriolis_φ
                end
            end
        end
    end
end

function add_thermal_buoyancy!(fields::SHTnsVelocityFields{T}, temp_field) where T
    # Add thermal buoyancy: Ra_T * Pr * T * ê_r

    if temp_field !== nothing
        buoyancy_factor = d_Ra * d_Pr
        vel = fields.velocity
        
        for r_idx in vel.r_component.local_radial_range
            if r_idx <= size(vel.r_component.data_r, 3) && r_idx <= size(temp_field.data_r, 3)
                for j_phi in 1:vel.r_component.nlon, i_theta in 1:vel.r_component.nlat
                    if (i_theta <= size(vel.r_component.data_r, 1) && 
                        j_phi <= size(vel.r_component.data_r, 2) &&
                        i_theta <= size(temp_field.data_r, 1) && 
                        j_phi <= size(temp_field.data_r, 2))
                        
                        vel.r_component.data_r[i_theta, j_phi, r_idx] += 
                            buoyancy_factor * temp_field.data_r[i_theta, j_phi, r_idx]
                    end
                end
            end
        end
    end
end

function add_compositional_buoyancy!(fields::SHTnsVelocityFields{T}, comp_field) where T
    # Add compositional buoyancy: Ra_C * Pr * C * ê_r  
    buoyancy_factor = d_Ra * d_Pr  # Should be separate Ra_C parameter
    vel = fields.velocity
    
    for r_idx in vel.r_component.local_radial_range
        if r_idx <= size(vel.r_component.data_r, 3) && r_idx <= size(comp_field.data_r, 3)
            for j_phi in 1:vel.r_component.nlon, i_theta in 1:vel.r_component.nlat
                if (i_theta <= size(vel.r_component.data_r, 1) && 
                    j_phi <= size(vel.r_component.data_r, 2) &&
                    i_theta <= size(comp_field.data_r, 1) && 
                    j_phi <= size(comp_field.data_r, 2))
                    
                    vel.r_component.data_r[i_theta, j_phi, r_idx] += 
                        buoyancy_factor * comp_field.data_r[i_theta, j_phi, r_idx]
                end
            end
        end
    end
end

# function add_lorentz_force!(fields::SHTnsVelocityFields{T}, mag_field) where T
#     # Add Lorentz force: j × B = (∇ × B) × B
#     # This is the magnetic force per unit volume in the momentum equation
    
#     if mag_field === nothing
#         return  # No magnetic field, no Lorentz force
#     end
    
#     # Ensure magnetic field is in physical space
#     shtns_vector_synthesis!(mag_field.toroidal, mag_field.poloidal, mag_field.magnetic)
    
#     # Compute current density j = ∇ × B using SHTns spectral derivatives
#     compute_current_density_shtns!(mag_field.magnetic, mag_field.current)
    
#     # Compute Lorentz force j × B in physical space
#     compute_cross_product_jxB!(mag_field.current, mag_field.magnetic, fields.velocity)
    
#     # Scale by magnetic interaction parameter
#     scale_lorentz_force!(fields.velocity, 1.0 / d_Pm)  # Inverse magnetic Reynolds number
# end


# function compute_current_density_shtns!(magnetic::SHTnsVectorField{T}, 
#                                 current::SHTnsVectorField{T}) where T

#     # Compute current density j = ∇ × B using SHTns spectral operations
#     # This is more accurate than finite differences for smooth fields
    
#     config = magnetic.r_component.config
#     sht = config.sht
#     nlm = config.nlm
    
#     # Create temporary spectral fields for each magnetic component
#     B_r_spec = create_shtns_spectral_field(T, config, 
#                                           RadialDomain(i_N, 1:i_N, zeros(i_N, 7), 
#                                                       [], zeros(2*i_KL+1, i_N), zeros(i_N)),
#                                           magnetic.r_component.data_r)
#     B_θ_spec = create_shtns_spectral_field(T, config, 
#                                           RadialDomain(i_N, 1:i_N, zeros(i_N, 7), 
#                                                       [], zeros(2*i_KL+1, i_N), zeros(i_N)),
#                                           magnetic.θ_component.data_r)
#     B_φ_spec = create_shtns_spectral_field(T, config, 
#                                           RadialDomain(i_N, 1:i_N, zeros(i_N, 7), 
#                                                       [], zeros(2*i_KL+1, i_N), zeros(i_N)),
#                                           magnetic.φ_component.data_r)
    
#     # Convert magnetic components to spectral space
#     shtns_physical_to_spectral!(magnetic.r_component, B_r_spec)
#     shtns_physical_to_spectral!(magnetic.θ_component, B_θ_spec)
#     shtns_physical_to_spectral!(magnetic.φ_component, B_φ_spec)
    
#     # Process each radial level
#     @views for r_idx in magnetic.r_component.local_radial_range
#         if r_idx <= size(B_r_spec.data_real, 3)
            
#             # Get radius for this level
#             r = magnetic.r_component.config.theta_grid[1]  # Placeholder - would get from domain
#             r_inv = 1.0 / max(r, 1e-10)
            
#             # Prepare spectral coefficients for this radial level
#             B_r_coeffs = zeros(ComplexF64, nlm)
#             B_θ_coeffs = zeros(ComplexF64, nlm)
#             B_φ_coeffs = zeros(ComplexF64, nlm)
            
#             for lm_idx in 1:nlm
#                 B_r_coeffs[lm_idx] = complex(B_r_spec.data_real[lm_idx, 1, r_idx], 
#                                            B_r_spec.data_imag[lm_idx, 1, r_idx])
#                 B_θ_coeffs[lm_idx] = complex(B_θ_spec.data_real[lm_idx, 1, r_idx], 
#                                            B_θ_spec.data_imag[lm_idx, 1, r_idx])
#                 B_φ_coeffs[lm_idx] = complex(B_φ_spec.data_real[lm_idx, 1, r_idx], 
#                                            B_φ_spec.data_imag[lm_idx, 1, r_idx])
#             end
            
#             # Compute curl components using SHTns spectral derivatives
#             # j_r = (1/(r sin θ)) * [∂B_φ/∂θ - ∂(sin θ B_θ)/∂φ]
#             j_r_coeffs = compute_curl_r_component(sht, B_θ_coeffs, B_φ_coeffs, r_inv)
            
#             # j_θ = (1/r) * [∂B_r/∂φ/(sin θ) - ∂(r B_φ)/∂r]
#             j_θ_coeffs = compute_curl_theta_component(sht, B_r_coeffs, B_φ_coeffs, r, r_inv)
            
#             # j_φ = (1/r) * [∂(r B_θ)/∂r - ∂B_r/∂θ]
#             j_φ_coeffs = compute_curl_phi_component(sht, B_r_coeffs, B_θ_coeffs, r, r_inv)
            
#             # Convert back to physical space and store
#             j_r_phys = synthesis(sht, j_r_coeffs)
#             j_θ_phys = synthesis(sht, j_θ_coeffs)
#             j_φ_phys = synthesis(sht, j_φ_coeffs)
            
#             # Store in current field
#             for j_phi in 1:current.r_component.nlon, i_theta in 1:current.r_component.nlat
#                 if (i_theta <= size(current.r_component.data_r, 1) && 
#                     j_phi <= size(current.r_component.data_r, 2))
                    
#                     current.r_component.data_r[i_theta, j_phi, r_idx] = real(j_r_phys[i_theta, j_phi])
#                     current.θ_component.data_r[i_theta, j_phi, r_idx] = real(j_θ_phys[i_theta, j_phi])
#                     current.φ_component.data_r[i_theta, j_phi, r_idx] = real(j_φ_phys[i_theta, j_phi])
#                 end
#             end
#         end
#     end
# end


# function compute_curl_r_component(sht, B_θ_coeffs::Vector{ComplexF64}, 
#                                  B_φ_coeffs::Vector{ComplexF64}, r_inv::Float64)
#     # j_r = (1/(r sin θ)) * [∂B_φ/∂θ - ∂(sin θ B_θ)/∂φ]
    
#     # Compute ∂B_φ/∂θ using SHTns
#     dB_φ_dtheta_phys = synthesis_dtheta(sht, B_φ_coeffs)
    
#     # Compute ∂(sin θ B_θ)/∂φ
#     # First convert B_θ to physical space, multiply by sin θ, then take ∂/∂φ
#     B_θ_phys = synthesis(sht, B_θ_coeffs)
    
#     # Get grid information
#     nlat, nlon = size(B_θ_phys)
#     theta_grid = get_theta_array(sht)
    
#     # Multiply by sin θ
#     sinθ_B_θ = zeros(ComplexF64, nlat, nlon)
#     for i_theta in 1:nlat
#         sin_theta = sin(theta_grid[i_theta])
#         for j_phi in 1:nlon
#             sinθ_B_θ[i_theta, j_phi] = sin_theta * B_θ_phys[i_theta, j_phi]
#         end
#     end
    
#     # Convert back to spectral and take φ derivative
#     sinθ_B_θ_coeffs = analysis(sht, sinθ_B_θ)
#     d_sinθB_θ_dphi_phys = synthesis_dphi(sht, sinθ_B_θ_coeffs)
    
#     # Compute j_r
#     j_r_phys = zeros(ComplexF64, nlat, nlon)
#     for i_theta in 1:nlat
#         sin_theta = sin(theta_grid[i_theta])
#         sin_theta_inv = 1.0 / max(sin_theta, 1e-10)
#         for j_phi in 1:nlon
#             j_r_phys[i_theta, j_phi] = r_inv * sin_theta_inv * 
#                 (dB_φ_dtheta_phys[i_theta, j_phi] - d_sinθB_θ_dphi_phys[i_theta, j_phi])
#         end
#     end
    
#     # Convert back to spectral coefficients
#     return analysis(sht, j_r_phys)
# end


# function compute_curl_theta_component(sht, B_r_coeffs::Vector{ComplexF64}, 
#                                      B_φ_coeffs::Vector{ComplexF64}, r::Float64, r_inv::Float64)
#     # j_θ = (1/r) * [∂B_r/∂φ/(sin θ) - ∂(r B_φ)/∂r]
    
#     # Get grid
#     nlat, nlon = size(synthesis(sht, B_r_coeffs))
#     theta_grid = get_theta_array(sht)
    
#     # Compute ∂B_r/∂φ
#     dB_r_dphi_phys = synthesis_dphi(sht, B_r_coeffs)
    
#     # Divide by sin θ
#     dB_r_dphi_over_sinθ = zeros(ComplexF64, nlat, nlon)
#     for i_theta in 1:nlat
#         sin_theta = sin(theta_grid[i_theta])
#         sin_theta_inv = 1.0 / max(sin_theta, 1e-10)
#         for j_phi in 1:nlon
#             dB_r_dphi_over_sinθ[i_theta, j_phi] = sin_theta_inv * dB_r_dphi_phys[i_theta, j_phi]
#         end
#     end
    
#     # For ∂(r B_φ)/∂r, we need radial derivative
#     # This requires finite differences in radial direction
#     # For now, approximate as r * ∂B_φ/∂r + B_φ ≈ B_φ (simplified)
#     B_φ_phys = synthesis(sht, B_φ_coeffs)
    
#     # Compute j_θ  
#     j_θ_phys = zeros(ComplexF64, nlat, nlon)
#     for i_theta in 1:nlat, j_phi in 1:nlon
#         j_θ_phys[i_theta, j_phi] = r_inv * 
#             (dB_r_dphi_over_sinθ[i_theta, j_phi] - B_φ_phys[i_theta, j_phi])
#     end
    
#     return analysis(sht, j_θ_phys)
# end


# function compute_curl_phi_component(sht, B_r_coeffs::Vector{ComplexF64}, 
#                                    B_θ_coeffs::Vector{ComplexF64}, r::Float64, r_inv::Float64)
#     # j_φ = (1/r) * [∂(r B_θ)/∂r - ∂B_r/∂θ]
    
#     # Compute ∂B_r/∂θ
#     dB_r_dtheta_phys = synthesis_dtheta(sht, B_r_coeffs)
    
#     # For ∂(r B_θ)/∂r, approximate as B_θ (simplified)
#     B_θ_phys = synthesis(sht, B_θ_coeffs)
    
#     # Compute j_φ
#     nlat, nlon = size(B_θ_phys)
#     j_φ_phys = zeros(ComplexF64, nlat, nlon)
#     for i_theta in 1:nlat, j_phi in 1:nlon
#         j_φ_phys[i_theta, j_phi] = r_inv * 
#             (B_θ_phys[i_theta, j_phi] - dB_r_dtheta_phys[i_theta, j_phi])
#     end
    
#     return analysis(sht, j_φ_phys)
# end

# function compute_cross_product_jxB!(current::SHTnsVectorField{T}, 
#                                    magnetic::SHTnsVectorField{T},
#                                    velocity::SHTnsVectorField{T}) where T
#     # Compute Lorentz force: F = j × B
#     # F_r = j_θ B_φ - j_φ B_θ
#     # F_θ = j_φ B_r - j_r B_φ  
#     # F_φ = j_r B_θ - j_θ B_r
    
#     for r_idx in current.r_component.local_radial_range
#         if (r_idx <= size(current.r_component.data_r, 3) && 
#             r_idx <= size(magnetic.r_component.data_r, 3) &&
#             r_idx <= size(velocity.r_component.data_r, 3))
            
#             for j_phi in 1:current.r_component.nlon, i_theta in 1:current.r_component.nlat
#                 if (i_theta <= size(current.r_component.data_r, 1) && 
#                     j_phi <= size(current.r_component.data_r, 2) &&
#                     i_theta <= size(magnetic.r_component.data_r, 1) && 
#                     j_phi <= size(magnetic.r_component.data_r, 2) &&
#                     i_theta <= size(velocity.r_component.data_r, 1) && 
#                     j_phi <= size(velocity.r_component.data_r, 2))
                    
#                     # Current density components
#                     j_r = current.r_component.data_r[i_theta, j_phi, r_idx]
#                     j_θ = current.θ_component.data_r[i_theta, j_phi, r_idx]
#                     j_φ = current.φ_component.data_r[i_theta, j_phi, r_idx]
                    
#                     # Magnetic field components
#                     B_r = magnetic.r_component.data_r[i_theta, j_phi, r_idx]
#                     B_θ = magnetic.θ_component.data_r[i_theta, j_phi, r_idx]
#                     B_φ = magnetic.φ_component.data_r[i_theta, j_phi, r_idx]
                    
#                     # Compute cross product j × B
#                     F_r = j_θ * B_φ - j_φ * B_θ
#                     F_θ = j_φ * B_r - j_r * B_φ
#                     F_φ = j_r * B_θ - j_θ * B_r
                    
#                     # Add to velocity equation (momentum equation)
#                     velocity.r_component.data_r[i_theta, j_phi, r_idx] += F_r
#                     velocity.θ_component.data_r[i_theta, j_phi, r_idx] += F_θ
#                     velocity.φ_component.data_r[i_theta, j_phi, r_idx] += F_φ
#                 end
#             end
#         end
#     end
# end

# function scale_lorentz_force!(velocity::SHTnsVectorField{T}, scale_factor::Float64) where T
#     # Scale the Lorentz force by the magnetic interaction parameter
#     # Typically 1/Pm (inverse magnetic Reynolds number) or Ha²/Re (Hartmann number squared / Reynolds number)
    
#     for r_idx in velocity.r_component.local_radial_range
#         if r_idx <= size(velocity.r_component.data_r, 3)
#             for j_phi in 1:velocity.r_component.nlon, i_theta in 1:velocity.r_component.nlat
#                 if (i_theta <= size(velocity.r_component.data_r, 1) && 
#                     j_phi <= size(velocity.r_component.data_r, 2))
                    
#                     velocity.r_component.data_r[i_theta, j_phi, r_idx] *= scale_factor
#                     velocity.θ_component.data_r[i_theta, j_phi, r_idx] *= scale_factor
#                     velocity.φ_component.data_r[i_theta, j_phi, r_idx] *= scale_factor
#                 end
#             end
#         end
#     end
# end


# # Additional utility function for computing radial derivatives needed in curl
# function compute_radial_derivative!(input_field::SHTnsSpectralField{T}, 
#                                    output_field::SHTnsSpectralField{T},
#                                    domain::RadialDomain) where T
#     # Compute radial derivative using finite differences
#     # This is used in the curl computation where spectral methods don't apply (radial direction)
    
#     dr_matrix = create_derivative_matrix(1, domain)
    
#     # Apply radial derivative matrix to each spectral mode
#     @views for lm_idx in 1:input_field.nlm
#         # Real part
#         apply_banded_vector!(output_field.data_real[lm_idx, 1, :], 
#                            dr_matrix, input_field.data_real[lm_idx, 1, :])
        
#         # Imaginary part
#         apply_banded_vector!(output_field.data_imag[lm_idx, 1, :], 
#                            dr_matrix, input_field.data_imag[lm_idx, 1, :])
#     end
# end


# Alternative implementation using vector spherical harmonic transforms
function add_lorentz_force_vectorSH!(fields::SHTnsVelocityFields{T}, mag_field) where T
    # Alternative implementation using vector spherical harmonic decomposition
    # This is more efficient for vector fields and maintains spectral accuracy
    
    if mag_field === nothing
        return
    end
    
    # Compute current density in spectral space using vector curl
    j_toroidal = similar(mag_field.toroidal)
    j_poloidal = similar(mag_field.poloidal)
    
    # Vector curl: j = ∇ × B
    compute_vector_curl_shtns!(mag_field.toroidal, mag_field.poloidal, 
                              j_toroidal, j_poloidal)
    
    # Compute Lorentz force: F = j × B in vector spectral space
    F_toroidal = similar(fields.toroidal)
    F_poloidal = similar(fields.poloidal)
    
    compute_vector_cross_product_shtns!(j_toroidal, j_poloidal,
                                       mag_field.toroidal, mag_field.poloidal,
                                       F_toroidal, F_poloidal)
    
    # Add to velocity equation
    scale_factor = 1.0 / d_Pm
    @views for lm_idx in 1:fields.toroidal.nlm
        for r_idx in fields.toroidal.local_radial_range
            if r_idx <= size(fields.toroidal.data_real, 3)
                fields.toroidal.data_real[lm_idx, 1, r_idx] += 
                    scale_factor * F_toroidal.data_real[lm_idx, 1, r_idx]
                fields.toroidal.data_imag[lm_idx, 1, r_idx] += 
                    scale_factor * F_toroidal.data_imag[lm_idx, 1, r_idx]
                
                fields.poloidal.data_real[lm_idx, 1, r_idx] += 
                    scale_factor * F_poloidal.data_real[lm_idx, 1, r_idx]
                fields.poloidal.data_imag[lm_idx, 1, r_idx] += 
                    scale_factor * F_poloidal.data_imag[lm_idx, 1, r_idx]
            end
        end
    end
end


function compute_vector_curl_shtns!(B_toroidal::SHTnsSpectralField{T}, 
                                   B_poloidal::SHTnsSpectralField{T},
                                   j_toroidal::SHTnsSpectralField{T}, 
                                   j_poloidal::SHTnsSpectralField{T}) where T
    # Compute curl of vector field in vector spherical harmonic space
    # For toroidal-poloidal decomposition:
    # If B = ∇ × (T r̂) + ∇ × ∇ × (P r̂)
    # Then ∇ × B has specific relationships between T,P and curl components
    
    config = B_toroidal.config
    
    @views for lm_idx in 1:B_toroidal.nlm
        l = config.l_values[lm_idx]
        m = config.m_values[lm_idx]
        
        # Vector spherical harmonic curl operations
        # This involves derivatives and l,m-dependent operations
        l_factor = Float64(l * (l + 1))
        
        for r_idx in B_toroidal.local_radial_range
            if r_idx <= size(B_toroidal.data_real, 3)
                # Simplified curl operation (would need full vector SH implementation)
                # j_toroidal comes from poloidal component derivatives
                j_toroidal.data_real[lm_idx, 1, r_idx] = 
                    l_factor * B_poloidal.data_real[lm_idx, 1, r_idx]
                j_toroidal.data_imag[lm_idx, 1, r_idx] = 
                    l_factor * B_poloidal.data_imag[lm_idx, 1, r_idx]
                
                # j_poloidal comes from toroidal component derivatives  
                j_poloidal.data_real[lm_idx, 1, r_idx] = 
                    -l_factor * B_toroidal.data_real[lm_idx, 1, r_idx]
                j_poloidal.data_imag[lm_idx, 1, r_idx] = 
                    -l_factor * B_toroidal.data_imag[lm_idx, 1, r_idx]
            end
        end
    end
end


function compute_vector_cross_product_shtns!(j_toroidal::SHTnsSpectralField{T}, 
                                            j_poloidal::SHTnsSpectralField{T},
                                            B_toroidal::SHTnsSpectralField{T}, 
                                            B_poloidal::SHTnsSpectralField{T},
                                            F_toroidal::SHTnsSpectralField{T}, 
                                            F_poloidal::SHTnsSpectralField{T}) where T

    # Compute cross product j × B in vector spherical harmonic space
    # This is a complex operation involving coupling between different l,m modes
    # For full implementation, would need vector spherical harmonic coupling coefficients
    
    # Simplified implementation - direct mode-by-mode operation
    @views for lm_idx in 1:j_toroidal.nlm
        for r_idx in j_toroidal.local_radial_range
            if r_idx <= size(j_toroidal.data_real, 3)
                # Simplified cross product (would need proper vector SH coupling)
                F_toroidal.data_real[lm_idx, 1, r_idx] = 
                    j_toroidal.data_real[lm_idx, 1, r_idx] * B_poloidal.data_real[lm_idx, 1, r_idx] -
                    j_toroidal.data_imag[lm_idx, 1, r_idx] * B_poloidal.data_imag[lm_idx, 1, r_idx]
                
                F_toroidal.data_imag[lm_idx, 1, r_idx] = 
                    j_toroidal.data_real[lm_idx, 1, r_idx] * B_poloidal.data_imag[lm_idx, 1, r_idx] +
                    j_toroidal.data_imag[lm_idx, 1, r_idx] * B_poloidal.data_real[lm_idx, 1, r_idx]
                
                F_poloidal.data_real[lm_idx, 1, r_idx] = 
                    j_poloidal.data_real[lm_idx, 1, r_idx] * B_toroidal.data_real[lm_idx, 1, r_idx] -
                    j_poloidal.data_imag[lm_idx, 1, r_idx] * B_toroidal.data_imag[lm_idx, 1, r_idx]
                
                F_poloidal.data_imag[lm_idx, 1, r_idx] = 
                    j_poloidal.data_real[lm_idx, 1, r_idx] * B_toroidal.data_imag[lm_idx, 1, r_idx] +
                    j_poloidal.data_imag[lm_idx, 1, r_idx] * B_toroidal.data_real[lm_idx, 1, r_idx]
            end
        end
    end
end



# export SHTnsVelocityFields, create_shtns_velocity_fields, compute_velocity_nonlinear!

#end