# module MagneticField
#     using PencilArrays
#     using ..Parameters
#     using ..VariableTypes
#     using ..SHTnsSetup
#     using ..SHTnsTransforms
#     using ..Timestepping
#     using ..LinearOps
#     using ..PencilSetup
    
struct SHTnsMagneticFields{T}
    # Physical space magnetic field
    magnetic::SHTnsVectorField{T}
    current::SHTnsVectorField{T}
    
    # Spectral representation
    toroidal::SHTnsSpectralField{T}
    poloidal::SHTnsSpectralField{T}
    
    # Inner core fields
    ic_toroidal::SHTnsSpectralField{T}
    ic_poloidal::SHTnsSpectralField{T}
    
    # Nonlinear terms (induction)
    nl_toroidal::SHTnsSpectralField{T}
    nl_poloidal::SHTnsSpectralField{T}
    
    # Imposed field (if any)
    imposed_field::Union{SHTnsVectorField{T}, Nothing}
end

function create_shtns_magnetic_fields(::Type{T}, config::SHTnsConfig, 
                                        domain_oc::RadialDomain, 
                                        domain_ic::RadialDomain, 
                                        pencils, pencil_spec) where T

    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Physical space fields
    magnetic = create_shtns_vector_field(T, config, domain_oc, pencils)
    current  = create_shtns_vector_field(T, config, domain_oc, pencils)
    
    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    
    # Inner core fields (different domain)
    ic_toroidal = create_shtns_spectral_field(T, config, domain_ic, pencil_spec)
    ic_poloidal = create_shtns_spectral_field(T, config, domain_ic, pencil_spec)
    
    # Nonlinear terms
    nl_toroidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    
    imposed_field = nothing
    
    return SHTnsMagneticFields{T}(magnetic, current, 
                                toroidal, poloidal,
                                ic_toroidal, ic_poloidal, 
                                nl_toroidal, nl_poloidal,
                                imposed_field)
end


function compute_magnetic_nonlinear!(mag_fields::SHTnsMagneticFields{T}, 
                                    vel_fields, rotation_rate) where T
    # Convert spectral B to physical space
    shtns_vector_synthesis!(mag_fields.toroidal, mag_fields.poloidal, 
                            mag_fields.magnetic)
    
    # # Compute current density j = ∇ × B
    # compute_current_density!(mag_fields.magnetic, mag_fields.current)
    
    # # Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    # compute_induction_term!(mag_fields, vel_fields)

    # Compute induction locally
    compute_induction_term_local!(mag_fields, vel_fields)
    
    # # Inner core rotation effects
    # add_inner_core_rotation!(mag_fields, rotation_rate)

    # Inner core effects
    if rotation_rate != 0.0
        add_inner_core_rotation_local!(mag_fields, rotation_rate)
    end
    
    # Transform to spectral space
    shtns_vector_analysis!(mag_fields.magnetic, 
                        mag_fields.nl_toroidal, mag_fields.nl_poloidal)
end

# function compute_current_density!(magnetic::SHTnsVectorField{T}, current::SHTnsVectorField{T}) where T
#     # Compute j = ∇ × B using SHTns curl operations
#     # This would use spectral differentiation with SHTns
    
#     # Placeholder - would compute curl using SHTns spectral operators
#     for r_idx in magnetic.r_component.local_radial_range
#         # Curl computation using SHTns
#     end
# end

# # Vector spherical harmonic implementation
# function compute_current_density!(magnetic::SHTnsVectorField{T}, 
#                                            current::SHTnsVectorField{T}) where T
    
#     # Compute j = ∇ × B using SHTns curl operations
#     # Compute current density using vector spherical harmonic curl operation
#     # This is the most efficient approach for spectral methods
    
#     config = magnetic.r_component.config
    
#     # Get toroidal-poloidal representation of magnetic field
#     B_toroidal = create_temp_spectral_field(T, config, magnetic.r_component)
#     B_poloidal = create_temp_spectral_field(T, config, magnetic.r_component)
    
#     # Convert magnetic field to toroidal-poloidal decomposition
#     shtns_vector_analysis!(magnetic, B_toroidal, B_poloidal)
    
#     # Compute curl in toroidal-poloidal space
#     j_toroidal = similar(B_toroidal)
#     j_poloidal = similar(B_poloidal)
    
#     compute_vector_curl_toroidal_poloidal!(B_toroidal, B_poloidal, 
#                                           j_toroidal, j_poloidal, config)
    
#     # Convert back to physical vector components
#     shtns_vector_synthesis!(j_toroidal, j_poloidal, current)
# end

# function compute_vector_curl_toroidal_poloidal!(B_toroidal::SHTnsSpectralField{T}, 
#                                                B_poloidal::SHTnsSpectralField{T},
#                                                j_toroidal::SHTnsSpectralField{T}, 
#                                                j_poloidal::SHTnsSpectralField{T},
#                                                config::SHTnsConfig) where T
#     # Compute curl using vector spherical harmonic relationships
#     # For B = ∇ × (T r̂) + ∇ × ∇ × (P r̂), the curl has specific forms
    
#     @views for lm_idx in 1:B_toroidal.nlm
#         l = config.l_values[lm_idx]
#         m = config.m_values[lm_idx]
#         l_factor = Float64(l * (l + 1))
        
#         for r_idx in B_toroidal.local_radial_range
#             if r_idx <= size(B_toroidal.data_real, 3)
                
#                 # Vector spherical harmonic curl relationships
#                 # These involve radial derivatives and l,m factors
                
#                 T_B = complex(B_toroidal.data_real[lm_idx, 1, r_idx], 
#                              B_toroidal.data_imag[lm_idx, 1, r_idx])
#                 P_B = complex(B_poloidal.data_real[lm_idx, 1, r_idx], 
#                              B_poloidal.data_imag[lm_idx, 1, r_idx])
                
#                 # Simplified curl (full implementation needs proper radial derivatives)
#                 # j has toroidal component from poloidal B and vice versa
#                 T_j = l_factor * P_B  # Simplified relationship
#                 P_j = -l_factor * T_B  # Simplified relationship
                
#                 j_toroidal.data_real[lm_idx, 1, r_idx] = real(T_j)
#                 j_toroidal.data_imag[lm_idx, 1, r_idx] = imag(T_j)
#                 j_poloidal.data_real[lm_idx, 1, r_idx] = real(P_j)
#                 j_poloidal.data_imag[lm_idx, 1, r_idx] = imag(P_j)
#             end
#         end
#     end
# end

# # Utility function
# function get_radius_at_level(r_idx::Int, config::SHTnsConfig)
#     # Get radius at radial level r_idx
#     # This would come from the radial domain
#     # Placeholder implementation
#     return 0.5 + 0.5 * cos(π * (r_idx - 1) / (i_N - 1))  # Chebyshev grid
# end


function compute_induction_term_local!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
    if vel_fields === nothing
        return
    end
    
    # Get local data
    mag_r = parent(mag_fields.magnetic.r_component.data)
    mag_θ = parent(mag_fields.magnetic.θ_component.data)
    mag_φ = parent(mag_fields.magnetic.φ_component.data)
    
    vel_r = parent(vel_fields.velocity.r_component.data)
    vel_θ = parent(vel_fields.velocity.θ_component.data)
    vel_φ = parent(vel_fields.velocity.φ_component.data)
    
    # Compute u × B locally
    for idx in eachindex(mag_r)
        if idx <= length(vel_r)
            u_r = vel_r[idx]
            u_θ = vel_θ[idx]
            u_φ = vel_φ[idx]
            
            B_r = mag_r[idx]
            B_θ = mag_θ[idx]
            B_φ = mag_φ[idx]
            
            mag_r[idx] = u_θ * B_φ - u_φ * B_θ
            mag_θ[idx] = u_φ * B_r - u_r * B_φ
            mag_φ[idx] = u_r * B_θ - u_θ * B_r
        end
    end
end

# function compute_induction_term!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
#     # Compute u × B in physical space
#     vel = vel_fields.velocity
#     mag = mag_fields.magnetic
    
#     # u × B = (u_θ B_φ - u_φ B_θ, u_φ B_r - u_r B_φ, u_r B_θ - u_θ B_r)
#     for r_idx in mag.r_component.local_radial_range
#         if r_idx <= size(mag.r_component.data_r, 3) && r_idx <= size(vel.r_component.data_r, 3)
#             for j_phi in 1:mag.r_component.nlon, i_theta in 1:mag.r_component.nlat
#                 if (i_theta <= size(mag.r_component.data_r, 1) && 
#                     j_phi <= size(mag.r_component.data_r, 2) &&
#                     i_theta <= size(vel.r_component.data_r, 1) && 
#                     j_phi <= size(vel.r_component.data_r, 2))
                    
#                     u_r = vel.r_component.data_r[i_theta, j_phi, r_idx]
#                     u_θ = vel.θ_component.data_r[i_theta, j_phi, r_idx]
#                     u_φ = vel.φ_component.data_r[i_theta, j_phi, r_idx]
                    
#                     B_r = mag.r_component.data_r[i_theta, j_phi, r_idx]
#                     B_θ = mag.θ_component.data_r[i_theta, j_phi, r_idx]
#                     B_φ = mag.φ_component.data_r[i_theta, j_phi, r_idx]
                    
#                     # Store u × B temporarily in magnetic field
#                     mag.r_component.data_r[i_theta, j_phi, r_idx] = u_θ * B_φ - u_φ * B_θ
#                     mag.θ_component.data_r[i_theta, j_phi, r_idx] = u_φ * B_r - u_r * B_φ
#                     mag.φ_component.data_r[i_theta, j_phi, r_idx] = u_r * B_θ - u_θ * B_r
#                 end
#             end
#         end
#     end
# end


# function add_inner_core_rotation!(mag_fields::SHTnsMagneticFields{T}, Ω::Float64) where T
#     # Inner core rotation: -Ω × B_ic
#     # This affects the boundary conditions and coupling
    
#     # Rotation effects on inner core field (simplified)
#     for lm_idx in 1:mag_fields.ic_toroidal.nlm
#         for r_idx in mag_fields.ic_toroidal.local_radial_range
#             if r_idx <= size(mag_fields.ic_toroidal.data_real, 3)
#                 mag_fields.ic_toroidal.data_real[lm_idx, 1, r_idx] *= (1.0 - Ω * 1e-3)
#                 mag_fields.ic_poloidal.data_real[lm_idx, 1, r_idx] *= (1.0 - Ω * 1e-3)
#             end
#         end
#     end
# end


function add_inner_core_rotation_local!(mag_fields::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Apply rotation effects to inner core fields
    ic_tor_real = parent(mag_fields.ic_toroidal.data_real)
    ic_tor_imag = parent(mag_fields.ic_toroidal.data_imag)
    ic_pol_real = parent(mag_fields.ic_poloidal.data_real)
    ic_pol_imag = parent(mag_fields.ic_poloidal.data_imag)
    
    rotation_factor = 1.0 - Ω * 1e-3
    
    ic_tor_real .*= rotation_factor
    ic_tor_imag .*= rotation_factor
    ic_pol_real .*= rotation_factor
    ic_pol_imag .*= rotation_factor
end


#export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!


# end