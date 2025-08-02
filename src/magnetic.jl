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
                                        domain_oc::RadialDomain, domain_ic::RadialDomain, 
                                        pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Physical space fields
    magnetic = create_shtns_vector_field(T, config, domain_oc, pencils)
    current = create_shtns_vector_field(T, config, domain_oc, pencils)
    
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
    
    return SHTnsMagneticFields{T}(magnetic, current, toroidal, poloidal,
                                    ic_toroidal, ic_poloidal, nl_toroidal, nl_poloidal,
                                    imposed_field)
end

function compute_magnetic_nonlinear!(mag_fields::SHTnsMagneticFields{T}, 
                                    vel_fields, rotation_rate) where T
    # Convert spectral B to physical space
    shtns_vector_synthesis!(mag_fields.toroidal, mag_fields.poloidal, mag_fields.magnetic)
    
    # Compute current density j = ∇ × B
    compute_current_density!(mag_fields.magnetic, mag_fields.current)
    
    # Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    compute_induction_term!(mag_fields, vel_fields)
    
    # Inner core rotation effects
    add_inner_core_rotation!(mag_fields, rotation_rate)
    
    # Transform to spectral space
    shtns_vector_analysis!(mag_fields.magnetic, mag_fields.nl_toroidal, mag_fields.nl_poloidal)
end

function compute_current_density!(magnetic::SHTnsVectorField{T}, current::SHTnsVectorField{T}) where T
    # Compute j = ∇ × B using SHTns curl operations
    # This would use spectral differentiation with SHTns
    
    # Placeholder - would compute curl using SHTns spectral operators
    for r_idx in magnetic.r_component.local_radial_range
        # Curl computation using SHTns
    end
end

function compute_induction_term!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
    # Compute u × B in physical space
    vel = vel_fields.velocity
    mag = mag_fields.magnetic
    
    # u × B = (u_θ B_φ - u_φ B_θ, u_φ B_r - u_r B_φ, u_r B_θ - u_θ B_r)
    for r_idx in mag.r_component.local_radial_range
        if r_idx <= size(mag.r_component.data_r, 3) && r_idx <= size(vel.r_component.data_r, 3)
            for j_phi in 1:mag.r_component.nlon, i_theta in 1:mag.r_component.nlat
                if (i_theta <= size(mag.r_component.data_r, 1) && 
                    j_phi <= size(mag.r_component.data_r, 2) &&
                    i_theta <= size(vel.r_component.data_r, 1) && 
                    j_phi <= size(vel.r_component.data_r, 2))
                    
                    u_r = vel.r_component.data_r[i_theta, j_phi, r_idx]
                    u_θ = vel.θ_component.data_r[i_theta, j_phi, r_idx]
                    u_φ = vel.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    B_r = mag.r_component.data_r[i_theta, j_phi, r_idx]
                    B_θ = mag.θ_component.data_r[i_theta, j_phi, r_idx]
                    B_φ = mag.φ_component.data_r[i_theta, j_phi, r_idx]
                    
                    # Store u × B temporarily in magnetic field
                    mag.r_component.data_r[i_theta, j_phi, r_idx] = u_θ * B_φ - u_φ * B_θ
                    mag.θ_component.data_r[i_theta, j_phi, r_idx] = u_φ * B_r - u_r * B_φ
                    mag.φ_component.data_r[i_theta, j_phi, r_idx] = u_r * B_θ - u_θ * B_r
                end
            end
        end
    end
end

function add_inner_core_rotation!(mag_fields::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Inner core rotation: -Ω × B_ic
    # This affects the boundary conditions and coupling
    
    # Rotation effects on inner core field (simplified)
    for lm_idx in 1:mag_fields.ic_toroidal.nlm
        for r_idx in mag_fields.ic_toroidal.local_radial_range
            if r_idx <= size(mag_fields.ic_toroidal.data_real, 3)
                mag_fields.ic_toroidal.data_real[lm_idx, 1, r_idx] *= (1.0 - Ω * 1e-3)
                mag_fields.ic_poloidal.data_real[lm_idx, 1, r_idx] *= (1.0 - Ω * 1e-3)
            end
        end
    end
end

#export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!

# end