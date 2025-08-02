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
                                        domain::RadialDomain, pencils, pencil_spec) where T
    pencil_θ, pencil_φ, pencil_r = pencils
    
    # Create vector field
    velocity = create_shtns_vector_field(T, config, domain, pencils)
    vorticity = create_shtns_vector_field(T, config, domain, pencils)
    
    # Spectral fields
    toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_toroidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    nl_poloidal = create_shtns_spectral_field(T, config, domain, pencil_spec)
    pressure = create_shtns_spectral_field(T, config, domain, pencil_spec)
    
    return SHTnsVelocityFields{T}(velocity, vorticity, toroidal, poloidal, 
                                    nl_toroidal, nl_poloidal, pressure)
end

function compute_velocity_nonlinear!(fields::SHTnsVelocityFields{T}, 
                                    temp_field, comp_field, mag_field) where T
    # Convert spectral toroidal/poloidal to physical velocity
    shtns_vector_synthesis!(fields.toroidal, fields.poloidal, fields.velocity)
    
    # Compute vorticity from velocity
    compute_vorticity!(fields.velocity, fields.vorticity)
    
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
        add_lorentz_force!(fields, mag_field)
    end
    
    # Transform nonlinear terms back to spectral space
    shtns_vector_analysis!(fields.velocity, fields.nl_toroidal, fields.nl_poloidal)
end

function compute_vorticity!(velocity::SHTnsVectorField{T}, vorticity::SHTnsVectorField{T}) where T
    # Compute vorticity = ∇ × u using SHTns
    # This would use SHTns curl operations
    
    # Simplified implementation - would use proper SHTns curl
    config = velocity.r_component.config
    
    # For now, copy structure (placeholder)
    for r_idx in velocity.r_component.local_radial_range
        # Compute curl components using SHTns operators
        # ω_r = (1/sin θ)(∂v_φ/∂θ - ∂(sin θ v_θ)/∂φ)
        # ω_θ = (1/sin θ)(∂v_r/∂φ) - ∂v_φ/∂r
        # ω_φ = ∂v_θ/∂r - (1/r)(∂v_r/∂θ)
        
        # This requires spectral differentiation - placeholder
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

function add_lorentz_force!(fields::SHTnsVelocityFields{T}, mag_field) where T
    # Add Lorentz force: j × B = (∇ × B) × B
    # This is the magnetic force per unit volume in the momentum equation
    
    if mag_field === nothing
        return  # No magnetic field, no Lorentz force
    end
    
    # Ensure magnetic field is in physical space
    shtns_vector_synthesis!(mag_field.toroidal, mag_field.poloidal, mag_field.magnetic)
    
    # Compute current density j = ∇ × B using SHTns spectral derivatives
    compute_current_density_shtns!(mag_field.magnetic, mag_field.current)
    
    # Compute Lorentz force j × B in physical space
    compute_cross_product_jxB!(mag_field.current, mag_field.magnetic, fields.velocity)
    
    # Scale by magnetic interaction parameter
    scale_lorentz_force!(fields.velocity, 1.0 / d_Pm)  # Inverse magnetic Reynolds number
end

# export SHTnsVelocityFields, create_shtns_velocity_fields, compute_velocity_nonlinear!

#end