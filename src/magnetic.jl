# =========================================
#  Magnetic field components with SHTns
# =========================================
#
# Transform Flow:
# ===============
#  Spectral B(tor/pol) → [shtns_vector_synthesis!] → Physical B
#                             ↓
#                   Compute j = ∇×B in spectral
#                             ↓
#               Compute u×B in physical space
#                             ↓
# Physical (u×B) → [shtns_vector_analysis!] → Spectral (u×B)
#                             ↓
#               Compute ∇×(u×B) in spectral
#                             ↓
#                   Add to nonlinear terms
#
#
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
    
    # Work arrays
    work_tor::SHTnsSpectralField{T}
    work_pol::SHTnsSpectralField{T}
    work_physical::SHTnsVectorField{T}
    
    # Pre-computed coefficients
    l_factors::Vector{Float64}  # l(l+1) values
    
    # Transform manager
    transform_manager::SHTnsTransformManager{T}
    
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
    
    # Work arrays
    work_tor = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    work_pol = create_shtns_spectral_field(T, config, domain_oc, pencil_spec)
    work_physical = create_shtns_vector_field(T, config, domain_oc, pencils)
    induction_physical = create_shtns_vector_field(T, config, domain_oc, pencils)
    
    # Pre-compute l(l+1) factors
    l_factors = Float64[l * (l + 1) for l in config.l_values]
    
    # Create transform manager
    transform_manager = get_transform_manager(T, config, pencil_spec)
    
    imposed_field = nothing
    
    return SHTnsMagneticFields{T}(magnetic, current, 
                                toroidal, poloidal,
                                ic_toroidal, ic_poloidal, 
                                nl_toroidal, nl_poloidal,
                                work_tor, work_pol, work_physical,
                                induction_physical,
                                l_factors, transform_manager,
                                imposed_field)
end


# ========================================================
# Main nonlinear computation using optimized transforms
# ========================================================
function compute_magnetic_nonlinear!(mag_fields::SHTnsMagneticFields{T}, 
                                    vel_fields, rotation_rate) where T
    # Zero work arrays
    zero_magnetic_work_arrays!(mag_fields)
    
    # Step 1: Convert spectral B to physical space using optimized transforms
    shtns_vector_synthesis!(mag_fields.toroidal, mag_fields.poloidal, 
                            mag_fields.magnetic)
    
    # Step 2: Compute current density j = ∇ × B in spectral space
    compute_current_density_spectral!(mag_fields)
    
    # Step 3: Transform current to physical space
    shtns_vector_synthesis!(mag_fields.work_tor, mag_fields.work_pol, 
                            mag_fields.current)
    
    # Step 4: Compute induction equation: ∂B/∂t = ∇ × (u × B) + η∇²B
    if vel_fields !== nothing
        compute_induction_term!(mag_fields, vel_fields)
    end
    
    # Step 5: Inner core rotation effects
    if rotation_rate != 0.0
        add_inner_core_rotation!(mag_fields, rotation_rate)
    end
    
    # Note: The nonlinear terms are now in mag_fields.nl_toroidal/poloidal
end




# ==============================
# Induction term computation
# ==============================
function compute_induction_term!(mag_fields::SHTnsMagneticFields{T}, vel_fields) where T
    # Compute ∇ × (u × B) for the induction equation
    
    # Step 1: Compute u × B in physical space
    compute_velocity_cross_magnetic!(mag_fields, vel_fields)
    
    # Step 2: Transform u × B to spectral space
    shtns_vector_analysis!(mag_fields.induction_physical, 
                          mag_fields.work_tor, mag_fields.work_pol)
    
    # Step 3: Compute curl of (u × B) in spectral space
    compute_curl_of_induction!(mag_fields)
end

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



function compute_ohmic_dissipation(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute Ohmic dissipation: η |∇ × B|²
    
    # Current density already computed in work arrays
    j_tor_real = parent(mag_fields.work_tor.data_real)
    j_tor_imag = parent(mag_fields.work_tor.data_imag)
    j_pol_real = parent(mag_fields.work_pol.data_real)
    j_pol_imag = parent(mag_fields.work_pol.data_imag)
    
    local_dissipation = zero(Float64)
    
    lm_range = get_local_range(mag_fields.work_tor.pencil, 1)
    r_range  = get_local_range(mag_fields.work_tor.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.work_tor.nlm
            local_lm = lm_idx - first(lm_range) + 1
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(j_tor_real, 3)
                    local_dissipation += (
                        j_tor_real[local_lm, 1, local_r]^2 + 
                        j_tor_imag[local_lm, 1, local_r]^2 + 
                        j_pol_real[local_lm, 1, local_r]^2 + 
                        j_pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Scale by magnetic diffusivity and global sum
    return (1.0 / d_Pm) * MPI.Allreduce(local_dissipation, MPI.SUM, get_comm())
end


# =======================
# Utility functions
# =======================
function zero_magnetic_work_arrays!(mag_fields::SHTnsMagneticFields{T}) where T
    # Zero all work arrays
    fill!(parent(mag_fields.work_tor.data_real), zero(T))
    fill!(parent(mag_fields.work_tor.data_imag), zero(T))
    fill!(parent(mag_fields.work_pol.data_real), zero(T))
    fill!(parent(mag_fields.work_pol.data_imag), zero(T))
    
    fill!(parent(mag_fields.work_physical.r_component.data), zero(T))
    fill!(parent(mag_fields.work_physical.θ_component.data), zero(T))
    fill!(parent(mag_fields.work_physical.φ_component.data), zero(T))
    
    fill!(parent(mag_fields.induction_physical.r_component.data), zero(T))
    fill!(parent(mag_fields.induction_physical.θ_component.data), zero(T))
    fill!(parent(mag_fields.induction_physical.φ_component.data), zero(T))
end



