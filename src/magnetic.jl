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



function compute_curl_of_induction!(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute ∇ × (u × B) in spectral space
    # This becomes the nonlinear term for the induction equation
    
    # Get local data views
    uxB_tor_real = parent(mag_fields.work_tor.data_real)
    uxB_tor_imag = parent(mag_fields.work_tor.data_imag)
    uxB_pol_real = parent(mag_fields.work_pol.data_real)
    uxB_pol_imag = parent(mag_fields.work_pol.data_imag)
    
    nl_tor_real = parent(mag_fields.nl_toroidal.data_real)
    nl_tor_imag = parent(mag_fields.nl_toroidal.data_imag)
    nl_pol_real = parent(mag_fields.nl_poloidal.data_real)
    nl_pol_imag = parent(mag_fields.nl_poloidal.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(mag_fields.toroidal.pencil, 1)
    r_range  = get_local_range(mag_fields.toroidal.pencil, 3)
    
    # Apply curl in spectral space
    @inbounds for lm_idx in lm_range
        if lm_idx <= length(mag_fields.l_factors)
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(uxB_tor_real, 3)
                    # Curl of (u × B)
                    nl_tor_real[local_lm, 1, local_r] = l_factor * uxB_pol_real[local_lm, 1, local_r]
                    nl_tor_imag[local_lm, 1, local_r] = l_factor * uxB_pol_imag[local_lm, 1, local_r]
                    nl_pol_real[local_lm, 1, local_r] = -l_factor * uxB_tor_real[local_lm, 1, local_r]
                    nl_pol_imag[local_lm, 1, local_r] = -l_factor * uxB_tor_imag[local_lm, 1, local_r]
                end
            end
        end
    end
end


# ========================================================
# Inner core rotation effects
# ========================================================
function add_inner_core_rotation!(mag_fields::SHTnsMagneticFields{T}, Ω::Float64) where T
    # Inner core rotation: affects boundary coupling
    # This modifies the nonlinear terms based on inner core rotation
    
    # Get local data views
    ic_tor_real = parent(mag_fields.ic_toroidal.data_real)
    ic_tor_imag = parent(mag_fields.ic_toroidal.data_imag)
    ic_pol_real = parent(mag_fields.ic_poloidal.data_real)
    ic_pol_imag = parent(mag_fields.ic_poloidal.data_imag)
    
    nl_tor_real = parent(mag_fields.nl_toroidal.data_real)
    nl_tor_imag = parent(mag_fields.nl_toroidal.data_imag)
    nl_pol_real = parent(mag_fields.nl_poloidal.data_real)
    nl_pol_imag = parent(mag_fields.nl_poloidal.data_imag)
    
    # Get local ranges
    lm_range = get_local_range(mag_fields.ic_toroidal.pencil, 1)
    r_range  = get_local_range(mag_fields.ic_toroidal.pencil, 3)
    
    # Rotation factor for inner core coupling
    rotation_factor = Ω * 1e-3  # Scaled rotation rate
    
    # Add rotation effects to nonlinear terms at inner core boundary
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.ic_toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            m = mag_fields.toroidal.config.m_values[lm_idx]
            
            # Only affects m ≠ 0 modes (azimuthal dependence)
            if m != 0
                # Apply at inner core boundary (first radial point)
                if 1 in r_range
                    local_r = 1 - first(r_range) + 1
                    if local_r <= size(nl_tor_real, 3)
                        # Add rotation-induced coupling
                        coupling_factor = rotation_factor * Float64(m)
                        
                        # Cross-coupling between toroidal and poloidal due to rotation
                        nl_tor_real[local_lm, 1, local_r] += coupling_factor * ic_pol_imag[local_lm, 1, local_r]
                        nl_tor_imag[local_lm, 1, local_r] -= coupling_factor * ic_pol_real[local_lm, 1, local_r]
                        nl_pol_real[local_lm, 1, local_r] -= coupling_factor * ic_tor_imag[local_lm, 1, local_r]
                        nl_pol_imag[local_lm, 1, local_r] += coupling_factor * ic_tor_real[local_lm, 1, local_r]
                    end
                end
            end
        end
    end
end


# =======================
# Diagnostic functions
# =======================
function compute_magnetic_energy(mag_fields::SHTnsMagneticFields{T}) where T
    # Compute magnetic energy in spectral space
    
    tor_real = parent(mag_fields.toroidal.data_real)
    tor_imag = parent(mag_fields.toroidal.data_imag)
    pol_real = parent(mag_fields.poloidal.data_real)
    pol_imag = parent(mag_fields.poloidal.data_imag)
    
    local_energy = zero(Float64)
    
    # Get local ranges
    lm_range = get_local_range(mag_fields.toroidal.pencil, 1)
    r_range  = get_local_range(mag_fields.toroidal.pencil, 3)
    
    @inbounds for lm_idx in lm_range
        if lm_idx <= mag_fields.toroidal.nlm
            local_lm = lm_idx - first(lm_range) + 1
            l_factor = mag_fields.l_factors[lm_idx]
            
            # Weight by l(l+1) for proper spectral integration
            weight = 1.0 / max(l_factor, 1.0)
            
            @simd for r_idx in r_range
                local_r = r_idx - first(r_range) + 1
                if local_r <= size(tor_real, 3)
                    local_energy += weight * (
                        tor_real[local_lm, 1, local_r]^2 + 
                        tor_imag[local_lm, 1, local_r]^2 + 
                        pol_real[local_lm, 1, local_r]^2 + 
                        pol_imag[local_lm, 1, local_r]^2
                    )
                end
            end
        end
    end
    
    # Global sum across all processes
    return 0.5 * MPI.Allreduce(local_energy, MPI.SUM, get_comm())
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



