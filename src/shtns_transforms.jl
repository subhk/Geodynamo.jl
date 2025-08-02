# ============================================================================
# SHTns Spherical Harmonic Transforms
# ============================================================================

# module SHTnsTransforms
# using SHTnsSpheres
# using PencilArrays
# using LinearAlgebra
# using ..Parameters
# using ..SHTnsSetup
# using ..VariableTypes
# using ..PencilSetup
    
# Transform from spectral to physical space using SHTns
function shtns_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                    phys::SHTnsPhysicalField{T}) where T
    sht = spec.config.sht
    nlm = spec.config.nlm
    
    # Process each radial level
    @views for r_idx in spec.local_radial_range
        if r_idx <= size(spec.data_real, 3) && r_idx <= size(phys.data_r, 3)
            # Prepare spectral coefficients
            # SHTns expects complex coefficients
            coeffs = zeros(ComplexF64, nlm)
            
            for lm_idx in 1:nlm
                real_part = spec.data_real[lm_idx, 1, r_idx]
                imag_part = spec.data_imag[lm_idx, 1, r_idx]
                coeffs[lm_idx] = complex(real_part, imag_part)
            end
            
            # Perform spherical harmonic synthesis using SHTns
            physical_data = synthesis(sht, coeffs)
            
            # Store result in physical field
            # Note: SHTns returns data in (theta, phi) order
            for j_phi in 1:phys.nlon, i_theta in 1:phys.nlat
                if i_theta <= size(phys.data_r, 1) && j_phi <= size(phys.data_r, 2)
                    phys.data_r[i_theta, j_phi, r_idx] = real(physical_data[i_theta, j_phi])
                end
            end
        end
    end
    
    # Copy to other pencil orientations if needed
    # This would involve transpose operations using PencilArrays
end

# Transform from physical to spectral space using SHTns
function shtns_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T}) where T
    sht = spec.config.sht
    nlm = spec.config.nlm
    
    # Process each radial level
    @views for r_idx in spec.local_radial_range
        if r_idx <= size(phys.data_r, 3) && r_idx <= size(spec.data_real, 3)
            # Prepare physical data for analysis
            physical_data = zeros(ComplexF64, phys.nlat, phys.nlon)
            
            for j_phi in 1:phys.nlon, i_theta in 1:phys.nlat
                if i_theta <= size(phys.data_r, 1) && j_phi <= size(phys.data_r, 2)
                    physical_data[i_theta, j_phi] = complex(phys.data_r[i_theta, j_phi, r_idx])
                end
            end
            
            # Perform spherical harmonic analysis using SHTns
            coeffs = analysis(sht, physical_data)
            
            # Store spectral coefficients
            for lm_idx in 1:nlm
                if lm_idx <= length(coeffs)
                    spec.data_real[lm_idx, 1, r_idx] = real(coeffs[lm_idx])
                    spec.data_imag[lm_idx, 1, r_idx] = imag(coeffs[lm_idx])
                    
                    # Ensure m=0 modes are real
                    m = spec.config.m_values[lm_idx]
                    if m == 0
                        spec.data_imag[lm_idx, 1, r_idx] = zero(T)
                    end
                end
            end
        end
    end
end

# Compute derivatives using SHTns
function shtns_compute_theta_derivative!(input::SHTnsSpectralField{T}, 
                                        output::SHTnsSpectralField{T}) where T
    sht = input.config.sht
    nlm = input.config.nlm
    
    @views for r_idx in input.local_radial_range
        if r_idx <= size(input.data_real, 3) && r_idx <= size(output.data_real, 3)
            # Prepare input coefficients
            coeffs = zeros(ComplexF64, nlm)
            for lm_idx in 1:nlm
                real_part = input.data_real[lm_idx, 1, r_idx]
                imag_part = input.data_imag[lm_idx, 1, r_idx]
                coeffs[lm_idx] = complex(real_part, imag_part)
            end
            
            # Compute θ derivative using SHTns
            d_theta_coeffs = synthesis_dtheta(sht, coeffs)
            
            # Convert back to analysis space if needed
            deriv_coeffs = analysis(sht, d_theta_coeffs)
            
            # Store result
            for lm_idx in 1:nlm
                if lm_idx <= length(deriv_coeffs)
                    output.data_real[lm_idx, 1, r_idx] = real(deriv_coeffs[lm_idx])
                    output.data_imag[lm_idx, 1, r_idx] = imag(deriv_coeffs[lm_idx])
                end
            end
        end
    end
end

function shtns_compute_phi_derivative!(input::SHTnsSpectralField{T}, 
                                        output::SHTnsSpectralField{T}) where T
    sht = input.config.sht
    nlm = input.config.nlm
    
    @views for r_idx in input.local_radial_range
        if r_idx <= size(input.data_real, 3) && r_idx <= size(output.data_real, 3)
            # Prepare input coefficients
            coeffs = zeros(ComplexF64, nlm)
            for lm_idx in 1:nlm
                real_part = input.data_real[lm_idx, 1, r_idx]
                imag_part = input.data_imag[lm_idx, 1, r_idx]
                coeffs[lm_idx] = complex(real_part, imag_part)
            end
            
            # Compute φ derivative using SHTns
            d_phi_coeffs = synthesis_dphi(sht, coeffs)
            
            # Convert back to analysis space if needed
            deriv_coeffs = analysis(sht, d_phi_coeffs)
            
            # Store result
            for lm_idx in 1:nlm
                if lm_idx <= length(deriv_coeffs)
                    output.data_real[lm_idx, 1, r_idx] = real(deriv_coeffs[lm_idx])
                    output.data_imag[lm_idx, 1, r_idx] = imag(deriv_coeffs[lm_idx])
                end
            end
        end
    end
end

# Compute vector spherical harmonic transforms
function shtns_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                pol_spec::SHTnsSpectralField{T},
                                vec_phys::SHTnsVectorField{T}) where T
    sht = tor_spec.config.sht
    nlm = tor_spec.config.nlm
    
    @views for r_idx in tor_spec.local_radial_range
        # Prepare toroidal and poloidal coefficients
        tor_coeffs = zeros(ComplexF64, nlm)
        pol_coeffs = zeros(ComplexF64, nlm)
        
        for lm_idx in 1:nlm
            tor_coeffs[lm_idx] = complex(tor_spec.data_real[lm_idx, 1, r_idx], 
                                        tor_spec.data_imag[lm_idx, 1, r_idx])
            pol_coeffs[lm_idx] = complex(pol_spec.data_real[lm_idx, 1, r_idx], 
                                        pol_spec.data_imag[lm_idx, 1, r_idx])
        end
        
        # Use SHTns vector synthesis
        # This computes (v_θ, v_φ) from toroidal and poloidal scalars
        v_theta, v_phi = vector_synthesis(sht, tor_coeffs, pol_coeffs)
        
        # Store in vector field (θ and φ components)
        for j_phi in 1:vec_phys.θ_component.nlon, i_theta in 1:vec_phys.θ_component.nlat
            vec_phys.θ_component.data_r[i_theta, j_phi, r_idx] = real(v_theta[i_theta, j_phi])
            vec_phys.φ_component.data_r[i_theta, j_phi, r_idx] = real(v_phi[i_theta, j_phi])
        end
        
        # Radial component needs to be computed separately from divergence
        # For incompressible flow: ∇·v = 0, so v_r can be derived from continuity
    end
end

function shtns_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                tor_spec::SHTnsSpectralField{T}, 
                                pol_spec::SHTnsSpectralField{T}) where T
    sht = tor_spec.config.sht
    nlm = tor_spec.config.nlm
    
    @views for r_idx in tor_spec.local_radial_range
        # Prepare physical vector components
        v_theta = zeros(ComplexF64, vec_phys.θ_component.nlat, vec_phys.θ_component.nlon)
        v_phi = zeros(ComplexF64, vec_phys.φ_component.nlat, vec_phys.φ_component.nlon)
        
        for j_phi in 1:size(v_theta, 2), i_theta in 1:size(v_theta, 1)
            v_theta[i_theta, j_phi] = complex(vec_phys.θ_component.data_r[i_theta, j_phi, r_idx])
            v_phi[i_theta, j_phi] = complex(vec_phys.φ_component.data_r[i_theta, j_phi, r_idx])
        end
        
        # Use SHTns vector analysis
        tor_coeffs, pol_coeffs = vector_analysis(sht, v_theta, v_phi)
        
        # Store spectral coefficients
        for lm_idx in 1:nlm
            if lm_idx <= length(tor_coeffs)
                tor_spec.data_real[lm_idx, 1, r_idx] = real(tor_coeffs[lm_idx])
                tor_spec.data_imag[lm_idx, 1, r_idx] = imag(tor_coeffs[lm_idx])
                pol_spec.data_real[lm_idx, 1, r_idx] = real(pol_coeffs[lm_idx])
                pol_spec.data_imag[lm_idx, 1, r_idx] = imag(pol_coeffs[lm_idx])
                
                # Ensure m=0 modes are real
                m = tor_spec.config.m_values[lm_idx]
                if m == 0
                    tor_spec.data_imag[lm_idx, 1, r_idx] = zero(T)
                    pol_spec.data_imag[lm_idx, 1, r_idx] = zero(T)
                end
            end
        end
    end
end
    
# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_compute_theta_derivative!, shtns_compute_phi_derivative!
# export shtns_vector_synthesis!, shtns_vector_analysis!

# end
