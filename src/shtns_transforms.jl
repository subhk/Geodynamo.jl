# ============================================================================
# SHTns Spherical Harmonic Transforms
# ============================================================================

struct SHTnsTransformManager{T}
    # Pre-allocated coefficient arrays
    coeffs_full::Vector{ComplexF64}
    coeffs_work::Vector{ComplexF64}
    
    # Pre-allocated physical arrays
    phys_work::Matrix{ComplexF64}
    phys_real::Matrix{T}
    
    # Communication buffers
    send_buffer::Vector{ComplexF64}
    recv_buffer::Vector{ComplexF64}
    
    # MPI requests for async operations
    requests::Vector{MPI.Request}
    
    # Configuration
    nlm::Int
    nlat::Int
    nlon::Int
    needs_allreduce::Bool
end


function create_transform_manager(::Type{T}, 
                            config::SHTnsConfig, 
                            pencil::Pencil{3}) where T

    nlm  = config.nlm
    nlat = config.nlat
    nlon = config.nlon
    
    # Check if we need communication
    lm_range = get_local_range(pencil, 1)
    needs_allreduce = length(lm_range) < nlm
    
    return SHTnsTransformManager{T}(
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlat, nlon),
        zeros(T, nlat, nlon),
        zeros(ComplexF64, nlm),
        zeros(ComplexF64, nlm),
        MPI.Request[],
        nlm, nlat, nlon, needs_allreduce
    )
end
    

# Global transform managers (one per thread for thread safety)
const TRANSFORM_MANAGERS = Dict{UInt64, SHTnsTransformManager}()

function get_transform_manager(::Type{T}, config::SHTnsConfig, pencil::Pencil{3}) where T
    thread_id = Threads.threadid()
    key = hash((thread_id, T, config.nlm, config.nlat, config.nlon))
    
    if !haskey(TRANSFORM_MANAGERS, key)
        TRANSFORM_MANAGERS[key] = create_transform_manager(T, config, pencil)
    end
    
    return TRANSFORM_MANAGERS[key]
end



# Transform from spectral to physical space using SHTns
function shtns_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                    phys::SHTnsPhysicalField{T}) where T

    sht = spec.config.sht
    nlm = spec.config.nlm
    
    # Get local data
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    # Get local ranges once
    r_range = get_local_range(spec.pencil, 3)
    lm_range = get_local_range(spec.pencil, 1)
    
    # Pre-allocate coefficient array
    coeffs = zeros(ComplexF64, nlm)
    
    # Process each local radial level
        @inbounds for r_idx in r_range
            local_r = r_idx - first(r_range) + 1
            
            if local_r <= size(spec_real, 3)
                # Fill coefficients for this radial level
                fill!(coeffs, zero(ComplexF64))
                
                @simd for lm_idx in lm_range
                    if lm_idx <= nlm
                        local_lm = lm_idx - first(lm_range) + 1
                        coeffs[lm_idx] = complex(spec_real[local_lm, 1, local_r],
                                                spec_imag[local_lm, 1, local_r])
                    end
                end
                
                # Single collective communication
                if length(lm_range) < nlm
                    coeffs = MPI.Allreduce(coeffs, MPI.SUM, get_comm())
                end
                
                # Perform synthesis
                physical_data = synthesis(sht, coeffs)
                
                # Store with optimized memory access
                @inbounds @simd for idx in eachindex(physical_data)
                    phys_data[idx, local_r] = real(physical_data[idx])
                end
            end
        end
    
    # # Transpose if needed
    # if transpose_plan !== nothing
    #     transpose!(phys.data, transpose_plan)
    # end
end

# Transform from physical to spectral space using SHTns
function shtns_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T}) where T

    sht = spec.config.sht
    nlm = spec.config.nlm
    
    # Transpose if needed
    if transpose_plan !== nothing
        transpose!(phys.data, transpose_plan)
    end
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    
    # Get local ranges
    r_range = get_local_range(phys.pencil, 3)
    lm_range = get_local_range(spec.pencil, 1)
    
    @inbounds for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(phys_data, 3)
            # Extract and analyze
            physical_data = complex.(view(phys_data, :, :, local_r))
            coeffs = analysis(sht, physical_data)
            
            # Store spectral coefficients
            @simd for lm_idx in lm_range
                if lm_idx <= nlm
                    local_lm = lm_idx - first(lm_range) + 1
                    if local_lm <= size(spec_real, 1)
                        spec_real[local_lm, 1, local_r] = real(coeffs[lm_idx])
                        spec_imag[local_lm, 1, local_r] = imag(coeffs[lm_idx])
                        
                        # Ensure m=0 modes are real
                        m = spec.config.m_values[lm_idx]
                        if m == 0
                            spec_imag[local_lm, 1, local_r] = zero(T)
                        end
                    end
                end
            end
        end
    end
end

# # Compute derivatives using SHTns
# function shtns_compute_theta_derivative!(input::SHTnsSpectralField{T}, 
#                                         output::SHTnsSpectralField{T}) where T
#     sht = input.config.sht
#     nlm = input.config.nlm
    
#     @views for r_idx in input.local_radial_range
#         if r_idx <= size(input.data_real, 3) && r_idx <= size(output.data_real, 3)
#             # Prepare input coefficients
#             coeffs = zeros(ComplexF64, nlm)
#             for lm_idx in 1:nlm
#                 real_part = input.data_real[lm_idx, 1, r_idx]
#                 imag_part = input.data_imag[lm_idx, 1, r_idx]
#                 coeffs[lm_idx] = complex(real_part, imag_part)
#             end
            
#             # Compute θ derivative using SHTns
#             d_theta_coeffs = synthesis_dtheta(sht, coeffs)
            
#             # Convert back to analysis space if needed
#             deriv_coeffs = analysis(sht, d_theta_coeffs)
            
#             # Store result
#             for lm_idx in 1:nlm
#                 if lm_idx <= length(deriv_coeffs)
#                     output.data_real[lm_idx, 1, r_idx] = real(deriv_coeffs[lm_idx])
#                     output.data_imag[lm_idx, 1, r_idx] = imag(deriv_coeffs[lm_idx])
#                 end
#             end
#         end
#     end
# end

# function shtns_compute_phi_derivative!(input::SHTnsSpectralField{T}, 
#                                         output::SHTnsSpectralField{T}) where T
#     sht = input.config.sht
#     nlm = input.config.nlm
    
#     @views for r_idx in input.local_radial_range
#         if r_idx <= size(input.data_real, 3) && r_idx <= size(output.data_real, 3)
#             # Prepare input coefficients
#             coeffs = zeros(ComplexF64, nlm)
#             for lm_idx in 1:nlm
#                 real_part = input.data_real[lm_idx, 1, r_idx]
#                 imag_part = input.data_imag[lm_idx, 1, r_idx]
#                 coeffs[lm_idx] = complex(real_part, imag_part)
#             end
            
#             # Compute φ derivative using SHTns
#             d_phi_coeffs = synthesis_dphi(sht, coeffs)
            
#             # Convert back to analysis space if needed
#             deriv_coeffs = analysis(sht, d_phi_coeffs)
            
#             # Store result
#             for lm_idx in 1:nlm
#                 if lm_idx <= length(deriv_coeffs)
#                     output.data_real[lm_idx, 1, r_idx] = real(deriv_coeffs[lm_idx])
#                     output.data_imag[lm_idx, 1, r_idx] = imag(deriv_coeffs[lm_idx])
#                 end
#             end
#         end
#     end
# end


# Vector synthesis for PencilArrays
function shtns_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                pol_spec::SHTnsSpectralField{T},
                                vec_phys::SHTnsVectorField{T}) where T
    sht = tor_spec.config.sht
    nlm = tor_spec.config.nlm
    
    # Process each component
    for r_idx in get_local_range(tor_spec.pencil, 3)
        # Gather toroidal and poloidal coefficients
        tor_coeffs = zeros(ComplexF64, nlm)
        pol_coeffs = zeros(ComplexF64, nlm)
        
        # Fill local portions
        lm_range = get_local_range(tor_spec.pencil, 1)
        for lm_idx in lm_range
            if lm_idx <= nlm
                local_lm = lm_idx - first(lm_range) + 1
                local_r = r_idx - first(get_local_range(tor_spec.pencil, 3)) + 1
                
                tor_coeffs[lm_idx] = complex(parent(tor_spec.data_real)[local_lm, 1, local_r],
                                            parent(tor_spec.data_imag)[local_lm, 1, local_r])
                pol_coeffs[lm_idx] = complex(parent(pol_spec.data_real)[local_lm, 1, local_r],
                                            parent(pol_spec.data_imag)[local_lm, 1, local_r])
            end
        end
        
        # Gather from all processes
        tor_coeffs = MPI.Allreduce(tor_coeffs, MPI.SUM, get_comm())
        pol_coeffs = MPI.Allreduce(pol_coeffs, MPI.SUM, get_comm())
        
        # Vector synthesis
        v_theta, v_phi = vector_synthesis(sht, tor_coeffs, pol_coeffs)
        
        # Store in local portions of vector field
        local_r = r_idx - first(get_local_range(vec_phys.θ_component.pencil, 3)) + 1
        if local_r <= size(parent(vec_phys.θ_component.data), 3)
            parent(vec_phys.θ_component.data)[:, :, local_r] = real(v_theta)
            parent(vec_phys.φ_component.data)[:, :, local_r] = real(v_phi)
        end
    end
end


# Vector analysis for PencilArrays
function shtns_vector_analysis!(vec_phys::SHTnsVectorField{T},
                               tor_spec::SHTnsSpectralField{T}, 
                               pol_spec::SHTnsSpectralField{T}) where T

    sht = tor_spec.config.sht
    nlm = tor_spec.config.nlm
    
    for r_idx in get_local_range(vec_phys.θ_component.pencil, 3)
        local_r = r_idx - first(get_local_range(vec_phys.θ_component.pencil, 3)) + 1
        
        # Extract local physical data
        v_theta = complex.(parent(vec_phys.θ_component.data)[:, :, local_r])
        v_phi   = complex.(parent(vec_phys.φ_component.data)[:, :, local_r])
        
        # Vector analysis
        tor_coeffs, pol_coeffs = vector_analysis(sht, v_theta, v_phi)
        
        # Store in local spectral coefficients
        lm_range = get_local_range(tor_spec.pencil, 1)
        for lm_idx in lm_range
            if lm_idx <= nlm
                local_lm = lm_idx - first(lm_range) + 1
                
                parent(tor_spec.data_real)[local_lm, 1, local_r] = real(tor_coeffs[lm_idx])
                parent(tor_spec.data_imag)[local_lm, 1, local_r] = imag(tor_coeffs[lm_idx])
                parent(pol_spec.data_real)[local_lm, 1, local_r] = real(pol_coeffs[lm_idx])
                parent(pol_spec.data_imag)[local_lm, 1, local_r] = imag(pol_coeffs[lm_idx])
                
                # Ensure m=0 modes are real
                m = tor_spec.config.m_values[lm_idx]
                if m == 0
                    parent(tor_spec.data_imag)[local_lm, 1, local_r] = zero(T)
                    parent(pol_spec.data_imag)[local_lm, 1, local_r] = zero(T)
                end
            end
        end
    end
end


# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_vector_synthesis!, shtns_vector_analysis!


# export shtns_spectral_to_physical!, shtns_physical_to_spectral!
# export shtns_compute_theta_derivative!, shtns_compute_phi_derivative!
# export shtns_vector_synthesis!, shtns_vector_analysis!

# end
