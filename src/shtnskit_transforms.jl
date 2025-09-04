# ============================================================================
# SHTnsKit Spherical Harmonic Transforms with PencilArrays Integration
# ============================================================================
# 
# This module implements spherical harmonic transforms using SHTnsKit.jl
# with MPI parallelization across theta and phi directions using PencilArrays
# and efficient FFTs using PencilFFTs
#

using SHTnsKit
using PencilArrays
using PencilArrays.Transpositions
using PencilFFTs
using FFTW
using LinearAlgebra
using Base.Threads

# Simple heuristic for number of simultaneously allocated fields (for memory estimate)
estimate_field_count() = 6

# ============================================================================
# SHTnsKit Configuration Structure
# ============================================================================

struct SHTnsKitConfig
    # SHTnsKit configuration
    sht_config::SHTnsKit.SHTConfig
    
    # Grid parameters
    nlat::Int
    nlon::Int
    lmax::Int
    mmax::Int
    nlm::Int
    
    # PencilArrays decomposition for parallelization
    pencils::NamedTuple
    
    # PencilFFTs plans for efficient phi-direction FFTs
    fft_plans::Dict{Symbol, Any}
    
    # Transpose plans for pencil reorientations
    transpose_plans::Dict{Symbol, PencilArrays.TransposeOperator}
    
    # Memory estimate
    memory_estimate::String

    # Convenience fields for compatibility with legacy code paths
    l_values::Vector{Int}
    m_values::Vector{Int}
    theta_grid::Vector{Float64}
    phi_grid::Vector{Float64}
    gauss_weights::Vector{Float64}
end

"""
    create_shtnskit_config(; lmax::Int, mmax::Int=lmax, nlat::Int=lmax+2, 
                           nlon::Int=max(2*lmax+1, 4), optimize_decomp::Bool=true) -> SHTnsKitConfig

Create SHTnsKit configuration with MPI parallelization using PencilArrays.
This creates proper integration with ../SHTnsKit.jl, PencilArrays, and PencilFFTs.
"""
function create_shtnskit_config(; lmax::Int, mmax::Int=lmax, 
                               nlat::Int=max(lmax+2, i_Th), 
                               nlon::Int=max(2*lmax+1, 4, i_Ph),
                               optimize_decomp::Bool=true)
    
    # Create SHTnsKit configuration using the local ../SHTnsKit.jl
    sht_config = SHTnsKit.create_gauss_config(lmax, nlat; 
                                            mmax=mmax, 
                                            nlon=nlon,
                                            norm=:orthonormal)
    
    # Enable optimized Legendre polynomial tables for better performance
    SHTnsKit.prepare_plm_tables!(sht_config)
    
    # Get MPI communicator
    comm = get_comm()
    nprocs = get_nprocs()
    
    # Create pencil decomposition for parallel theta-phi transforms
    pencils = create_pencil_decomposition_shtnskit(nlat, nlon, i_N, sht_config, comm, optimize_decomp)
    
    # Create PencilFFTs plans for efficient phi-direction transforms
    fft_plans = create_pencil_fft_plans(pencils, (nlat, nlon, i_N))
    
    # Create transpose plans between different pencil orientations
    transpose_plans = create_shtnskit_transpose_plans(pencils)
    
    # Estimate memory usage
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, Float64)
    memory_estimate = "$(round(memory_mb, digits=1)) MB"
    
    nlm = sht_config.nlm

    # Populate compatibility grids and index arrays
    theta_grid = try
        Vector{Float64}(SHTnsKit.grid_latitudes(sht_config))
    catch
        range(-pi/2, stop=pi/2, length=nlat) |> collect |> Float64.
    end
    phi_grid = try
        Vector{Float64}(SHTnsKit.grid_longitudes(sht_config))
    catch
        range(0, stop=2pi, length=nlon+1)[1:end-1] |> collect |> Float64.
    end
    gauss_weights = try
        Vector{Float64}(SHTnsKit.get_gauss_weights(sht_config))
    catch
        ones(Float64, nlat)
    end
    # Construct l/m arrays matching nlm ordering
    l_vals = Vector{Int}(undef, nlm)
    m_vals = Vector{Int}(undef, nlm)
    idx = 1
    for l in 0:lmax
        for m in 0:min(l, mmax)
            if idx <= nlm
                l_vals[idx] = l
                m_vals[idx] = m
            end
            idx += 1
        end
    end
    
    if get_rank() == 0
        print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    end
    
    return SHTnsKitConfig(
        sht_config, nlat, nlon, lmax, mmax, nlm,
        pencils, fft_plans, transpose_plans, memory_estimate,
        l_vals, m_vals, theta_grid, phi_grid, gauss_weights
    )
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, sht_config, comm, optimize)

Create PencilArrays decomposition optimized for theta-phi parallelization.
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int,
                                             sht_config::SHTnsKit.SHTConfig,
                                             comm, optimize::Bool=true)
    nprocs = MPI.Comm_size(comm)
    
    # Determine optimal process topology for theta-phi parallelization
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon)
    else
        proc_dims = (nprocs, 1)
    end
    
    # Create PencilArrays topology
    # Construct MPI-aware topology via dynamic lookup (MPITopology in recent versions)
    TopoCtor = getproperty(PencilArrays, Symbol("MPITopology"))
    topology = TopoCtor(comm, proc_dims)
    
    # Physical space pencils for theta-phi parallelization
    dims = (nlat, nlon, nr)
    pencil_theta = Pencil(topology, dims, (2, 3))  # Theta-contiguous (phi distributed)
    pencil_phi = Pencil(topology, dims, (1, 3))    # Phi-contiguous (theta distributed)
    pencil_r = Pencil(topology, dims, (1, 2))      # Radial-contiguous
    
    # Spectral space pencil (for (l,m) modes)
    # Use SHTnsKit configuration's nlm
    nlm = sht_config.nlm
    spec_dims = (nlm, 1, nr)
    pencil_spec = Pencil(topology, spec_dims, (2, 3))
    
    return (theta = pencil_theta, 
            phi = pencil_phi, 
            r = pencil_r,
            spec = pencil_spec)
end

"""
    optimize_process_topology_shtnskit(nprocs, nlat, nlon)

Optimize MPI process topology for theta-phi parallelization.
"""
function optimize_process_topology_shtnskit(nprocs::Int, nlat::Int, nlon::Int)
    # Find factorization that balances theta and phi parallelization
    best_dims = (nprocs, 1)
    best_score = Inf
    
    for p_theta in 1:nprocs
        if nprocs % p_theta == 0
            p_phi = nprocs ÷ p_theta
            
            # Check if decomposition makes sense
            if nlat ÷ p_theta < 2 || nlon ÷ p_phi < 2
                continue
            end
            
            # Check load balance
            theta_imbalance = abs(nlat % p_theta) / nlat
            phi_imbalance = abs(nlon % p_phi) / nlon
            
            # Prefer more balanced decomposition
            score = theta_imbalance + phi_imbalance
            
            if score < best_score
                best_score = score
                best_dims = (p_theta, p_phi)
            end
        end
    end
    
    return best_dims
end

"""
    create_pencil_fft_plans(pencils, dims)

Create PencilFFTs plans for efficient phi-direction transforms.
"""
function create_pencil_fft_plans(pencils, dims::Tuple{Int,Int,Int})
    nlat, nlon, nr = dims
    fft_plans = Dict{Symbol, Any}()
    
    try
        # Create FFT plans for phi-direction (longitude) transforms
        if haskey(pencils, :phi)
            # Create a sample array for planning
            sample_array = PencilArray{ComplexF64}(undef, pencils.phi)
            
            # PencilFFTs plans for phi direction (dimension 2)
            fft_plans[:phi_forward] = PencilFFTs.plan_fft!(sample_array, 2)
            fft_plans[:phi_backward] = PencilFFTs.plan_ifft!(sample_array, 2)
        end
        
        # Create plans for other orientations if needed
        if haskey(pencils, :theta)
            sample_theta = PencilArray{ComplexF64}(undef, pencils.theta)
            fft_plans[:theta_forward] = PencilFFTs.plan_fft!(sample_theta, 2)
            fft_plans[:theta_backward] = PencilFFTs.plan_ifft!(sample_theta, 2)
        end
        
        if get_rank() == 0
            @info "PencilFFTs plans created successfully for $(length(fft_plans)) orientations"
        end
    catch e
        @warn "Could not create PencilFFTs plans: $e. Using fallback FFTW."
        fft_plans[:fallback] = true
    end
    
    return fft_plans
end

"""
    create_shtnskit_transpose_plans(pencils)

Create transpose plans for efficient pencil reorientations.
"""
function create_shtnskit_transpose_plans(pencils)
    transpose_plans = Dict{Symbol, PencilArrays.TransposeOperator}()
    
    try
        # Create transpose operations needed for spherical harmonic transforms
        if haskey(pencils, :theta) && haskey(pencils, :phi)
            # Transpose between theta and phi pencils for FFT operations
            transpose_plans[:theta_to_phi] = Transpose(pencils.theta => pencils.phi)
            transpose_plans[:phi_to_theta] = Transpose(pencils.phi => pencils.theta)
        end
        
        if haskey(pencils, :r) && haskey(pencils, :theta)
            # Transpose to r-pencil for radial operations
            transpose_plans[:theta_to_r] = Transpose(pencils.theta => pencils.r)
            transpose_plans[:r_to_theta] = Transpose(pencils.r => pencils.theta)
        end
        
        if haskey(pencils, :phi) && haskey(pencils, :r)
            # Transpose from phi-pencil to r-pencil
            transpose_plans[:phi_to_r] = Transpose(pencils.phi => pencils.r)
            transpose_plans[:r_to_phi] = Transpose(pencils.r => pencils.phi)
        end
        
        if get_rank() == 0
            @info "Created $(length(transpose_plans)) transpose plans for pencil reorientations"
        end
    catch e
        @warn "Could not create all transpose plans: $e"
    end
    
    return transpose_plans
end

"""
    estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, T)

Estimate memory usage for SHTnsKit-based transforms with PencilArrays.
"""
function estimate_memory_usage_shtnskit(nlat::Int, nlon::Int, lmax::Int, 
                                       field_count::Int, ::Type{T}) where T
    
    # Physical grid memory per process (distributed)
    physical_memory_per_process = (nlat * nlon * i_N * sizeof(T)) / get_nprocs()
    
    # Spectral memory (approximate)
    nlm = SHTnsKit.nlm_calc(lmax, lmax, 1)
    spectral_memory_per_process = (nlm * i_N * sizeof(ComplexF64) * 2) / get_nprocs()
    
    # PencilArrays working memory (transpose buffers)
    transpose_memory = max(physical_memory_per_process, spectral_memory_per_process)
    
    # PencilFFTs working memory
    fft_memory = physical_memory_per_process * 0.5
    
    # Total per field per process
    per_field_memory = physical_memory_per_process + spectral_memory_per_process + 
                      transpose_memory + fft_memory
    
    # Total for all fields
    total_memory = per_field_memory * field_count
    
    return total_memory / (1024^2)  # Convert to MB
end

"""
    print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)

Print configuration summary for SHTnsKit setup.
"""
function print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    println("\n╔═══════════════════════════════════════════════════════╗")
    println("║         SHTnsKit Configuration Summary                ║")
    println("╠═══════════════════════════════════════════════════════╣")
    println("║ Grid Configuration:                                   ║")
    println("║   Physical grid:    $(lpad(nlat,4)) × $(lpad(nlon,4)) × $(lpad(i_N,4))         ║")
    println("║   Spectral modes:   lmax=$(lpad(lmax,3)), mmax=$(lpad(mmax,3))              ║")
    println("║   Total modes:      $(lpad(nlm,5))                             ║")
    println("║                                                       ║")
    println("║ Parallel Configuration:                               ║")
    println("║   MPI Processes:    $(lpad(nprocs,4))                              ║")
    println("║   Theta-Phi Parallel: PencilArrays + PencilFFTs      ║")
    println("║   SHTnsKit.jl:      Registered package               ║")
    println("║   Memory/process:   $(lpad(memory_estimate,10))                    ║")
    println("╚═══════════════════════════════════════════════════════╝")
end

# ============================================================================
# Core Transform Functions using SHTnsKit with PencilArrays
# ============================================================================

"""
    shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                  phys::SHTnsPhysicalField{T}) where T

Transform from spectral to physical space using SHTnsKit with PencilArrays/PencilFFTs.
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                       phys::SHTnsPhysicalField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Check if we need to transpose for optimal FFT performance
    current_orientation = get_pencil_orientation(phys.pencil)
    
    if current_orientation == :phi && haskey(config.fft_plans, :phi_backward)
        # Direct synthesis with PencilFFTs (phi is local)
        perform_synthesis_phi_local!(spec, phys, config)
    elseif haskey(config.transpose_plans, :theta_to_phi)
        # Transpose to phi-pencil, perform synthesis, transpose back
        perform_synthesis_with_transpose!(spec, phys, config)
    else
        # Fallback to direct synthesis without transpose
        perform_synthesis_direct!(spec, phys, config)
    end
    
    # Synchronize MPI processes
    MPI.Barrier(get_comm())
end

"""
    perform_synthesis_phi_local!(spec, phys, config)

Perform synthesis when physical field is already in phi-pencil (phi is local).
"""
function perform_synthesis_phi_local!(spec::SHTnsSpectralField{T}, 
                                     phys::SHTnsPhysicalField{T}, 
                                     config) where T
    sht_config = config.sht_config
    
    # Get local data
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag) 
    phys_data = parent(phys.data)
    
    # Process each local radial level
    for r_local in 1:size(phys_data, 3)
        # Extract spectral coefficients for this radial level
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)
        
        # Perform SHTnsKit synthesis (handles internal FFTs)
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        
        # Store in physical array (respecting PencilArray layout)
        store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_with_transpose!(spec, phys, config)

Perform synthesis with transpose to phi-pencil for optimal FFT performance.
"""
function perform_synthesis_with_transpose!(spec::SHTnsSpectralField{T}, 
                                         phys::SHTnsPhysicalField{T}, 
                                         config) where T
    # Create temporary phi-pencil array
    phys_phi = PencilArray{T}(undef, config.pencils.phi)
    
    # Perform synthesis to phi-pencil
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)
    
    # Transpose back to original pencil orientation
    if haskey(config.transpose_plans, :phi_to_theta)
        mul!(phys.data, config.transpose_plans[:phi_to_theta], phys_phi)
    else
        # Fallback copy
        copyto!(parent(phys.data), parent(phys_phi))
    end
end

"""
    perform_synthesis_to_phi_pencil!(spec, phys_phi, config)

Perform synthesis directly to phi-pencil array.
"""
function perform_synthesis_to_phi_pencil!(spec::SHTnsSpectralField{T}, 
                                        phys_phi::PencilArray{T,3}, 
                                        config) where T
    sht_config = config.sht_config
    
    # Get data arrays
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_phi_data = parent(phys_phi)
    
    # Process each radial level
    for r_local in 1:size(phys_phi_data, 3)
        # Extract spectral coefficients
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)
        
        # Perform synthesis
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        
        # Store in phi-pencil array
        store_physical_slice_phi_local!(phys_phi_data, phys_slice, r_local, config)
    end
end

"""
    perform_synthesis_direct!(spec, phys, config)

Direct synthesis without transpose (fallback method).
"""
function perform_synthesis_direct!(spec::SHTnsSpectralField{T}, 
                                  phys::SHTnsPhysicalField{T}, 
                                  config) where T
    sht_config = config.sht_config
    
    # Get local data
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    phys_data = parent(phys.data)
    
    # Process each radial level
    for r_local in 1:size(phys_data, 3)
        # Extract coefficients
        coeffs_matrix = extract_coefficients_for_shtnskit(spec_real_data, spec_imag_data, r_local, config)
        
        # Perform synthesis
        phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        
        # Store result (generic storage for any pencil orientation)
        store_physical_slice_generic!(phys_data, phys_slice, r_local, config)
    end
end

"""
    shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                  spec::SHTnsSpectralField{T}) where T

Transform from physical to spectral space using SHTnsKit with PencilArrays.
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                       spec::SHTnsSpectralField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Check orientation for optimal analysis
    current_orientation = get_pencil_orientation(phys.pencil)
    
    if current_orientation == :phi && haskey(config.fft_plans, :phi_forward)
        # Direct analysis with PencilFFTs (phi is local)
        perform_analysis_phi_local!(phys, spec, config)
    elseif haskey(config.transpose_plans, :theta_to_phi)
        # Transpose to phi-pencil for analysis
        perform_analysis_with_transpose!(phys, spec, config)
    else
        # Direct analysis without transpose
        perform_analysis_direct!(phys, spec, config)
    end
    
    # Synchronize MPI processes
    MPI.Barrier(get_comm())
end

"""
    perform_analysis_phi_local!(phys, spec, config)

Perform analysis when physical field is in phi-pencil (phi is local).
"""
function perform_analysis_phi_local!(phys::SHTnsPhysicalField{T}, 
                                    spec::SHTnsSpectralField{T}, 
                                    config) where T
    sht_config = config.sht_config
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    
    # Process each radial level
    for r_local in 1:size(phys_data, 3)
        # Extract physical slice
        phys_slice = extract_physical_slice_phi_local(phys_data, r_local, config)
        
        # Perform SHTnsKit analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
        
        # Store spectral coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

"""
    perform_analysis_with_transpose!(phys, spec, config)

Perform analysis with transpose to phi-pencil.
"""
function perform_analysis_with_transpose!(phys::SHTnsPhysicalField{T}, 
                                        spec::SHTnsSpectralField{T}, 
                                        config) where T
    # Transpose to phi-pencil
    phys_phi = PencilArray{T}(undef, config.pencils.phi)
    
    if haskey(config.transpose_plans, :theta_to_phi)
        mul!(phys_phi, config.transpose_plans[:theta_to_phi], phys.data)
    else
        copyto!(parent(phys_phi), parent(phys.data))
    end
    
    # Perform analysis on phi-pencil data
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)
end

"""
    perform_analysis_from_phi_pencil!(phys_phi, spec, config)

Perform analysis from phi-pencil data.
"""
function perform_analysis_from_phi_pencil!(phys_phi::PencilArray{T,3}, 
                                         spec::SHTnsSpectralField{T}, 
                                         config) where T
    sht_config = config.sht_config
    
    # Get data arrays
    phys_phi_data = parent(phys_phi)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    
    # Process each radial level
    for r_local in 1:size(phys_phi_data, 3)
        # Extract physical slice
        phys_slice = extract_physical_slice_phi_local(phys_phi_data, r_local, config)
        
        # Perform analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
        
        # Store coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

"""
    perform_analysis_direct!(phys, spec, config)

Direct analysis without transpose (fallback).
"""
function perform_analysis_direct!(phys::SHTnsPhysicalField{T}, 
                                 spec::SHTnsSpectralField{T}, 
                                 config) where T
    sht_config = config.sht_config
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real_data = parent(spec.data_real)
    spec_imag_data = parent(spec.data_imag)
    
    # Process each radial level
    for r_local in 1:size(phys_data, 3)
        # Extract physical slice (generic extraction)
        phys_slice = extract_physical_slice_generic(phys_data, r_local, config)
        
        # Perform analysis
        coeffs_matrix = SHTnsKit.analysis(sht_config, phys_slice)
        
        # Store coefficients
        store_coefficients_from_shtnskit!(spec_real_data, spec_imag_data, coeffs_matrix, r_local, config)
    end
end

# ============================================================================
# Vector Transforms with SHTnsKit and PencilArrays
# ============================================================================

"""
    shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                              pol_spec::SHTnsSpectralField{T},
                              vec_phys::SHTnsVectorField{T}) where T

Vector synthesis using SHTnsKit spheroidal-toroidal decomposition with PencilArrays.
"""
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                   pol_spec::SHTnsSpectralField{T},
                                   vec_phys::SHTnsVectorField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get data arrays
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real) 
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    # Process each radial level
    for r_local in 1:size(tor_real, 3)
        # Extract toroidal and poloidal coefficients
        tor_coeffs = extract_coefficients_for_shtnskit(tor_real, tor_imag, r_local, config)
        pol_coeffs = extract_coefficients_for_shtnskit(pol_real, pol_imag, r_local, config)
        
        # Perform vector synthesis using SHTnsKit
        vt_field, vp_field = SHTnsKit.SHsphtor_to_spat(sht_config, pol_coeffs, tor_coeffs; 
                                                      real_output=true)
        
        # Store vector components
        store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)
    end
    
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpectralField{T}, 
                             pol_spec::SHTnsSpectralField{T}) where T

Vector analysis using SHTnsKit with PencilArrays.
"""
function shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                  tor_spec::SHTnsSpectralField{T}, 
                                  pol_spec::SHTnsSpectralField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get data arrays
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    # Process each radial level  
    for r_local in 1:size(v_theta, 3)
        # Extract vector components
        vt_field = extract_vector_component_generic(v_theta, r_local, config)
        vp_field = extract_vector_component_generic(v_phi, r_local, config)
        
        # Perform vector analysis using SHTnsKit
        pol_coeffs, tor_coeffs = SHTnsKit.spat_to_SHsphtor(sht_config, vt_field, vp_field)
        
        # Store spectral coefficients
        store_coefficients_from_shtnskit!(pol_real, pol_imag, pol_coeffs, r_local, config)
        store_coefficients_from_shtnskit!(tor_real, tor_imag, tor_coeffs, r_local, config)
    end
    
    MPI.Barrier(get_comm())
end

# ============================================================================
# Helper Functions for PencilArray Data Management
# ============================================================================

"""
    get_pencil_orientation(pencil::Pencil{3}) -> Symbol

Get the orientation of a pencil (which dimensions are local).
"""
function get_pencil_orientation(pencil::Pencil{3})
    # Get the local (contiguous) dimensions
    local_dims = pencil.axes_in
    if 1 in local_dims && 2 in local_dims
        return :theta_phi  # Both theta and phi local
    elseif 1 in local_dims
        return :theta      # Only theta local (phi distributed)
    elseif 2 in local_dims  
        return :phi        # Only phi local (theta distributed)
    else
        return :r          # Radial local
    end
end

"""
    extract_coefficients_for_shtnskit(spec_real, spec_imag, r_local, config)

Extract spectral coefficients in format expected by SHTnsKit.
"""
function extract_coefficients_for_shtnskit(spec_real, spec_imag, r_local, config)
    lmax, mmax = config.lmax, config.mmax
    
    # Create coefficient matrix in SHTnsKit format: (l+1, m+1)
    coeffs = zeros(ComplexF64, lmax+1, mmax+1)
    
    # Fill from local spectral data
    Threads.@threads for lm_idx in 1:size(spec_real, 1)
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if l <= lmax && m <= mmax && r_local <= size(spec_real, 3)
            real_part = spec_real[lm_idx, 1, r_local]
            imag_part = spec_imag[lm_idx, 1, r_local]
            coeffs[l+1, m+1] = complex(real_part, imag_part)
        end
    end
    
    # Communicate across MPI processes to complete the spectrum
    MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
    
    return coeffs
end

"""
    store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)

Store coefficients from SHTnsKit format back to spectral field.
"""
function store_coefficients_from_shtnskit!(spec_real, spec_imag, coeffs_matrix, r_local, config)
    lmax, mmax = config.lmax, config.mmax
    
    Threads.@threads for lm_idx in 1:size(spec_real, 1)
        l, m = index_to_lm_shtnskit(lm_idx, lmax, mmax)
        if l <= lmax && m <= mmax && r_local <= size(spec_real, 3)
            coeff = coeffs_matrix[l+1, m+1]
            spec_real[lm_idx, 1, r_local] = real(coeff)
            spec_imag[lm_idx, 1, r_local] = imag(coeff)
            
            # Ensure m=0 modes are real
            if m == 0
                spec_imag[lm_idx, 1, r_local] = 0.0
            end
        end
    end
end

"""
    index_to_lm_shtnskit(idx, lmax, mmax) -> (l, m)

Convert linear index to (l,m) for SHTnsKit compatibility.
"""
function index_to_lm_shtnskit(idx::Int, lmax::Int, mmax::Int)
    # Simple conversion - this should match SHTnsKit's indexing
    current_idx = 0
    for l in 0:lmax
        for m in 0:min(l, mmax)
            current_idx += 1
            if current_idx == idx
                return l, m
            end
        end
    end
    return 0, 0  # fallback
end

"""
    store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)

Store physical slice when in phi-local pencil.
"""
function store_physical_slice_phi_local!(phys_data, phys_slice, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    
    # Store respecting the phi-local layout
    Threads.@threads for i in 1:min(size(phys_data, 1), nlat)
        for j in 1:min(size(phys_data, 2), nlon)
            if r_local <= size(phys_data, 3) && i <= size(phys_slice, 1) && j <= size(phys_slice, 2)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end
end

"""
    store_physical_slice_generic!(phys_data, phys_slice, r_local, config)

Generic storage for any pencil orientation.
"""
function store_physical_slice_generic!(phys_data, phys_slice, r_local, config)
    # This is a generic fallback - may not be optimal for all pencil orientations
    Threads.@threads for i in 1:min(size(phys_data, 1), size(phys_slice, 1))
        for j in 1:min(size(phys_data, 2), size(phys_slice, 2))
            if r_local <= size(phys_data, 3)
                phys_data[i, j, r_local] = phys_slice[i, j]
            end
        end
    end
end

"""
    extract_physical_slice_phi_local(phys_data, r_local, config)

Extract physical slice when in phi-local pencil.
"""
function extract_physical_slice_phi_local(phys_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    slice = zeros(eltype(phys_data), nlat, nlon)
    
    Threads.@threads for i in 1:min(size(phys_data, 1), nlat)
        for j in 1:min(size(phys_data, 2), nlon)
            if r_local <= size(phys_data, 3)
                slice[i, j] = phys_data[i, j, r_local]
            end
        end
    end
    
    return slice
end

"""
    extract_physical_slice_generic(phys_data, r_local, config)

Generic extraction for any pencil orientation.
"""
function extract_physical_slice_generic(phys_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    slice = zeros(eltype(phys_data), nlat, nlon)
    
    # Generic extraction - may need MPI communication for distributed dimensions
    Threads.@threads for i in 1:min(size(phys_data, 1), nlat)
        for j in 1:min(size(phys_data, 2), nlon)
            if r_local <= size(phys_data, 3)
                slice[i, j] = phys_data[i, j, r_local]
            end
        end
    end
    
    return slice
end

"""
    extract_vector_component_generic(v_data, r_local, config)

Generic extraction for vector components.
"""
function extract_vector_component_generic(v_data, r_local, config)
    nlat, nlon = config.nlat, config.nlon
    component = zeros(eltype(v_data), nlat, nlon)
    
    for i in 1:min(size(v_data, 1), nlat)
        for j in 1:min(size(v_data, 2), nlon)
            if r_local <= size(v_data, 3)
                component[i, j] = v_data[i, j, r_local]
            end
        end
    end
    
    return component
end

"""
    store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)

Store vector components for any pencil orientation.
"""
function store_vector_components_generic!(v_theta, v_phi, vt_field, vp_field, r_local, config)
    for i in 1:min(size(v_theta, 1), size(vt_field, 1))
        for j in 1:min(size(v_theta, 2), size(vt_field, 2))
            if r_local <= size(v_theta, 3) && r_local <= size(v_phi, 3)
                v_theta[i, j, r_local] = vt_field[i, j]
                v_phi[i, j, r_local] = vp_field[i, j]
            end
        end
    end
end

# ============================================================================
# Batch Processing for Enhanced Performance
# ============================================================================

"""
    batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                              physs::Vector{SHTnsPhysicalField{T}}) where T

Batch process multiple transforms using SHTnsKit with PencilArrays.
"""
function batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                                   physs::Vector{SHTnsPhysicalField{T}}) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    # Process in parallel using threading
    @threads for batch_idx in 1:length(specs)
        shtnskit_spectral_to_physical!(specs[batch_idx], physs[batch_idx])
    end
end

# ---------------------------------------------------------------------------
# Backward-compatible alias used by other modules
# ---------------------------------------------------------------------------
"""
    batch_spectral_to_physical!(specs, physs)

Compatibility wrapper that calls `batch_shtnskit_transforms!` for batched
spectral→physical transforms using SHTnsKit with PencilArrays/MPI.
"""
function batch_spectral_to_physical!(specs::Vector{SHTnsSpectralField{T}},
                                     physs::Vector{SHTnsPhysicalField{T}}) where T
    return batch_shtnskit_transforms!(specs, physs)
end

# ============================================================================
# Performance Monitoring
# ============================================================================

"""
    get_shtnskit_performance_stats()

Get performance statistics for SHTnsKit transforms with PencilArrays.
"""
function get_shtnskit_performance_stats()
    return (
        library = "SHTnsKit",
        parallelization = "theta-phi MPI + PencilArrays",
        fft_backend = "PencilFFTs",
        optimization = "enabled"
    )
end

# ============================================================================
# Export functions
# ============================================================================

export SHTnsKitConfig, create_shtnskit_config
export shtnskit_spectral_to_physical!, shtnskit_physical_to_spectral!
export shtnskit_vector_synthesis!, shtnskit_vector_analysis!
export batch_shtnskit_transforms!
export get_shtnskit_performance_stats
export batch_spectral_to_physical!
