# ============================================================================
# SHTnsKit Spherical Harmonic Transforms
# ============================================================================
# 
# This module implements spherical harmonic transforms using SHTnsKit.jl
# with MPI parallelization across theta and phi directions using PencilArrays
#

using SHTnsKit
using MPI
using PencilArrays
using PencilArrays.Transpositions
using PencilFFTs
using FFTW
using LinearAlgebra
using Base.Threads

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
    
    # Performance optimization
    plan::Union{SHTnsKit.SHTPlan, Nothing}
    
    # Memory estimate
    memory_estimate::String
end

"""
    create_shtnskit_config(; lmax::Int, mmax::Int=lmax, nlat::Int=lmax+2, 
                           nlon::Int=max(2*lmax+1, 4), optimize_decomp::Bool=true) -> SHTnsKitConfig

Create SHTnsKit configuration with MPI parallelization using PencilArrays.
This replaces the old SHTns configuration with a cleaner SHTnsKit-based approach.
"""
function create_shtnskit_config(; lmax::Int, mmax::Int=lmax, 
                               nlat::Int=max(lmax+2, i_Th), 
                               nlon::Int=max(2*lmax+1, 4, i_Ph),
                               optimize_decomp::Bool=true)
    
    # Create SHTnsKit configuration with Gauss-Legendre grid
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
    pencils = create_pencil_decomposition_shtnskit(nlat, nlon, i_N, comm, optimize_decomp)
    
    # Create PencilFFTs plans for efficient phi-direction transforms
    fft_plans = create_pencil_fft_plans(pencils, (nlat, nlon, i_N))
    
    # Create transpose plans between different pencil orientations
    transpose_plans = create_shtnskit_transpose_plans(pencils)
    
    # Create optimized transform plan for repeated operations
    plan = try
        SHTnsKit.SHTPlan(sht_config)
    catch e
        @warn "Could not create SHTPlan: $e"
        nothing
    end
    
    # Estimate memory usage
    field_count = estimate_field_count()
    memory_mb = estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, Float64)
    memory_estimate = "$(round(memory_mb, digits=1)) MB"
    
    nlm = sht_config.nlm
    
    if get_rank() == 0
        print_shtnskit_config_summary(nlat, nlon, lmax, mmax, nlm, nprocs, memory_estimate)
    end
    
    return SHTnsKitConfig(sht_config, nlat, nlon, lmax, mmax, nlm, 
                         pencils, fft_plans, transpose_plans, plan, memory_estimate)
end

"""
    create_pencil_decomposition_shtnskit(nlat, nlon, nr, comm, optimize)

Create PencilArrays decomposition optimized for theta-phi parallelization.
"""
function create_pencil_decomposition_shtnskit(nlat::Int, nlon::Int, nr::Int, 
                                             comm, optimize::Bool=true)
    nprocs = MPI.Comm_size(comm)
    
    # Determine optimal process topology for theta-phi parallelization
    if optimize && nprocs > 1
        proc_dims = optimize_process_topology_shtnskit(nprocs, nlat, nlon)
    else
        proc_dims = (nprocs, 1)
    end
    
    # Create PencilArrays topology
    topology = PencilArrays.Topology(comm, proc_dims)
    
    # Physical space pencils for theta-phi parallelization
    dims = (nlat, nlon, nr)
    pencil_theta = Pencil(topology, dims, (2, 3))  # Theta-contiguous
    pencil_phi = Pencil(topology, dims, (1, 3))    # Phi-contiguous  
    pencil_r = Pencil(topology, dims, (1, 2))      # Radial-contiguous
    
    # Spectral space pencil (for (l,m) modes)
    nlm = SHTnsKit.nlm_calc(maximum([nlat-1, nlon÷2]), 
                           min(maximum([nlat-1, nlon÷2]), 
                               minimum([nlat-1, nlon÷2])), 1)
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
        # This uses the phi-pencil orientation where phi is local
        if haskey(pencils, :phi)
            # Forward FFT plan for phi direction
            dummy_phi = PencilArray{ComplexF64}(undef, pencils.phi)
            fft_plans[:phi_forward] = PencilFFTs.plan_fft!(dummy_phi, 2)  # FFT along phi (dim 2)
            fft_plans[:phi_backward] = PencilFFTs.plan_ifft!(dummy_phi, 2) # IFFT along phi
        end
        
        # Create plans for theta-pencil orientation if needed
        if haskey(pencils, :theta)
            dummy_theta = PencilArray{ComplexF64}(undef, pencils.theta)
            # For theta pencil, phi is distributed, so we need different approach
            fft_plans[:theta_phi_forward] = PencilFFTs.plan_fft!(dummy_theta, 2)
            fft_plans[:theta_phi_backward] = PencilFFTs.plan_ifft!(dummy_theta, 2)
        end
        
        @info "PencilFFTs plans created successfully"
    catch e
        @warn "Could not create PencilFFTs plans: $e. Falling back to regular FFTW."
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
        # Create common transpose operations needed for SHT
        if haskey(pencils, :theta) && haskey(pencils, :phi)
            # Transpose from theta-pencil to phi-pencil (for FFTs)
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
        
        if haskey(pencils, :spec) && haskey(pencils, :theta)
            # Transpose between spectral and physical representations
            transpose_plans[:spec_to_theta] = Transpose(pencils.spec => pencils.theta)
            transpose_plans[:theta_to_spec] = Transpose(pencils.theta => pencils.spec)
        end
        
        @info "Created $(length(transpose_plans)) transpose plans"
    catch e
        @warn "Could not create all transpose plans: $e"
    end
    
    return transpose_plans
end

"""
    estimate_memory_usage_shtnskit(nlat, nlon, lmax, field_count, T)

Estimate memory usage for SHTnsKit-based transforms.
"""
function estimate_memory_usage_shtnskit(nlat::Int, nlon::Int, lmax::Int, 
                                       field_count::Int, ::Type{T}) where T
    
    # Physical grid memory
    physical_memory = nlat * nlon * i_N * sizeof(T)
    
    # Spectral memory (approximate)
    nlm = (lmax + 1) * (lmax + 2) ÷ 2
    spectral_memory = nlm * i_N * sizeof(ComplexF64) * 2  # real + imag
    
    # Working arrays (SHTnsKit buffers)
    working_memory = max(physical_memory, spectral_memory) * 2
    
    # Total per field
    per_field_memory = physical_memory + spectral_memory + working_memory
    
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
    println("║   Theta-Phi Parallel: Enabled                        ║")
    println("║   Memory/process:   $(lpad(memory_estimate,10))                    ║")
    println("╚═══════════════════════════════════════════════════════╝")
end

# ============================================================================
# Core Transform Functions using SHTnsKit
# ============================================================================

"""
    shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                  phys::SHTnsPhysicalField{T}) where T

Transform from spectral to physical space using SHTnsKit with MPI parallelization.
This function uses PencilArrays and PencilFFTs for optimal parallel performance.
"""
function shtnskit_spectral_to_physical!(spec::SHTnsSpectralField{T}, 
                                       phys::SHTnsPhysicalField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Determine if we need to transpose to phi-pencil for FFTs
    current_pencil_spec = spec.pencil
    target_pencil_phys = phys.pencil
    
    # Check if physical field is in phi-pencil orientation for efficient FFTs
    need_transpose_for_fft = !is_phi_local(target_pencil_phys)
    
    if need_transpose_for_fft && haskey(config.transpose_plans, :theta_to_phi)
        # Transpose to phi-pencil for FFT operations
        phys_phi_data = transpose_to_phi_pencil(phys, config)
        perform_synthesis_with_pencil_fft!(spec, phys_phi_data, config, :phi)
        # Transpose back to original pencil orientation
        transpose_from_phi_pencil!(phys_phi_data, phys, config)
    else
        # Direct synthesis without transpose (if already in suitable orientation)
        perform_synthesis_with_pencil_fft!(spec, phys, config, get_pencil_orientation(target_pencil_phys))
    end
    
    # Ensure all MPI processes are synchronized
    MPI.Barrier(get_comm())
end

"""
    shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                  spec::SHTnsSpectralField{T}) where T

Transform from physical to spectral space using SHTnsKit with MPI parallelization.
"""
function shtnskit_physical_to_spectral!(phys::SHTnsPhysicalField{T}, 
                                       spec::SHTnsSpectralField{T}) where T
    config = spec.config
    sht_config = config.sht_config
    
    # Get local data
    phys_data = parent(phys.data)
    spec_real = parent(spec.data_real)
    spec_imag = parent(spec.data_imag)
    
    # Get local ranges
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(phys_data, 3)
            # Extract physical slice
            phys_slice = extract_physical_slice(phys_data, local_r, config)
            
            # Perform SHTnsKit analysis
            if config.plan !== nothing
                # Use optimized plan
                coeffs = zeros(ComplexF64, config.lmax+1, config.mmax+1)
                SHTnsKit.analysis!(config.plan, coeffs, phys_slice)
            else
                # Direct analysis
                coeffs = SHTnsKit.analysis(sht_config, phys_slice)
            end
            
            # Store spectral coefficients with parallelization
            store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)
        end
    end
    
    # Synchronize across MPI processes
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                              pol_spec::SHTnsSpectralField{T},
                              vec_phys::SHTnsVectorField{T}) where T

Vector synthesis using SHTnsKit spheroidal-toroidal decomposition.
"""
function shtnskit_vector_synthesis!(tor_spec::SHTnsSpectralField{T}, 
                                   pol_spec::SHTnsSpectralField{T},
                                   vec_phys::SHTnsVectorField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get local data
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    # Get local ranges for parallelization
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with theta-phi parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(tor_real, 3)
            # Extract toroidal and poloidal coefficients
            tor_coeffs = extract_spectral_coefficients(tor_real, tor_imag, local_r, config)
            pol_coeffs = extract_spectral_coefficients(pol_real, pol_imag, local_r, config)
            
            # Perform vector synthesis using SHTnsKit
            vt_field, vp_field = SHTnsKit.SHsphtor_to_spat(sht_config, pol_coeffs, tor_coeffs; 
                                                          real_output=true)
            
            # Store vector components with parallelization
            store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)
        end
    end
    
    # Synchronize across processes
    MPI.Barrier(get_comm())
end

"""
    shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                             tor_spec::SHTnsSpectralField{T}, 
                             pol_spec::SHTnsSpectralField{T}) where T

Vector analysis using SHTnsKit spheroidal-toroidal decomposition.
"""
function shtnskit_vector_analysis!(vec_phys::SHTnsVectorField{T},
                                  tor_spec::SHTnsSpectralField{T}, 
                                  pol_spec::SHTnsSpectralField{T}) where T
    config = tor_spec.config
    sht_config = config.sht_config
    
    # Get local data
    v_theta = parent(vec_phys.θ_component.data)
    v_phi = parent(vec_phys.φ_component.data)
    
    tor_real = parent(tor_spec.data_real)
    tor_imag = parent(tor_spec.data_imag)
    pol_real = parent(pol_spec.data_real)
    pol_imag = parent(pol_spec.data_imag)
    
    # Get local ranges
    r_range = range_local(config.pencils.r, 3)
    
    # Process each radial level with parallelization
    @threads for r_idx in r_range
        local_r = r_idx - first(r_range) + 1
        
        if local_r <= size(v_theta, 3)
            # Extract vector components
            vt_field = extract_physical_slice(v_theta, local_r, config)
            vp_field = extract_physical_slice(v_phi, local_r, config)
            
            # Perform vector analysis using SHTnsKit
            pol_coeffs, tor_coeffs = SHTnsKit.spat_to_SHsphtor(sht_config, vt_field, vp_field)
            
            # Store spectral coefficients
            store_spectral_coefficients!(pol_real, pol_imag, pol_coeffs, local_r, config)
            store_spectral_coefficients!(tor_real, tor_imag, tor_coeffs, local_r, config)
        end
    end
    
    # Synchronize across processes
    MPI.Barrier(get_comm())
end

# ============================================================================
# PencilArray and PencilFFT Helper Functions  
# ============================================================================

"""
    is_phi_local(pencil::Pencil{3}) -> Bool

Check if phi dimension is local (contiguous) in the given pencil.
"""
function is_phi_local(pencil::Pencil{3})
    # Check if phi (dimension 2) is in the contiguous dimensions
    return 2 in pencil.axes_in
end

"""
    get_pencil_orientation(pencil::Pencil{3}) -> Symbol

Get the orientation of a pencil (which dimensions are local).
"""
function get_pencil_orientation(pencil::Pencil{3})
    local_dims = pencil.axes_in
    if 1 in local_dims && 2 in local_dims
        return :theta_phi  # Both theta and phi local
    elseif 1 in local_dims
        return :theta      # Only theta local
    elseif 2 in local_dims  
        return :phi        # Only phi local
    else
        return :r          # Radial local
    end
end

"""
    transpose_to_phi_pencil(phys::SHTnsPhysicalField{T}, config) -> PencilArray

Transpose physical field to phi-pencil orientation for efficient FFTs.
"""
function transpose_to_phi_pencil(phys::SHTnsPhysicalField{T}, config) where T
    if haskey(config.transpose_plans, :theta_to_phi)
        phi_data = PencilArray{T}(undef, config.pencils.phi)
        mul!(phi_data, config.transpose_plans[:theta_to_phi], phys.data)
        return phi_data
    else
        @warn "No theta_to_phi transpose plan available"
        return phys.data  # Return original data as fallback
    end
end

"""
    transpose_from_phi_pencil!(phi_data, phys::SHTnsPhysicalField{T}, config) where T

Transpose from phi-pencil back to original pencil orientation.
"""
function transpose_from_phi_pencil!(phi_data, phys::SHTnsPhysicalField{T}, config) where T
    if haskey(config.transpose_plans, :phi_to_theta)
        mul!(phys.data, config.transpose_plans[:phi_to_theta], phi_data)
    else
        @warn "No phi_to_theta transpose plan available"
        # Copy data directly as fallback
        copyto!(parent(phys.data), parent(phi_data))
    end
end

"""
    perform_synthesis_with_pencil_fft!(spec, phys, config, orientation)

Perform synthesis using PencilFFTs based on pencil orientation.
"""
function perform_synthesis_with_pencil_fft!(spec::SHTnsSpectralField{T}, 
                                          phys::Union{SHTnsPhysicalField{T}, PencilArray{T,3}}, 
                                          config, orientation::Symbol) where T
    sht_config = config.sht_config
    
    # Get data arrays
    spec_real = spec.data_real
    spec_imag = spec.data_imag
    phys_data = isa(phys, SHTnsPhysicalField) ? phys.data : phys
    
    # Get local ranges for this pencil orientation
    r_range = get_local_radial_range(phys_data)
    
    # Process each radial level
    @threads for r_local in r_range
        # Extract spectral coefficients for this radial level
        coeffs_matrix = extract_spectral_coefficients_pencil(spec_real, spec_imag, r_local, config)
        
        # Perform synthesis for this radial slice
        if config.plan !== nothing
            # Use optimized SHTnsKit plan
            phys_slice = zeros(T, config.nlat, config.nlon)
            SHTnsKit.synthesis!(config.plan, phys_slice, coeffs_matrix)
        else
            # Direct synthesis
            phys_slice = SHTnsKit.synthesis(sht_config, coeffs_matrix; real_output=true)
        end
        
        # Store result in PencilArray with proper data layout
        store_physical_slice_pencil!(phys_data, phys_slice, r_local, config, orientation)
    end
    
    # Apply PencilFFTs if in phi-pencil orientation
    if orientation == :phi && haskey(config.fft_plans, :phi_backward)
        try
            # Convert to complex for IFFT if needed (for compatibility)
            if eltype(phys_data) <: Real
                complex_data = complex.(phys_data)
                # Apply IFFT using PencilFFTs
                config.fft_plans[:phi_backward] * complex_data
                # Extract real part back to original array
                copyto!(parent(phys_data), real.(parent(complex_data)))
            else
                config.fft_plans[:phi_backward] * phys_data
            end
        catch e
            @warn "PencilFFT failed, using fallback: $e"
            # Fallback to regular FFT processing per slice
            fallback_fft_synthesis!(phys_data, config, orientation)
        end
    end
end

"""
    fallback_fft_synthesis!(phys_data, config, orientation)

Fallback FFT synthesis when PencilFFTs is not available.
"""
function fallback_fft_synthesis!(phys_data, config, orientation)
    # Simple fallback - data is already synthesized by SHTnsKit
    # No additional FFT processing needed as SHTnsKit handles it internally
    @debug "Using SHTnsKit internal FFT processing"
end

"""
    get_local_radial_range(phys_data::PencilArray{T,3}) where T

Get the local radial range for the current MPI process.
"""
function get_local_radial_range(phys_data::PencilArray{T,3}) where T
    return 1:size(phys_data, 3)  # Local radial extent
end

# ============================================================================
# Helper Functions for Data Management
# ============================================================================

"""
    extract_spectral_coefficients(spec_real, spec_imag, local_r, config)

Extract spectral coefficients for a radial level with proper MPI communication.
"""
function extract_spectral_coefficients(spec_real, spec_imag, local_r, config)
    nlm = config.nlm
    lmax, mmax = config.lmax, config.mmax
    
    # Initialize coefficient matrix
    coeffs = zeros(ComplexF64, lmax+1, mmax+1)
    
    # Get local spectral data
    if local_r <= size(spec_real, 3)
        for lm_idx in 1:min(size(spec_real, 1), nlm)
            # Convert from packed format to (l,m) matrix format
            l, m = index_to_lm(lm_idx, lmax, mmax)
            if l <= lmax && m <= mmax
                real_part = spec_real[lm_idx, 1, local_r]
                imag_part = spec_imag[lm_idx, 1, local_r]
                coeffs[l+1, m+1] = complex(real_part, imag_part)
            end
        end
    end
    
    # Communicate across MPI processes to gather complete spectrum
    MPI.Allreduce!(coeffs, MPI.SUM, get_comm())
    
    return coeffs
end

"""
    store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)

Store spectral coefficients with proper parallel decomposition.
"""
function store_spectral_coefficients!(spec_real, spec_imag, coeffs, local_r, config)
    lmax, mmax = config.lmax, config.mmax
    
    if local_r <= size(spec_real, 3)
        for lm_idx in 1:min(size(spec_real, 1), config.nlm)
            # Convert from (l,m) matrix to packed format
            l, m = index_to_lm(lm_idx, lmax, mmax)
            if l <= lmax && m <= mmax
                coeff = coeffs[l+1, m+1]
                spec_real[lm_idx, 1, local_r] = real(coeff)
                spec_imag[lm_idx, 1, local_r] = imag(coeff)
                
                # Ensure m=0 modes are real
                if m == 0
                    spec_imag[lm_idx, 1, local_r] = 0.0
                end
            end
        end
    end
end

"""
    extract_physical_slice(phys_data, local_r, config)

Extract a physical space slice for a given radial level.
"""
function extract_physical_slice(phys_data, local_r, config)
    nlat, nlon = config.nlat, config.nlon
    slice = zeros(eltype(phys_data), nlat, nlon)
    
    if local_r <= size(phys_data, 3)
        for i in 1:nlat, j in 1:nlon
            if i <= size(phys_data, 1) && j <= size(phys_data, 2)
                slice[i, j] = phys_data[i, j, local_r]
            end
        end
    end
    
    return slice
end

"""
    store_physical_slice!(phys_data, phys_slice, local_r, config)

Store a physical space slice with proper bounds checking.
"""
function store_physical_slice!(phys_data, phys_slice, local_r, config)
    if local_r <= size(phys_data, 3)
        for i in 1:size(phys_slice, 1), j in 1:size(phys_slice, 2)
            if i <= size(phys_data, 1) && j <= size(phys_data, 2)
                phys_data[i, j, local_r] = phys_slice[i, j]
            end
        end
    end
end

"""
    store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)

Store vector components with proper parallelization.
"""
function store_vector_components!(v_theta, v_phi, vt_field, vp_field, local_r, config)
    if local_r <= size(v_theta, 3) && local_r <= size(v_phi, 3)
        for i in 1:size(vt_field, 1), j in 1:size(vt_field, 2)
            if i <= size(v_theta, 1) && j <= size(v_theta, 2) && 
               i <= size(v_phi, 1) && j <= size(v_phi, 2)
                v_theta[i, j, local_r] = vt_field[i, j]
                v_phi[i, j, local_r] = vp_field[i, j]
            end
        end
    end
end

"""
    index_to_lm(idx, lmax, mmax)

Convert linear index to (l,m) indices for spherical harmonics.
"""
function index_to_lm(idx::Int, lmax::Int, mmax::Int)
    # Simple conversion - can be optimized with lookup tables
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

# ============================================================================
# Batch Processing for Better Performance
# ============================================================================

"""
    batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                              physs::Vector{SHTnsPhysicalField{T}}) where T

Batch process multiple spectral to physical transforms for better performance.
"""
function batch_shtnskit_transforms!(specs::Vector{SHTnsSpectralField{T}},
                                   physs::Vector{SHTnsPhysicalField{T}}) where T
    @assert length(specs) == length(physs)
    
    if isempty(specs)
        return
    end
    
    config = specs[1].config
    
    # Process in parallel batches for better cache utilization
    @threads for batch_idx in 1:length(specs)
        shtnskit_spectral_to_physical!(specs[batch_idx], physs[batch_idx])
    end
end

# ============================================================================
# Performance Monitoring
# ============================================================================

"""
    get_shtnskit_performance_stats()

Get performance statistics for SHTnsKit transforms.
"""
function get_shtnskit_performance_stats()
    # Return basic performance metrics
    return (
        library = "SHTnsKit",
        parallelization = "theta-phi MPI",
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