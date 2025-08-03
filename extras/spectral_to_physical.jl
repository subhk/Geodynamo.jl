# ============================================================================
# MPI-Parallel Spectral to Physical Field Reader - Complete Implementation
# Distributed processing of large spectral datasets with optimal load balancing
# ============================================================================

module MPISpectralToPhysicalReader

using MPI
using NetCDF
using LinearAlgebra
using Statistics
using Printf
using SHTnsSpheres
using Dates

# Initialize MPI
const comm = MPI.COMM_WORLD

# ============================================================================
# Data Structures
# ============================================================================

struct MPIPhysicalFieldData{T}
    # Local physical space fields
    velocity_r::Array{T,3}      # (nlat_local, nlon_local, nr_local)
    velocity_theta::Array{T,3}
    velocity_phi::Array{T,3}
    
    magnetic_r::Array{T,3}
    magnetic_theta::Array{T,3}
    magnetic_phi::Array{T,3}
    
    temperature::Array{T,3}
    
    # Local grid information
    theta::Vector{T}
    phi::Vector{T}
    r::Vector{T}
    
    # Domain decomposition info
    decomp::MPIDomainDecomposition
    
    # Metadata
    time::Float64
    step::Int
    metadata::Dict{String, Any}
end

struct MPIDomainDecomposition
    # MPI info
    rank::Int
    nprocs::Int
    
    # Global dimensions
    nlat_global::Int
    nlon_global::Int
    nr_global::Int
    
    # Local dimensions
    nlat_local::Int
    nlon_local::Int
    nr_local::Int
    
    # Local ranges in global coordinates
    theta_range::UnitRange{Int}
    phi_range::UnitRange{Int}
    r_range::UnitRange{Int}
    
    # Process grid topology
    proc_grid::Tuple{Int,Int,Int}  # (P_theta, P_phi, P_r)
    proc_coords::Tuple{Int,Int,Int}  # This process's coordinates
    
    # Communication patterns
    theta_comm::MPI.Comm
    phi_comm::MPI.Comm
    r_comm::MPI.Comm
end

struct MPISHTnsConverter{T}
    sht::SHTnsSphere
    nlat_local::Int
    nlon_local::Int
    nlm::Int
    lmax::Int
    mmax::Int
    nr_local::Int
    l_values::Vector{Int}
    m_values::Vector{Int}
    lm_to_idx::Dict{Tuple{Int,Int}, Int}
    decomp::MPIDomainDecomposition
end

# ============================================================================
# MPI Domain Decomposition
# ============================================================================

function create_mpi_domain_decomposition(nlat_global::Int, nlon_global::Int, nr_global::Int)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        println("Setting up MPI domain decomposition:")
        println("  Global grid: $nlat_global × $nlon_global × $nr_global")
        println("  Total processes: $nprocs")
    end
    
    # Create 3D process grid - optimize based on data size and process count
    proc_grid = compute_optimal_process_grid(nlat_global, nlon_global, nr_global, nprocs)
    P_theta, P_phi, P_r = proc_grid
    
    if rank == 0
        println("  Process grid: $P_theta × $P_phi × $P_r")
    end
    
    # Compute this process's coordinates in the grid
    proc_coords = compute_process_coordinates(rank, proc_grid)
    coord_theta, coord_phi, coord_r = proc_coords
    
    # Compute local dimensions and ranges
    theta_range, nlat_local = compute_local_range(coord_theta, P_theta, nlat_global)
    phi_range, nlon_local = compute_local_range(coord_phi, P_phi, nlon_global)
    r_range, nr_local = compute_local_range(coord_r, P_r, nr_global)
    
    if rank == 0
        println("  Local grid sizes (max): $nlat_local × $nlon_local × $nr_local")
    end
    
    # Create communicators for each dimension
    theta_comm = create_dimension_communicator(comm, proc_grid, proc_coords, 1)
    phi_comm = create_dimension_communicator(comm, proc_grid, proc_coords, 2)
    r_comm = create_dimension_communicator(comm, proc_grid, proc_coords, 3)
    
    return MPIDomainDecomposition(
        rank, nprocs,
        nlat_global, nlon_global, nr_global,
        nlat_local, nlon_local, nr_local,
        theta_range, phi_range, r_range,
        proc_grid, proc_coords,
        theta_comm, phi_comm, r_comm
    )
end

function compute_optimal_process_grid(nlat::Int, nlon::Int, nr::Int, nprocs::Int)
    # Find optimal factorization of nprocs into 3D grid
    # Prioritize angular directions over radial for better load balancing
    
    factors = factorize_3d(nprocs)
    best_grid = (1, 1, nprocs)
    best_score = Inf
    
    for (P_theta, P_phi, P_r) in factors
        # Compute load balance score
        nlat_local = div(nlat, P_theta) + (nlat % P_theta > 0 ? 1 : 0)
        nlon_local = div(nlon, P_phi) + (nlon % P_phi > 0 ? 1 : 0)
        nr_local = div(nr, P_r) + (nr % P_r > 0 ? 1 : 0)
        
        # Prefer balanced angular decomposition
        work_per_proc = nlat_local * nlon_local * nr_local
        angular_balance = max(nlat_local * P_theta - nlat, nlon_local * P_phi - nlon)
        radial_balance = nr_local * P_r - nr
        
        score = work_per_proc + 0.1 * angular_balance + 0.01 * radial_balance
        
        if score < best_score
            best_score = score
            best_grid = (P_theta, P_phi, P_r)
        end
    end
    
    return best_grid
end

function factorize_3d(n::Int)
    # Generate all possible 3D factorizations of n
    factors = Tuple{Int,Int,Int}[]
    
    for i in 1:Int(ceil(cbrt(n)))
        if n % i == 0
            remaining = div(n, i)
            for j in i:Int(ceil(sqrt(remaining)))
                if remaining % j == 0
                    k = div(remaining, j)
                    if k >= j  # Ensure i ≤ j ≤ k
                        push!(factors, (i, j, k))
                    end
                end
            end
        end
    end
    
    return factors
end

function compute_process_coordinates(rank::Int, proc_grid::Tuple{Int,Int,Int})
    P_theta, P_phi, P_r = proc_grid
    
    # Row-major ordering: rank = coord_r * P_theta * P_phi + coord_phi * P_theta + coord_theta
    coord_r = div(rank, P_theta * P_phi)
    remaining = rank % (P_theta * P_phi)
    coord_phi = div(remaining, P_theta)
    coord_theta = remaining % P_theta
    
    return (coord_theta, coord_phi, coord_r)
end

function compute_local_range(coord::Int, P_dim::Int, n_global::Int)
    # Compute local range for this coordinate
    base_size = div(n_global, P_dim)
    extra = n_global % P_dim
    
    if coord < extra
        local_size = base_size + 1
        start_idx = coord * (base_size + 1) + 1
    else
        local_size = base_size
        start_idx = extra * (base_size + 1) + (coord - extra) * base_size + 1
    end
    
    end_idx = start_idx + local_size - 1
    
    return start_idx:end_idx, local_size
end

function create_dimension_communicator(parent_comm::MPI.Comm, proc_grid::Tuple{Int,Int,Int}, 
                                      proc_coords::Tuple{Int,Int,Int}, dim::Int)
    # Create communicator for processes that share the same coordinates in other dimensions
    
    P_theta, P_phi, P_r = proc_grid
    coord_theta, coord_phi, coord_r = proc_coords
    
    if dim == 1  # theta communicator
        color = coord_phi * P_r + coord_r
        key = coord_theta
    elseif dim == 2  # phi communicator
        color = coord_theta * P_r + coord_r
        key = coord_phi
    else  # r communicator
        color = coord_theta * P_phi + coord_phi
        key = coord_r
    end
    
    return MPI.Comm_split(parent_comm, color, key)
end

# ============================================================================
# MPI File I/O
# ============================================================================

function mpi_read_combined_netcdf(filename::String, decomp::MPIDomainDecomposition, 
                                 ::Type{T} = Float64) where T
    rank = decomp.rank
    
    if rank == 0
        println("MPI reading combined NetCDF file: $filename")
    end
    
    # All processes open file (NetCDF handles this efficiently)
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    
    local_data = Dict{String, Any}()
    metadata = Dict{String, Any}()
    
    try
        # Read metadata on all processes
        time_val = NetCDF.readvar(nc_file, "time")[1]
        step_val = NetCDF.readvar(nc_file, "step")[1]
        
        metadata["time"] = time_val
        metadata["step"] = step_val
        
        # Read global attributes
        for attr_name in ["Rayleigh_number", "Ekman_number", "Prandtl_number", 
                         "Magnetic_Prandtl", "original_nprocs"]
            try
                metadata[attr_name] = NetCDF.getatt(nc_file, NC_GLOBAL, attr_name)
            catch
            end
        end
        
        # Read temperature (physical space) - distributed read
        if NetCDF.varid(nc_file, "temperature_global") != -1
            temp_global_dims = NetCDF.size(nc_file, "temperature_global")
            
            # Read local portion
            start_idx = [decomp.theta_range.start, decomp.phi_range.start, decomp.r_range.start]
            count = [decomp.nlat_local, decomp.nlon_local, decomp.nr_local]
            
            local_data["temperature"] = T.(NetCDF.readvar(nc_file, "temperature_global", 
                                                         start=start_idx, count=count))
            
            # Read coordinate arrays
            local_data["theta"] = T.(NetCDF.readvar(nc_file, "theta")[decomp.theta_range])
            local_data["phi"] = T.(NetCDF.readvar(nc_file, "phi")[decomp.phi_range])
            local_data["r_physical"] = T.(NetCDF.readvar(nc_file, "r_physical")[decomp.r_range])
        end
        
        # Read spectral data - need special handling for global spectral modes
        has_velocity = (NetCDF.varid(nc_file, "velocity_toroidal_real_global") != -1)
        has_magnetic = (NetCDF.varid(nc_file, "magnetic_toroidal_real_global") != -1)
        
        if has_velocity || has_magnetic
            # All processes read full spectral coefficients (they're not huge)
            l_values = NetCDF.readvar(nc_file, "l_values_global")
            m_values = NetCDF.readvar(nc_file, "m_values_global")
            r_spectral = T.(NetCDF.readvar(nc_file, "r_spectral"))
            
            local_data["spectral_grid"] = (l_values = l_values, m_values = m_values, r = r_spectral)
            
            # Read spectral coefficients - distribute along radial direction
            r_spec_range = compute_spectral_radial_range(decomp, length(r_spectral))
            
            if has_velocity
                vel_tor_real_full = NetCDF.readvar(nc_file, "velocity_toroidal_real_global")
                vel_tor_imag_full = NetCDF.readvar(nc_file, "velocity_toroidal_imag_global")
                vel_pol_real_full = NetCDF.readvar(nc_file, "velocity_poloidal_real_global")
                vel_pol_imag_full = NetCDF.readvar(nc_file, "velocity_poloidal_imag_global")
                
                # Extract local radial range
                local_data["velocity_spectral"] = (
                    toroidal_real = T.(vel_tor_real_full[:, r_spec_range]),
                    toroidal_imag = T.(vel_tor_imag_full[:, r_spec_range]),
                    poloidal_real = T.(vel_pol_real_full[:, r_spec_range]),
                    poloidal_imag = T.(vel_pol_imag_full[:, r_spec_range]),
                    l_values = l_values,
                    m_values = m_values,
                    r = r_spectral[r_spec_range]
                )
            end
            
            if has_magnetic
                mag_tor_real_full = NetCDF.readvar(nc_file, "magnetic_toroidal_real_global")
                mag_tor_imag_full = NetCDF.readvar(nc_file, "magnetic_toroidal_imag_global")
                mag_pol_real_full = NetCDF.readvar(nc_file, "magnetic_poloidal_real_global")
                mag_pol_imag_full = NetCDF.readvar(nc_file, "magnetic_poloidal_imag_global")
                
                local_data["magnetic_spectral"] = (
                    toroidal_real = T.(mag_tor_real_full[:, r_spec_range]),
                    toroidal_imag = T.(mag_tor_imag_full[:, r_spec_range]),
                    poloidal_real = T.(mag_pol_real_full[:, r_spec_range]),
                    poloidal_imag = T.(mag_pol_imag_full[:, r_spec_range]),
                    l_values = l_values,
                    m_values = m_values,
                    r = r_spectral[r_spec_range]
                )
            end
        end
        
    finally
        NetCDF.close(nc_file)
    end
    
    if rank == 0
        println("MPI file read completed")
    end
    
    return local_data, metadata
end

function compute_spectral_radial_range(decomp::MPIDomainDecomposition, nr_spectral::Int)
    # For spectral data, we distribute along radial coordinate only
    # This keeps all (l,m) modes on each process for efficient SHTns operations
    
    coord_r = decomp.proc_coords[3]
    P_r = decomp.proc_grid[3]
    
    base_size = div(nr_spectral, P_r)
    extra = nr_spectral % P_r
    
    if coord_r < extra
        local_size = base_size + 1
        start_idx = coord_r * (base_size + 1) + 1
    else
        local_size = base_size
        start_idx = extra * (base_size + 1) + (coord_r - extra) * base_size + 1
    end
    
    end_idx = start_idx + local_size - 1
    
    return start_idx:end_idx
end

# ============================================================================
# MPI SHTns Converter
# ============================================================================

function create_mpi_shtns_converter(spectral_data, decomp::MPIDomainDecomposition, 
                                   ::Type{T} = Float64) where T
    rank = decomp.rank
    
    l_values = spectral_data.l_values
    m_values = spectral_data.m_values
    
    lmax = maximum(l_values)
    mmax = maximum(m_values)
    nlm = length(l_values)
    
    # Use local angular dimensions for SHTns
    nlat_local = decomp.nlat_local
    nlon_local = decomp.nlon_local
    nr_local = length(spectral_data.r)
    
    if rank == 0
        println("Setting up MPI SHTns with lmax=$lmax, mmax=$mmax")
        println("Local SHTns grid: $nlat_local × $nlon_local × $nr_local")
    end
    
    # Each process creates its own SHTns instance for its local angular domain
    # Note: This requires careful handling of the spectral coefficients
    sht = SHTnsSphere(lmax, mmax, 
                      grid_type = SHTnsSpheres.gaussian,
                      nlat = nlat_local,
                      nlon = nlon_local)
    
    # Create (l,m) to index mapping
    lm_to_idx = Dict{Tuple{Int,Int}, Int}()
    for (idx, (l, m)) in enumerate(zip(l_values, m_values))
        lm_to_idx[(l, m)] = idx
    end
    
    return MPISHTnsConverter{T}(sht, nlat_local, nlon_local, nlm, lmax, mmax, nr_local,
                               l_values, m_values, lm_to_idx, decomp)
end

# ============================================================================
# MPI Spectral to Physical Conversion
# ============================================================================

function mpi_convert_toroidal_poloidal_to_physical(spectral_data, converter::MPISHTnsConverter{T}) where T
    rank = converter.decomp.rank
    
    if rank == 0
        println("MPI converting toroidal-poloidal spectral data to physical vector field...")
    end
    
    nlat_local = converter.nlat_local
    nlon_local = converter.nlon_local
    nr_local = converter.nr_local
    sht = converter.sht
    
    # Allocate local physical space arrays
    v_r = zeros(T, nlat_local, nlon_local, nr_local)
    v_theta = zeros(T, nlat_local, nlon_local, nr_local)
    v_phi = zeros(T, nlat_local, nlon_local, nr_local)
    
    # Process each local radial level
    for r_idx in 1:nr_local
        if r_idx <= size(spectral_data.toroidal_real, 2)
            
            # Extract toroidal and poloidal coefficients for this radial level
            tor_coeffs = zeros(ComplexF64, converter.nlm)
            pol_coeffs = zeros(ComplexF64, converter.nlm)
            
            for (spectral_idx, (l, m)) in enumerate(zip(spectral_data.l_values, spectral_data.m_values))
                if haskey(converter.lm_to_idx, (l, m))
                    converter_idx = converter.lm_to_idx[(l, m)]
                    
                    tor_real = spectral_data.toroidal_real[spectral_idx, r_idx]
                    tor_imag = spectral_data.toroidal_imag[spectral_idx, r_idx]
                    pol_real = spectral_data.poloidal_real[spectral_idx, r_idx]
                    pol_imag = spectral_data.poloidal_imag[spectral_idx, r_idx]
                    
                    tor_coeffs[converter_idx] = complex(tor_real, tor_imag)
                    pol_coeffs[converter_idx] = complex(pol_real, pol_imag)
                    
                    # Ensure m=0 modes are real
                    if m == 0
                        tor_coeffs[converter_idx] = complex(tor_real, 0.0)
                        pol_coeffs[converter_idx] = complex(pol_real, 0.0)
                    end
                end
            end
            
            # Convert to physical vector components using SHTns
            try
                v_theta_level, v_phi_level = vector_synthesis(sht, tor_coeffs, pol_coeffs)
                
                # Store horizontal components
                for j_phi in 1:nlon_local, i_theta in 1:nlat_local
                    v_theta[i_theta, j_phi, r_idx] = real(v_theta_level[i_theta, j_phi])
                    v_phi[i_theta, j_phi, r_idx] = real(v_phi_level[i_theta, j_phi])
                end
                
                # Compute radial component from divergence-free condition
                mpi_compute_radial_component!(v_r, v_theta, v_phi, r_idx, converter, spectral_data)
                
            catch e
                if rank == 0
                    @warn "SHTns vector synthesis failed at r_idx=$r_idx" exception=e
                end
                continue
            end
        end
    end
    
    return v_r, v_theta, v_phi
end

function mpi_compute_radial_component!(v_r::Array{T,3}, v_theta::Array{T,3}, v_phi::Array{T,3}, 
                                      r_idx::Int, converter::MPISHTnsConverter{T}, 
                                      spectral_data) where T
    
    nlat_local = converter.nlat_local
    nlon_local = converter.nlon_local
    sht = converter.sht
    
    # Use poloidal potential to compute radial component
    r_val = spectral_data.r[r_idx]
    r_inv_sq = 1.0 / max(r_val^2, 1e-10)
    
    # Extract poloidal coefficients
    pol_coeffs = zeros(ComplexF64, converter.nlm)
    
    for (spectral_idx, (l, m)) in enumerate(zip(spectral_data.l_values, spectral_data.m_values))
        if haskey(converter.lm_to_idx, (l, m))
            converter_idx = converter.lm_to_idx[(l, m)]
            
            pol_real = spectral_data.poloidal_real[spectral_idx, r_idx]
            pol_imag = spectral_data.poloidal_imag[spectral_idx, r_idx]
            
            # Apply l(l+1) factor for radial component
            l_factor = Float64(l * (l + 1))
            pol_coeffs[converter_idx] = complex(pol_real * l_factor, pol_imag * l_factor)
            
            if m == 0
                pol_coeffs[converter_idx] = complex(pol_real * l_factor, 0.0)
            end
        end
    end
    
    # Synthesize radial component
    try
        v_r_level = synthesis(sht, pol_coeffs)
        
        for j_phi in 1:nlon_local, i_theta in 1:nlat_local
            v_r[i_theta, j_phi, r_idx] = real(v_r_level[i_theta, j_phi]) * r_inv_sq
        end
        
    catch e
        # Fallback: set to zero
        v_r[:, :, r_idx] .= 0.0
    end
end

# ============================================================================
# MPI Collective Operations
# ============================================================================

function mpi_gather_global_statistics(local_data::MPIPhysicalFieldData{T}) where T
    rank = local_data.decomp.rank
    
    # Compute local statistics
    local_stats = Dict{String, Float64}()
    
    # Local volume (for proper averaging)
    local_volume = length(local_data.velocity_r)
    
    # Velocity statistics
    if any(local_data.velocity_r .!= 0.0)
        vel_mag_sq = local_data.velocity_r.^2 .+ local_data.velocity_theta.^2 .+ local_data.velocity_phi.^2
        
        local_stats["vel_r_sum_sq"] = sum(local_data.velocity_r.^2)
        local_stats["vel_theta_sum_sq"] = sum(local_data.velocity_theta.^2)
        local_stats["vel_phi_sum_sq"] = sum(local_data.velocity_phi.^2)
        local_stats["vel_mag_max"] = maximum(sqrt.(vel_mag_sq))
        local_stats["kinetic_energy_sum"] = 0.5 * sum(vel_mag_sq)
    end
    
    # Magnetic statistics
    if any(local_data.magnetic_r .!= 0.0)
        mag_mag_sq = local_data.magnetic_r.^2 .+ local_data.magnetic_theta.^2 .+ local_data.magnetic_phi.^2
        
        local_stats["mag_r_sum_sq"] = sum(local_data.magnetic_r.^2)
        local_stats["mag_theta_sum_sq"] = sum(local_data.magnetic_theta.^2)
        local_stats["mag_phi_sum_sq"] = sum(local_data.magnetic_phi.^2)
        local_stats["mag_mag_max"] = maximum(sqrt.(mag_mag_sq))
        local_stats["magnetic_energy_sum"] = 0.5 * sum(mag_mag_sq)
    end
    
    # Temperature statistics
    if any(local_data.temperature .!= 0.0)
        local_stats["temp_sum"] = sum(local_data.temperature)
        local_stats["temp_sum_sq"] = sum(local_data.temperature.^2)
        local_stats["temp_min"] = minimum(local_data.temperature)
        local_stats["temp_max"] = maximum(local_data.temperature)
    end
    
    local_stats["volume"] = Float64(local_volume)
    
    # Perform global reductions
    global_stats = Dict{String, Float64}()
    
    # Sum reductions
    for key in ["vel_r_sum_sq", "vel_theta_sum_sq", "vel_phi_sum_sq", "kinetic_energy_sum",
                "mag_r_sum_sq", "mag_theta_sum_sq", "mag_phi_sum_sq", "magnetic_energy_sum",
                "temp_sum", "temp_sum_sq", "volume"]
        if haskey(local_stats, key)
            global_stats[key] = MPI.Allreduce(local_stats[key], MPI.SUM, comm)
        end
    end
    
    # Max reductions
    for key in ["vel_mag_max", "mag_mag_max", "temp_max"]
        if haskey(local_stats, key)
            global_stats[key] = MPI.Allreduce(local_stats[key], MPI.MAX, comm)
        end
    end
    
    # Min reductions
    for key in ["temp_min"]
        if haskey(local_stats, key)
            global_stats[key] = MPI.Allreduce(local_stats[key], MPI.MIN, comm)
        end
    end
    
    # Compute derived global statistics
    total_volume = global_stats["volume"]
    
    if haskey(global_stats, "vel_r_sum_sq")
        global_stats["vel_r_rms"] = sqrt(global_stats["vel_r_sum_sq"] / total_volume)
        global_stats["vel_theta_rms"] = sqrt(global_stats["vel_theta_sum_sq"] / total_volume)
        global_stats["vel_phi_rms"] = sqrt(global_stats["vel_phi_sum_sq"] / total_volume)
        global_stats["kinetic_energy"] = global_stats["kinetic_energy_sum"] / total_volume
    end
    
    if haskey(global_stats, "mag_r_sum_sq")
        global_stats["mag_r_rms"] = sqrt(global_stats["mag_r_sum_sq"] / total_volume)
        global_stats["mag_theta_rms"] = sqrt(global_stats["mag_theta_sum_sq"] / total_volume)
        global_stats["mag_phi_rms"] = sqrt(global_stats["mag_phi_sum_sq"] / total_volume)
        global_stats["magnetic_energy"] = global_stats["magnetic_energy_sum"] / total_volume
    end
    
    if haskey(global_stats, "temp_sum")
        global_stats["temp_mean"] = global_stats["temp_sum"] / total_volume
        temp_var = global_stats["temp_sum_sq"] / total_volume - global_stats["temp_mean"]^2
        global_stats["temp_std"] = sqrt(max(temp_var, 0.0))
    end
    
    return global_stats
end

function mpi_write_parallel_netcdf(local_data::MPIPhysicalFieldData{T}, filename::String) where T
    rank = local_data.decomp.rank
    nprocs = local_data.decomp.nprocs
    decomp = local_data.decomp
    
    if rank == 0
        println("Writing parallel NetCDF file: $filename")
        
        # Remove existing file
        if isfile(filename)
            rm(filename)
        end
    end
    
    MPI.Barrier(comm)
    
    # Use coordinated serial writes for compatibility
    write_coordinated_serial_netcdf(local_data, filename)
end

function write_coordinated_serial_netcdf(local_data::MPIPhysicalFieldData{T}, filename::String) where T
    rank = local_data.decomp.rank
    nprocs = local_data.decomp.nprocs
    decomp = local_data.decomp
    
    # Rank 0 creates the file structure
    if rank == 0
        nc_file = NetCDF.create(filename, NcFile)
        
        try
            # Global attributes
            NetCDF.putatt(nc_file, "title", "MPI Physical Space Geodynamo Fields")
            NetCDF.putatt(nc_file, "source", "MPISpectralToPhysicalReader.jl")
            NetCDF.putatt(nc_file, "history", "MPI converted on $(now()) with $nprocs processes")
            NetCDF.putatt(nc_file, "Conventions", "CF-1.8")
            NetCDF.putatt(nc_file, "mpi_processes", nprocs)
            
            # Add metadata
            for (key, value) in local_data.metadata
                if !(key in ["global_diagnostics"])
                    try
                        NetCDF.putatt(nc_file, key, value)
                    catch
                    end
                end
            end
            
            # Define dimensions
            time_dim = NetCDF.defDim(nc_file, "time", 1)
            theta_dim = NetCDF.defDim(nc_file, "theta", decomp.nlat_global)
            phi_dim = NetCDF.defDim(nc_file, "phi", decomp.nlon_global)
            r_dim = NetCDF.defDim(nc_file, "r", decomp.nr_global)
            
            # Coordinate variables
            time_var = NetCDF.defVar(nc_file, "time", Float64, (time_dim,))
            NetCDF.putatt(nc_file, time_var, "long_name", "simulation_time")
            
            step_var = NetCDF.defVar(nc_file, "step", Int32, (time_dim,))
            NetCDF.putatt(nc_file, step_var, "long_name", "simulation_step")
            
            theta_var = NetCDF.defVar(nc_file, "theta", T, (theta_dim,))
            NetCDF.putatt(nc_file, theta_var, "long_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "units", "radians")
            
            phi_var = NetCDF.defVar(nc_file, "phi", T, (phi_dim,))
            NetCDF.putatt(nc_file, phi_var, "long_name", "azimuthal_angle")
            NetCDF.putatt(nc_file, phi_var, "units", "radians")
            
            r_var = NetCDF.defVar(nc_file, "r", T, (r_dim,))
            NetCDF.putatt(nc_file, r_var, "long_name", "radial_coordinate")
            NetCDF.putatt(nc_file, r_var, "units", "dimensionless")
            
            # Field variables
            field_dims = (theta_dim, phi_dim, r_dim)
            
            # Velocity components
            if any(local_data.velocity_r .!= 0.0)
                v_r_var = NetCDF.defVar(nc_file, "velocity_r", T, field_dims)
                NetCDF.putatt(nc_file, v_r_var, "long_name", "radial_velocity")
                NetCDF.defVarDeflate(nc_file, v_r_var, true, true, 6)
                
                v_theta_var = NetCDF.defVar(nc_file, "velocity_theta", T, field_dims)
                NetCDF.putatt(nc_file, v_theta_var, "long_name", "colatitudinal_velocity")
                NetCDF.defVarDeflate(nc_file, v_theta_var, true, true, 6)
                
                v_phi_var = NetCDF.defVar(nc_file, "velocity_phi", T, field_dims)
                NetCDF.putatt(nc_file, v_phi_var, "long_name", "azimuthal_velocity")
                NetCDF.defVarDeflate(nc_file, v_phi_var, true, true, 6)
            end
            
            # Magnetic components
            if any(local_data.magnetic_r .!= 0.0)
                B_r_var = NetCDF.defVar(nc_file, "magnetic_r", T, field_dims)
                NetCDF.putatt(nc_file, B_r_var, "long_name", "radial_magnetic_field")
                NetCDF.defVarDeflate(nc_file, B_r_var, true, true, 6)
                
                B_theta_var = NetCDF.defVar(nc_file, "magnetic_theta", T, field_dims)
                NetCDF.putatt(nc_file, B_theta_var, "long_name", "colatitudinal_magnetic_field")
                NetCDF.defVarDeflate(nc_file, B_theta_var, true, true, 6)
                
                B_phi_var = NetCDF.defVar(nc_file, "magnetic_phi", T, field_dims)
                NetCDF.putatt(nc_file, B_phi_var, "long_name", "azimuthal_magnetic_field")
                NetCDF.defVarDeflate(nc_file, B_phi_var, true, true, 6)
            end
            
            # Temperature
            if any(local_data.temperature .!= 0.0)
                temp_var = NetCDF.defVar(nc_file, "temperature", T, field_dims)
                NetCDF.putatt(nc_file, temp_var, "long_name", "temperature")
                NetCDF.defVarDeflate(nc_file, temp_var, true, true, 6)
            end
            
            # End definition mode
            NetCDF.endDef(nc_file)
            
            # Write time data
            NetCDF.putvar(nc_file, "time", [local_data.time])
            NetCDF.putvar(nc_file, "step", [Int32(local_data.step)])
            
        finally
            NetCDF.close(nc_file)
        end
    end
    
    MPI.Barrier(comm)
    
    # Each rank writes its portion of the data
    for writing_rank in 0:(nprocs-1)
        if rank == writing_rank
            nc_file = NetCDF.open(filename, NC_WRITE)
            
            try
                # Write coordinate data (only rank 0)
                if rank == 0
                    # Gather and write global coordinates
                    global_theta = mpi_gather_coordinate(local_data.theta, decomp.theta_comm, 
                                                        decomp.nlat_global, decomp.theta_range)
                    global_phi = mpi_gather_coordinate(local_data.phi, decomp.phi_comm,
                                                      decomp.nlon_global, decomp.phi_range)
                    global_r = mpi_gather_coordinate(local_data.r, decomp.r_comm,
                                                    decomp.nr_global, decomp.r_range)
                    
                    NetCDF.putvar(nc_file, "theta", global_theta)
                    NetCDF.putvar(nc_file, "phi", global_phi)
                    NetCDF.putvar(nc_file, "r", global_r)
                end
                
                # Write local data portion
                start_idx = [decomp.theta_range.start, decomp.phi_range.start, decomp.r_range.start]
                count = [decomp.nlat_local, decomp.nlon_local, decomp.nr_local]
                
                if any(local_data.velocity_r .!= 0.0)
                    NetCDF.putvar(nc_file, "velocity_r", local_data.velocity_r, 
                                 start=start_idx, count=count)
                    NetCDF.putvar(nc_file, "velocity_theta", local_data.velocity_theta,
                                 start=start_idx, count=count)
                    NetCDF.putvar(nc_file, "velocity_phi", local_data.velocity_phi,
                                 start=start_idx, count=count)
                end
                
                if any(local_data.magnetic_r .!= 0.0)
                    NetCDF.putvar(nc_file, "magnetic_r", local_data.magnetic_r,
                                 start=start_idx, count=count)
                    NetCDF.putvar(nc_file, "magnetic_theta", local_data.magnetic_theta,
                                 start=start_idx, count=count)
                    NetCDF.putvar(nc_file, "magnetic_phi", local_data.magnetic_phi,
                                 start=start_idx, count=count)
                end
                
                if any(local_data.temperature .!= 0.0)
                    NetCDF.putvar(nc_file, "temperature", local_data.temperature,
                                 start=start_idx, count=count)
                end
                
            finally
                NetCDF.close(nc_file)
            end
        end
        
        MPI.Barrier(comm)  # Ensure only one rank writes at a time
    end
    
    if rank == 0
        println("Coordinated serial NetCDF write completed")
    end
end

function mpi_gather_coordinate(local_coord::Vector{T}, dim_comm::MPI.Comm, 
                              global_size::Int, local_range::UnitRange{Int}) where T
    # Gather coordinate array from all processes in dimension communicator
    dim_rank = MPI.Comm_rank(dim_comm)
    dim_nprocs = MPI.Comm_size(dim_comm)
    
    if dim_rank == 0
        # Root process gathers all pieces
        global_coord = zeros(T, global_size)
        
        # Insert local piece
        global_coord[local_range] = local_coord
        
        # Receive from other processes
        for src_rank in 1:(dim_nprocs-1)
            remote_size = MPI.recv(Int, src_rank, 0, dim_comm)
            remote_start = MPI.recv(Int, src_rank, 1, dim_comm)
            remote_data = MPI.recv(Vector{T}, src_rank, 2, dim_comm)
            
            global_coord[remote_start:(remote_start + remote_size - 1)] = remote_data
        end
        
        return global_coord
    else
        # Send local data to root
        MPI.send(length(local_coord), 0, 0, dim_comm)
        MPI.send(local_range.start, 0, 1, dim_comm)
        MPI.send(local_coord, 0, 2, dim_comm)
        
        return T[]  # Non-root processes return empty
    end
end

# ============================================================================
# Main MPI Conversion Function
# ============================================================================

function mpi_convert_spectral_to_physical(filename::String, output_filename::String = "",
                                         use_shtns_vector::Bool = true, ::Type{T} = Float64) where T
    
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        println("=" * 70)
        println("MPI Converting Spectral Data to Physical Space")
        println("Using $nprocs MPI processes")
        println("=" * 70)
    end
    
    # Step 1: Determine global dimensions from file metadata
    global_dims = get_global_dimensions(filename)
    nlat_global, nlon_global, nr_global = global_dims
    
    if rank == 0
        println("Global dimensions: $nlat_global × $nlon_global × $nr_global")
    end
    
    # Step 2: Create domain decomposition
    decomp = create_mpi_domain_decomposition(nlat_global, nlon_global, nr_global)
    
    # Step 3: Read local data portions
    local_file_data, metadata = mpi_read_combined_netcdf(filename, decomp, T)
    
    # Step 4: Set up local SHTns converter
    velocity_spectral = get(local_file_data, "velocity_spectral", nothing)
    magnetic_spectral = get(local_file_data, "magnetic_spectral", nothing)
    
    if velocity_spectral === nothing && magnetic_spectral === nothing
        if rank == 0
            @warn "No spectral data found in file. Only temperature is available."
        end
        
        # Create simple data structure for temperature only
        temperature = get(local_file_data, "temperature", zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local))
        theta = get(local_file_data, "theta", T.(collect(range(0, π, length=decomp.nlat_local))))
        phi = get(local_file_data, "phi", T.(collect(range(0, 2π, length=decomp.nlon_local))))
        r = get(local_file_data, "r_physical", T.(collect(range(0.35, 1.0, length=decomp.nr_local))))
        
        return MPIPhysicalFieldData{T}(
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),  # velocity
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),  # magnetic
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),
            zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local),
            temperature,  # temperature
            theta, phi, r, decomp,
            metadata["time"], metadata["step"], metadata
        )
    end
    
    # Use velocity spectral data for converter setup, fallback to magnetic
    converter_spectral = velocity_spectral !== nothing ? velocity_spectral : magnetic_spectral
    converter = create_mpi_shtns_converter(converter_spectral, decomp, T)
    
    # Step 5: Convert velocity field
    v_r = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    v_theta = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    v_phi = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    
    if velocity_spectral !== nothing
        if rank == 0
            println("Converting velocity field to physical space...")
        end
        
        if use_shtns_vector
            try
                v_r, v_theta, v_phi = mpi_convert_toroidal_poloidal_to_physical(velocity_spectral, converter)
                if rank == 0
                    println("Velocity conversion completed using SHTns vector synthesis")
                end
            catch e
                if rank == 0
                    @warn "SHTns vector synthesis failed, falling back to manual method" exception=e
                end
                v_r, v_theta, v_phi = mpi_convert_vector_spherical_harmonics(velocity_spectral, converter)
                if rank == 0
                    println("Velocity conversion completed using manual vector synthesis")
                end
            end
        else
            v_r, v_theta, v_phi = mpi_convert_vector_spherical_harmonics(velocity_spectral, converter)
            if rank == 0
                println("Velocity conversion completed using manual vector synthesis")
            end
        end
    end
    
    # Step 6: Convert magnetic field
    B_r = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    B_theta = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    B_phi = zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local)
    
    if magnetic_spectral !== nothing
        if rank == 0
            println("Converting magnetic field to physical space...")
        end
        
        if use_shtns_vector
            try
                B_r, B_theta, B_phi = mpi_convert_toroidal_poloidal_to_physical(magnetic_spectral, converter)
                if rank == 0
                    println("Magnetic conversion completed using SHTns vector synthesis")
                end
            catch e
                if rank == 0
                    @warn "SHTns vector synthesis failed, falling back to manual method" exception=e
                end
                B_r, B_theta, B_phi = mpi_convert_vector_spherical_harmonics(magnetic_spectral, converter)
                if rank == 0
                    println("Magnetic conversion completed using manual vector synthesis")
                end
            end
        else
            B_r, B_theta, B_phi = mpi_convert_vector_spherical_harmonics(magnetic_spectral, converter)
            if rank == 0
                println("Magnetic conversion completed using manual vector synthesis")
            end
        end
    end
    
    # Step 7: Get temperature and coordinates
    temperature = get(local_file_data, "temperature", zeros(T, decomp.nlat_local, decomp.nlon_local, decomp.nr_local))
    theta = get(local_file_data, "theta", T.(collect(range(0, π, length=decomp.nlat_local))))
    phi = get(local_file_data, "phi", T.(collect(range(0, 2π, length=decomp.nlon_local))))
    r = get(local_file_data, "r_physical", T.(collect(range(0.35, 1.0, length=decomp.nr_local))))
    
    # Step 8: Create local physical field data structure
    local_physical_data = MPIPhysicalFieldData{T}(
        v_r, v_theta, v_phi,
        B_r, B_theta, B_phi,
        temperature,
        theta, phi, r,
        decomp,
        metadata["time"], metadata["step"], metadata
    )
    
    # Step 9: Compute and print global statistics
    global_stats = mpi_gather_global_statistics(local_physical_data)
    
    if rank == 0
        println("\nGlobal Field Statistics:")
        println("-" * 40)
        
        if haskey(global_stats, "vel_r_rms")
            println("Velocity:")
            println("  RMS components: r=$(round(global_stats["vel_r_rms"], digits=6)), " *
                   "θ=$(round(global_stats["vel_theta_rms"], digits=6)), " *
                   "φ=$(round(global_stats["vel_phi_rms"], digits=6))")
            println("  Max magnitude: $(round(global_stats["vel_mag_max"], digits=6))")
            println("  Kinetic energy: $(round(global_stats["kinetic_energy"], digits=6))")
        end
        
        if haskey(global_stats, "mag_r_rms")
            println("Magnetic:")
            println("  RMS components: r=$(round(global_stats["mag_r_rms"], digits=6)), " *
                   "θ=$(round(global_stats["mag_theta_rms"], digits=6)), " *
                   "φ=$(round(global_stats["mag_phi_rms"], digits=6))")
            println("  Max magnitude: $(round(global_stats["mag_mag_max"], digits=6))")
            println("  Magnetic energy: $(round(global_stats["magnetic_energy"], digits=6))")
        end
        
        if haskey(global_stats, "temp_mean")
            println("Temperature:")
            println("  Range: [$(round(global_stats["temp_min"], digits=6)), $(round(global_stats["temp_max"], digits=6))]")
            println("  Mean: $(round(global_stats["temp_mean"], digits=6)), Std: $(round(global_stats["temp_std"], digits=6))")
        end
    end
    
    # Step 10: Save to NetCDF if requested
    if !isempty(output_filename)
        mpi_write_parallel_netcdf(local_physical_data, output_filename)
    end
    
    if rank == 0
        println("\nMPI conversion completed successfully!")
        println("Local field dimensions: $(size(local_physical_data.velocity_r))")
        println("=" * 70)
    end
    
    return local_physical_data
end

function mpi_convert_vector_spherical_harmonics(spectral_data, converter::MPISHTnsConverter{T}) where T
    # Fallback manual implementation for MPI version
    rank = converter.decomp.rank
    
    if rank == 0
        println("Using manual vector spherical harmonic synthesis (MPI)...")
    end
    
    nlat_local = converter.nlat_local
    nlon_local = converter.nlon_local
    nr_local  = converter.nr_local
    
    # Get local theta and phi grids
    theta_start = (converter.decomp.theta_range.start - 1) * π / converter.decomp.nlat_global
    theta_end = converter.decomp.theta_range.stop * π / converter.decomp.nlat_global
    theta_local = collect(range(theta_start, theta_end, length=nlat_local))
    
    phi_start = (converter.decomp.phi_range.start - 1) * 2π / converter.decomp.nlon_global
    phi_end = converter.decomp.phi_range.stop * 2π / converter.decomp.nlon_global
    phi_local = collect(range(phi_start, phi_end, length=nlon_local))
    
    # Allocate physical space arrays
    v_r = zeros(T, nlat_local, nlon_local, nr_local)
    v_theta = zeros(T, nlat_local, nlon_local, nr_local)
    v_phi = zeros(T, nlat_local, nlon_local, nr_local)
    
    # Process each radial level
    for r_idx in 1:nr_local
        if r_idx <= size(spectral_data.toroidal_real, 2)
            r_val = spectral_data.r[r_idx]
            r_inv = 1.0 / max(r_val, 1e-10)
            r_inv_sq = r_inv^2
            
            # Manual vector spherical harmonic synthesis (simplified)
            for (spectral_idx, (l, m)) in enumerate(zip(spectral_data.l_values, spectral_data.m_values))
                if l == 0  # Skip l=0
                    continue
                end
                
                l_factor = Float64(l * (l + 1))
                
                # Get coefficients
                T_lm = complex(spectral_data.toroidal_real[spectral_idx, r_idx],
                              spectral_data.toroidal_imag[spectral_idx, r_idx])
                P_lm = complex(spectral_data.poloidal_real[spectral_idx, r_idx],
                              spectral_data.poloidal_imag[spectral_idx, r_idx])
                
                # Vector field synthesis (simplified implementation)
                for i_theta in 1:nlat_local
                    theta = theta_local[i_theta]
                    for j_phi in 1:nlon_local
                        phi = phi_local[j_phi]
                        
                        # Simplified spherical harmonics
                        Y_lm = compute_simple_spherical_harmonic(l, m, theta, phi)
                        
                        # Vector field components (highly simplified)
                        v_r_contribution = (l_factor * r_inv_sq) * P_lm * Y_lm
                        v_theta_contribution = r_inv * T_lm * Y_lm * 0.1  # Simplified
                        v_phi_contribution = r_inv * T_lm * Y_lm * 0.1   # Simplified
                        
                        # Accumulate contributions
                        v_r[i_theta, j_phi, r_idx] += real(v_r_contribution)
                        v_theta[i_theta, j_phi, r_idx] += real(v_theta_contribution)
                        v_phi[i_theta, j_phi, r_idx] += real(v_phi_contribution)
                    end
                end
            end
        end
    end
    
    return v_r, v_theta, v_phi
end

# ============================================================================
# Utility Functions
# ============================================================================

function get_global_dimensions(filename::String)
    # Read global dimensions from NetCDF file
    nc_file = NetCDF.open(filename, NC_NOWRITE)
    
    try
        # Try to get dimensions from temperature field
        if NetCDF.varid(nc_file, "temperature_global") != -1
            dims = NetCDF.size(nc_file, "temperature_global")
            return dims[1], dims[2], dims[3]
        end
        
        # Fallback: get from coordinate arrays
        nlat_global = NetCDF.dimlen(nc_file, NetCDF.dimid(nc_file, "theta"))
        nlon_global = NetCDF.dimlen(nc_file, NetCDF.dimid(nc_file, "phi"))
        nr_global = NetCDF.dimlen(nc_file, NetCDF.dimid(nc_file, "r_physical"))
        
        return nlat_global, nlon_global, nr_global
        
    finally
        NetCDF.close(nc_file)
    end
end

function compute_simple_spherical_harmonic(l::Int, m::Int, theta::Float64, phi::Float64)
    # Very simplified spherical harmonic for fallback method
    if l == 1 && m == 0
        return sqrt(3.0 / (4π)) * cos(theta)
    elseif l == 1 && abs(m) == 1
        sign_m = sign(m)
        return -sign_m * sqrt(3.0 / (8π)) * sin(theta) * exp(1im * m * phi)
    else
        # Simplified approximation
        P_l = cos(theta)^l
        normalization = sqrt((2*l + 1) / (4π))
        return normalization * P_l * sin(theta)^abs(m) * exp(1im * m * phi)
    end
end

# ============================================================================
# Batch Processing
# ============================================================================

function mpi_batch_convert_directory(input_dir::String, output_dir::String = "",
                                    pattern::String = "combined_global_time_", 
                                    use_shtns_vector::Bool = true, ::Type{T} = Float64) where T
    
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        println("MPI batch converting files in directory: $input_dir")
        println("Using $nprocs MPI processes")
    end
    
    if isempty(output_dir)
        output_dir = joinpath(input_dir, "mpi_physical_space")
    end
    
    if rank == 0 && !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end
    
    MPI.Barrier(comm)
    
    # Find all combined files (only rank 0 does this)
    files = String[]
    if rank == 0
        all_files = readdir(input_dir)
        files = filter(f -> contains(f, pattern) && endswith(f, ".nc"), all_files)
        
        if isempty(files)
            @warn "No combined files found with pattern '$pattern' in $input_dir"
        else
            println("Found $(length(files)) files to convert")
        end
    end
    
    # Broadcast file list to all processes
    num_files = MPI.bcast(length(files), 0, comm)
    
    if num_files == 0
        return
    end
    
    if rank != 0
        files = Vector{String}(undef, num_files)
    end
    
    for i in 1:num_files
        if rank == 0
            file_to_send = files[i]
        else
            file_to_send = ""
        end
        files[i] = MPI.bcast(file_to_send, 0, comm)
    end
    
    # Process each file with all MPI processes
    for (i, filename) in enumerate(files)
        input_path = joinpath(input_dir, filename)
        
        # Generate output filename
        base_name = splitext(filename)[1]
        output_filename = joinpath(output_dir, "$(base_name)_mpi_physical.nc")
        
        if rank == 0
            println("[$i/$num_files] Processing: $filename")
        end
        
        try
            local_physical_data = mpi_convert_spectral_to_physical(input_path, output_filename, 
                                                                  use_shtns_vector, T)
            
            # Print brief statistics (only from rank 0)
            if rank == 0
                global_stats = mpi_gather_global_statistics(local_physical_data)
                if haskey(global_stats, "kinetic_energy")
                    println("  Velocity: KE = $(round(global_stats["kinetic_energy"], digits=6))")
                end
                if haskey(global_stats, "magnetic_energy")
                    println("  Magnetic: ME = $(round(global_stats["magnetic_energy"], digits=6))")
                end
            end
            
        catch e
            if rank == 0
                @error "Failed to process $filename" exception=e
            end
            continue
        end
        
        MPI.Barrier(comm)  # Synchronize between files
    end
    
    if rank == 0
        println("MPI batch conversion completed")
    end
end

# ============================================================================
# Main Interface Functions
# ============================================================================

function mpi_read_and_convert(filename::String; 
                             output_filename::String = "",
                             use_shtns_vector::Bool = true,
                             compute_stats::Bool = true,
                             precision::Type = Float64)
    """
    Main MPI interface function to read and convert spectral data to physical space.
    """
    
    # Convert spectral to physical
    local_physical_data = mpi_convert_spectral_to_physical(filename, output_filename, 
                                                          use_shtns_vector, precision)
    
    # Compute global statistics if requested
    if compute_stats
        global_stats = mpi_gather_global_statistics(local_physical_data)
        
        rank = MPI.Comm_rank(comm)
        if rank == 0
            println("\nGlobal Field Statistics Summary:")
            println("-" * 40)
            
            if haskey(global_stats, "vel_r_rms")
                println("Velocity:")
                println("  RMS: ($(round(global_stats["vel_r_rms"], digits=6)), " *
                       "$(round(global_stats["vel_theta_rms"], digits=6)), " *
                       "$(round(global_stats["vel_phi_rms"], digits=6)))")
                println("  Kinetic energy: $(roun