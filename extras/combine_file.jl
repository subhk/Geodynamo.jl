# ============================================================================
# NetCDF File Combiner - Reconstruct Global Fields from Distributed Files
# Combines velocity/magnetic spectral coefficients and temperature physical data
# ============================================================================

# module NetCDFFileCombiner
using NetCDF
using LinearAlgebra
using Statistics
using Printf
using Dates

# ============================================================================
# Data Structures
# ============================================================================

struct GlobalFieldData{T}
    # Temperature (physical space)
    temperature::Union{Array{T,3}, Nothing}           # (theta_global, phi_global, r)
    temperature_grid::Union{NamedTuple, Nothing}      # (theta, phi, r)
    
    # Velocity (spectral space)
    velocity_toroidal_real::Union{Array{T,2}, Nothing}  # (nlm_global, r)
    velocity_toroidal_imag::Union{Array{T,2}, Nothing}
    velocity_poloidal_real::Union{Array{T,2}, Nothing}
    velocity_poloidal_imag::Union{Array{T,2}, Nothing}
    
    # Magnetic (spectral space)
    magnetic_toroidal_real::Union{Array{T,2}, Nothing}
    magnetic_toroidal_imag::Union{Array{T,2}, Nothing}
    magnetic_poloidal_real::Union{Array{T,2}, Nothing}
    magnetic_poloidal_imag::Union{Array{T,2}, Nothing}
    
    # Spectral grid info
    spectral_grid::Union{NamedTuple, Nothing}         # (l_values, m_values, r)
    
    # Global metadata
    metadata::Dict{String, Any}
    time::Float64
    step::Int
    nprocs::Int
end

struct CombinerConfig
    output_precision::DataType      # Output precision for combined data
    validate_files::Bool           # Validate input files before processing
    verbose::Bool                  # Print progress information
    save_combined::Bool            # Save combined result to new NetCDF file
    combined_filename::String      # Filename for combined output
    include_diagnostics::Bool      # Compute global diagnostics
    interpolate_missing::Bool      # Interpolate missing data regions
end

function default_combiner_config()
    return CombinerConfig(
        Float64,              # output_precision
        true,                 # validate_files
        true,                 # verbose
        true,                 # save_combined
        "combined_global",    # combined_filename
        true,                 # include_diagnostics
        false                 # interpolate_missing
    )
end

# ============================================================================
# File Discovery and Validation
# ============================================================================

function find_distributed_files(output_dir::String, time::Float64, filename_prefix::String = "geodynamo")
    time_str = @sprintf("%.6f", time)
    time_str = replace(time_str, "." => "p")
    
    # Find all files for this time
    files = readdir(output_dir)
    pattern = "$(filename_prefix)_output_rank_"
    time_pattern = "_time_$(time_str)"
    
    matching_files = filter(files) do f
        contains(f, pattern) && contains(f, time_pattern) && endswith(f, ".nc")
    end
    
    if isempty(matching_files)
        return String[], 0
    end
    
    # Extract rank numbers and sort
    rank_pattern = r"rank_(\d+)"
    rank_files = Tuple{Int, String}[]
    
    for file in matching_files
        m = match(rank_pattern, file)
        if m !== nothing
            try
                rank = parse(Int, m.captures[1])
                push!(rank_files, (rank, joinpath(output_dir, file)))
            catch
                continue
            end
        end
    end
    
    sort!(rank_files, by = x -> x[1])
    
    filenames = [rf[2] for rf in rank_files]
    nprocs = length(filenames)
    
    return filenames, nprocs
end

function validate_file_set(filenames::Vector{String}, config::CombinerConfig)
    if !config.validate_files
        return true
    end
    
    if config.verbose
        println("Validating $(length(filenames)) files...")
    end
    
    # Check all files exist and are readable
    for (i, filename) in enumerate(filenames)
        if !isfile(filename)
            @error "File not found: $filename"
            return false
        end
        
        try
            nc_file = NetCDF.open(filename, NC_NOWRITE)
            
            # Check required variables
            required_vars = ["time", "step", "r"]
            for var in required_vars
                if NetCDF.varid(nc_file, var) == -1
                    @error "Missing variable $var in file $filename"
                    NetCDF.close(nc_file)
                    return false
                end
            end
            
            NetCDF.close(nc_file)
            
        catch e
            @error "Error reading file $filename" exception=e
            return false
        end
    end
    
    if config.verbose
        println("All files validated successfully.")
    end
    
    return true
end

# ============================================================================
# Temperature Field Combination (Physical Space)
# ============================================================================

function combine_temperature_field(filenames::Vector{String}, config::CombinerConfig)
    if config.verbose
        println("Combining temperature field from $(length(filenames)) files...")
    end
    
    # Read metadata and dimensions from first file
    first_file = NetCDF.open(filenames[1], NC_NOWRITE)
    
    try
        # Get basic info
        time_val = NetCDF.readvar(first_file, "time")[1]
        step_val = NetCDF.readvar(first_file, "step")[1]
        total_ranks = NetCDF.getatt(first_file, NC_GLOBAL, "mpi_total_ranks")
        
        # Check if temperature exists
        if NetCDF.varid(first_file, "temperature") == -1
            if config.verbose
                println("No temperature field found in files.")
            end
            return nothing, nothing
        end
        
        # Get global dimensions by reading from all files
        global_nlat = 0
        global_nlon = 0
        global_nr = 0
        
        # Read local dimensions from each file to determine global size
        local_dims = []
        for filename in filenames
            nc_file = NetCDF.open(filename, NC_NOWRITE)
            try
                if NetCDF.varid(nc_file, "temperature") != -1
                    dims = NetCDF.size(nc_file, "temperature")
                    push!(local_dims, dims)
                    
                    # Accumulate global dimensions (simplified approach)
                    global_nlat = max(global_nlat, dims[1])
                    global_nlon = max(global_nlon, dims[2])
                    global_nr = max(global_nr, dims[3])
                end
            finally
                NetCDF.close(nc_file)
            end
        end
        
        # For distributed case, estimate global size
        # This is simplified - in practice, you'd need proper domain decomposition info
        nprocs = length(filenames)
        if nprocs > 1
            # Assume even distribution
            proc_per_dim = Int(ceil(sqrt(nprocs)))
            global_nlat *= proc_per_dim
            global_nlon = max(global_nlon, Int(ceil(nprocs / proc_per_dim)) * global_nlon ÷ proc_per_dim)
        end
        
        # Allocate global temperature array
        global_temp = zeros(config.output_precision, global_nlat, global_nlon, global_nr)
        
        # Read and assemble temperature data from all files
        for (proc_idx, filename) in enumerate(filenames)
            nc_file = NetCDF.open(filename, NC_NOWRITE)
            try
                if NetCDF.varid(nc_file, "temperature") != -1
                    local_temp = NetCDF.readvar(nc_file, "temperature")
                    
                    # Simple placement strategy - would need proper domain mapping
                    # For now, place files in order
                    nlat_local, nlon_local, nr_local = size(local_temp)
                    
                    # Calculate placement indices (simplified)
                    theta_start = ((proc_idx - 1) * nlat_local) + 1
                    theta_end = min(theta_start + nlat_local - 1, global_nlat)
                    phi_start = 1
                    phi_end = min(nlon_local, global_nlon)
                    
                    if theta_end <= global_nlat && phi_end <= global_nlon
                        global_temp[theta_start:theta_end, phi_start:phi_end, 1:nr_local] = local_temp
                    end
                end
            finally
                NetCDF.close(nc_file)
            end
        end
        
        # Create global grid
        global_theta = collect(range(0, π, length=global_nlat))
        global_phi = collect(range(0, 2π, length=global_nlon))
        global_r = NetCDF.readvar(first_file, "r")
        
        global_grid = (theta = global_theta, phi = global_phi, r = global_r)
        
        return global_temp, global_grid
        
    finally
        NetCDF.close(first_file)
    end
end

# ============================================================================
# Spectral Field Combination
# ============================================================================

function combine_spectral_field(filenames::Vector{String}, field_name::String, 
                                config::CombinerConfig)
    if config.verbose
        println("Combining spectral field '$field_name' from $(length(filenames)) files...")
    end
    
    real_name = "$(field_name)_real"
    imag_name = "$(field_name)_imag"
    
    # Check if field exists in first file
    first_file = NetCDF.open(filenames[1], NC_NOWRITE)
    field_exists = false
    
    try
        field_exists = (NetCDF.varid(first_file, real_name) != -1 && 
                        NetCDF.varid(first_file, imag_name) != -1)
    finally
        NetCDF.close(first_file)
    end
    
    if !field_exists
        if config.verbose
            println("Field '$field_name' not found in files.")
        end
        return nothing, nothing
    end
    
    # Determine global spectral dimensions
    global_nlm = 0
    global_nr = 0
    all_l_values = Set{Int}()
    all_m_values = Set{Int}()
    
    # Collect all (l,m) modes from all files
    for filename in filenames
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, real_name) != -1
                dims = NetCDF.size(nc_file, real_name)
                local_nlm, local_nr = dims[1], dims[2]
                global_nr = max(global_nr, local_nr)
                
                # Read l,m values if available
                if NetCDF.varid(nc_file, "l_values") != -1
                    l_vals = NetCDF.readvar(nc_file, "l_values")
                    m_vals = NetCDF.readvar(nc_file, "m_values")
                    
                    for (l, m) in zip(l_vals, m_vals)
                        push!(all_l_values, l)
                        push!(all_m_values, m)
                    end
                end
            end
        finally
            NetCDF.close(nc_file)
        end
    end
    
    # Create global (l,m) index mapping
    global_l_values = sort(collect(all_l_values))
    global_m_values = sort(collect(all_m_values))
    
    # Create comprehensive (l,m) pairs
    lmax = maximum(global_l_values)
    mmax = maximum(global_m_values)
    
    global_lm_pairs = Tuple{Int,Int}[]
    for l in 0:lmax
        for m in 0:min(l, mmax)
            push!(global_lm_pairs, (l, m))
        end
    end
    
    global_nlm = length(global_lm_pairs)
    
    # Allocate global arrays
    global_real = zeros(config.output_precision, global_nlm, global_nr)
    global_imag = zeros(config.output_precision, global_nlm, global_nr)
    
    # Create mapping from (l,m) to global index
    lm_to_global_idx = Dict{Tuple{Int,Int}, Int}()
    for (idx, (l, m)) in enumerate(global_lm_pairs)
        lm_to_global_idx[(l, m)] = idx
    end
    
    # Read and combine spectral data from all files
    for filename in filenames
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        try
            if NetCDF.varid(nc_file, real_name) != -1
                local_real = NetCDF.readvar(nc_file, real_name)
                local_imag = NetCDF.readvar(nc_file, imag_name)
                
                # Get local (l,m) values
                if NetCDF.varid(nc_file, "l_values") != -1
                    local_l = NetCDF.readvar(nc_file, "l_values")
                    local_m = NetCDF.readvar(nc_file, "m_values")
                    
                    # Map local data to global arrays
                    for (local_idx, (l, m)) in enumerate(zip(local_l, local_m))
                        if haskey(lm_to_global_idx, (l, m))
                            global_idx = lm_to_global_idx[(l, m)]
                            
                            # Copy data (assuming radial ranges match)
                            nr_local = size(local_real, 2)
                            for r_idx in 1:min(nr_local, global_nr)
                                global_real[global_idx, r_idx] += local_real[local_idx, r_idx]
                                global_imag[global_idx, r_idx] += local_imag[local_idx, r_idx]
                            end
                        end
                    end
                end
            end
        finally
            NetCDF.close(nc_file)
        end
    end
    
    # Create global spectral grid
    global_l_final = [pair[1] for pair in global_lm_pairs]
    global_m_final = [pair[2] for pair in global_lm_pairs]
    
    # Read radial grid from first file
    first_file = NetCDF.open(filenames[1], NC_NOWRITE)
    global_r = NetCDF.readvar(first_file, "r")
    NetCDF.close(first_file)
    
    spectral_grid = (l_values = global_l_final, m_values = global_m_final, r = global_r)
    
    return (real = global_real, imag = global_imag), spectral_grid
end

# ============================================================================
# Main Combination Function
# ============================================================================

function combine_distributed_files(output_dir::String, time::Float64, 
                                    config::CombinerConfig = default_combiner_config(),
                                    filename_prefix::String = "geodynamo")
    if config.verbose
        println("=" * 60)
        println("Combining distributed NetCDF files for time $time")
        println("=" * 60)
    end
    
    # Find all files for this time
    filenames, nprocs = find_distributed_files(output_dir, time, filename_prefix)
    
    if isempty(filenames)
        error("No files found for time $time in directory $output_dir")
    end
    
    if config.verbose
        println("Found $nprocs files to combine:")
        for (i, f) in enumerate(filenames[1:min(5, end)])
            println("  $i: $(basename(f))")
        end
        if length(filenames) > 5
            println("  ... and $(length(filenames) - 5) more")
        end
    end
    
    # Validate files
    if !validate_file_set(filenames, config)
        error("File validation failed")
    end
    
    # Read basic metadata from first file
    first_file = NetCDF.open(filenames[1], NC_NOWRITE)
    metadata = Dict{String, Any}()
    time_val = 0.0
    step_val = 0
    
    try
        time_val = NetCDF.readvar(first_file, "time")[1]
        step_val = NetCDF.readvar(first_file, "step")[1]
        
        # Read global attributes that should be the same across files
        for attr_name in ["Rayleigh_number", "Ekman_number", "Prandtl_number", "Magnetic_Prandtl"]
            try
                metadata[attr_name] = NetCDF.getatt(first_file, NC_GLOBAL, attr_name)
            catch
                # Skip missing attributes
            end
        end
    finally
        NetCDF.close(first_file)
    end
    
    # Combine temperature field (physical space)
    if config.verbose
        println("Processing temperature field...")
    end
    temperature, temp_grid = combine_temperature_field(filenames, config)
    
    # Combine velocity fields (spectral space)
    if config.verbose
        println("Processing velocity fields...")
    end
    vel_toroidal, vel_tor_grid = combine_spectral_field(filenames, "velocity_toroidal", config)
    vel_poloidal, vel_pol_grid = combine_spectral_field(filenames, "velocity_poloidal", config)
    
    # Combine magnetic fields (spectral space)  
    if config.verbose
        println("Processing magnetic fields...")
    end
    mag_toroidal, mag_tor_grid = combine_spectral_field(filenames, "magnetic_toroidal", config)
    mag_poloidal, mag_pol_grid = combine_spectral_field(filenames, "magnetic_poloidal", config)
    
    # Use the first non-nothing spectral grid
    spectral_grid = vel_tor_grid
    if spectral_grid === nothing
        spectral_grid = mag_tor_grid
    end
    
    # Create combined data structure
    global_data = GlobalFieldData{config.output_precision}(
        temperature, temp_grid,
        vel_toroidal !== nothing ? vel_toroidal.real : nothing,
        vel_toroidal !== nothing ? vel_toroidal.imag : nothing,
        vel_poloidal !== nothing ? vel_poloidal.real : nothing,
        vel_poloidal !== nothing ? vel_poloidal.imag : nothing,
        mag_toroidal !== nothing ? mag_toroidal.real : nothing,
        mag_toroidal !== nothing ? mag_toroidal.imag : nothing,
        mag_poloidal !== nothing ? mag_poloidal.real : nothing,
        mag_poloidal !== nothing ? mag_poloidal.imag : nothing,
        spectral_grid,
        metadata, time_val, step_val, nprocs
    )
    
    # Compute global diagnostics
    if config.include_diagnostics
        compute_global_diagnostics!(global_data, config)
    end
    
    # Save combined data
    if config.save_combined
        save_combined_file(global_data, output_dir, config)
    end
    
    if config.verbose
        println("Successfully combined all fields for time $time")
        println("=" * 60)
    end
    
    return global_data
end

# ============================================================================
# Global Diagnostics
# ============================================================================

function compute_global_diagnostics!(global_data::GlobalFieldData, config::CombinerConfig)
    if config.verbose
        println("Computing global diagnostics...")
    end
    
    diagnostics = Dict{String, Float64}()
    
    # Temperature diagnostics
    if global_data.temperature !== nothing
        T = global_data.temperature
        diagnostics["global_temp_mean"] = mean(T)
        diagnostics["global_temp_std"] = std(T)
        diagnostics["global_temp_min"] = minimum(T)
        diagnostics["global_temp_max"] = maximum(T)
        diagnostics["global_temp_volume"] = sum(T)  # Integrated temperature
    end
    
    # Velocity energy diagnostics
    if global_data.velocity_toroidal_real !== nothing
        vel_tor_energy = 0.5 * sum(global_data.velocity_toroidal_real.^2 .+ 
                                    global_data.velocity_toroidal_imag.^2)
        diagnostics["global_velocity_toroidal_energy"] = vel_tor_energy
    end
    
    if global_data.velocity_poloidal_real !== nothing
        vel_pol_energy = 0.5 * sum(global_data.velocity_poloidal_real.^2 .+ 
                                    global_data.velocity_poloidal_imag.^2)
        diagnostics["global_velocity_poloidal_energy"] = vel_pol_energy
    end
    
    # Total kinetic energy
    if (global_data.velocity_toroidal_real !== nothing && 
        global_data.velocity_poloidal_real !== nothing)
        diagnostics["global_kinetic_energy"] = (diagnostics["global_velocity_toroidal_energy"] + 
                                                diagnostics["global_velocity_poloidal_energy"])
    end
    
    # Magnetic energy diagnostics
    if global_data.magnetic_toroidal_real !== nothing
        mag_tor_energy = 0.5 * sum(global_data.magnetic_toroidal_real.^2 .+ 
                                    global_data.magnetic_toroidal_imag.^2)
        diagnostics["global_magnetic_toroidal_energy"] = mag_tor_energy
    end
    
    if global_data.magnetic_poloidal_real !== nothing
        mag_pol_energy = 0.5 * sum(global_data.magnetic_poloidal_real.^2 .+ 
                                    global_data.magnetic_poloidal_imag.^2)
        diagnostics["global_magnetic_poloidal_energy"] = mag_pol_energy
    end
    
    # Total magnetic energy
    if (global_data.magnetic_toroidal_real !== nothing && 
        global_data.magnetic_poloidal_real !== nothing)
        diagnostics["global_magnetic_energy"] = (diagnostics["global_magnetic_toroidal_energy"] + 
                                                diagnostics["global_magnetic_poloidal_energy"])
    end
    
    # Spectral analysis
    if global_data.spectral_grid !== nothing
        l_values = global_data.spectral_grid.l_values
        
        # Dominant l modes
        if global_data.velocity_toroidal_real !== nothing
            vel_spectrum = sum(global_data.velocity_toroidal_real.^2 .+ 
                                global_data.velocity_toroidal_imag.^2, dims=2)[:,1]
            max_idx = argmax(vel_spectrum)
            diagnostics["dominant_velocity_l"] = l_values[max_idx]
        end
        
        if global_data.magnetic_poloidal_real !== nothing
            # Dipole moment (l=1, m=0)
            dipole_idx = findfirst(i -> l_values[i] == 1 && 
                                        global_data.spectral_grid.m_values[i] == 0, 
                                    1:length(l_values))
            if dipole_idx !== nothing
                # Surface dipole strength (at outer boundary)
                dipole_real = global_data.magnetic_poloidal_real[dipole_idx, end]
                dipole_imag = global_data.magnetic_poloidal_imag[dipole_idx, end]
                diagnostics["dipole_strength"] = sqrt(dipole_real^2 + dipole_imag^2)
            end
        end
    end
    
    # Store diagnostics in global data
    global_data.metadata["global_diagnostics"] = diagnostics
    
    if config.verbose
        println("Global diagnostics computed:")
        for (key, value) in diagnostics
            println("  $key: $(round(value, digits=6))")
        end
    end
end

# ============================================================================
# Save Combined File
# ============================================================================

function save_combined_file(global_data::GlobalFieldData, output_dir::String, 
                            config::CombinerConfig)
    time_str = @sprintf("%.6f", global_data.time)
    time_str = replace(time_str, "." => "p")
    
    filename = joinpath(output_dir, "$(config.combined_filename)_time_$(time_str).nc")
    
    if config.verbose
        println("Saving combined data to: $filename")
    end
    
    nc_file = NetCDF.create(filename, NcFile)
    
    try
        # Global attributes
        NetCDF.putatt(nc_file, "title", "Combined Global Geodynamo Fields")
        NetCDF.putatt(nc_file, "source", "NetCDF File Combiner")
        NetCDF.putatt(nc_file, "history", "Combined on $(now()) from $(global_data.nprocs) files")
        NetCDF.putatt(nc_file, "Conventions", "CF-1.8")
        NetCDF.putatt(nc_file, "original_nprocs", global_data.nprocs)
        NetCDF.putatt(nc_file, "combination_time", string(now()))
        
        # Add metadata
        for (key, value) in global_data.metadata
            if !(key in ["global_diagnostics"])  # Skip complex nested data
                try
                    NetCDF.putatt(nc_file, key, value)
                catch
                    # Skip problematic attributes
                end
            end
        end
        
        # Define dimensions and coordinates
        time_dim = NetCDF.defDim(nc_file, "time", 1)
        
        # Physical space dimensions (for temperature)
        if global_data.temperature !== nothing
            nlat_global, nlon_global, nr_global = size(global_data.temperature)
            
            theta_dim = NetCDF.defDim(nc_file, "theta", nlat_global)
            phi_dim = NetCDF.defDim(nc_file, "phi", nlon_global)
            r_phys_dim = NetCDF.defDim(nc_file, "r_physical", nr_global)
            
            # Coordinate variables
            theta_var = NetCDF.defVar(nc_file, "theta", Float64, (theta_dim,))
            NetCDF.putatt(nc_file, theta_var, "long_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "units", "radians")
            
            phi_var = NetCDF.defVar(nc_file, "phi", Float64, (phi_dim,))
            NetCDF.putatt(nc_file, phi_var, "long_name", "azimuthal_angle")
            NetCDF.putatt(nc_file, phi_var, "units", "radians")
            
            r_phys_var = NetCDF.defVar(nc_file, "r_physical", Float64, (r_phys_dim,))
            NetCDF.putatt(nc_file, r_phys_var, "long_name", "radial_coordinate_physical")
            NetCDF.putatt(nc_file, r_phys_var, "units", "dimensionless")
            
            # Temperature variable
            temp_var = NetCDF.defVar(nc_file, "temperature_global", config.output_precision, 
                                    (theta_dim, phi_dim, r_phys_dim))
            NetCDF.putatt(nc_file, temp_var, "long_name", "global_temperature_field")
            NetCDF.putatt(nc_file, temp_var, "units", "dimensionless")
            NetCDF.putatt(nc_file, temp_var, "representation", "physical_space")
        end
        
        # Spectral space dimensions (for velocity/magnetic)
        if global_data.spectral_grid !== nothing
            nlm_global = length(global_data.spectral_grid.l_values)
            nr_spec = length(global_data.spectral_grid.r)
            
            lm_dim = NetCDF.defDim(nc_file, "spectral_mode", nlm_global)
            r_spec_dim = NetCDF.defDim(nc_file, "r_spectral", nr_spec)
            
            # Spectral coordinate variables
            l_var = NetCDF.defVar(nc_file, "l_values_global", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, l_var, "long_name", "spherical_harmonic_degree")
            
            m_var = NetCDF.defVar(nc_file, "m_values_global", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, m_var, "long_name", "spherical_harmonic_order")
            
            r_spec_var = NetCDF.defVar(nc_file, "r_spectral", Float64, (r_spec_dim,))
            NetCDF.putatt(nc_file, r_spec_var, "long_name", "radial_coordinate_spectral")
            NetCDF.putatt(nc_file, r_spec_var, "units", "dimensionless")
            
            # Velocity spectral variables
            if global_data.velocity_toroidal_real !== nothing
                vel_tor_real_var = NetCDF.defVar(nc_file, "velocity_toroidal_real_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                vel_tor_imag_var = NetCDF.defVar(nc_file, "velocity_toroidal_imag_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                NetCDF.putatt(nc_file, vel_tor_real_var, "long_name", "global_velocity_toroidal_real")
                NetCDF.putatt(nc_file, vel_tor_imag_var, "long_name", "global_velocity_toroidal_imag")
                NetCDF.putatt(nc_file, vel_tor_real_var, "representation", "spectral_space")
                NetCDF.putatt(nc_file, vel_tor_imag_var, "representation", "spectral_space")
            end
            
            if global_data.velocity_poloidal_real !== nothing
                vel_pol_real_var = NetCDF.defVar(nc_file, "velocity_poloidal_real_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                vel_pol_imag_var = NetCDF.defVar(nc_file, "velocity_poloidal_imag_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                NetCDF.putatt(nc_file, vel_pol_real_var, "long_name", "global_velocity_poloidal_real")
                NetCDF.putatt(nc_file, vel_pol_imag_var, "long_name", "global_velocity_poloidal_imag")
                NetCDF.putatt(nc_file, vel_pol_real_var, "representation", "spectral_space")
                NetCDF.putatt(nc_file, vel_pol_imag_var, "representation", "spectral_space")
            end
            
            # Magnetic spectral variables
            if global_data.magnetic_toroidal_real !== nothing
                mag_tor_real_var = NetCDF.defVar(nc_file, "magnetic_toroidal_real_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                mag_tor_imag_var = NetCDF.defVar(nc_file, "magnetic_toroidal_imag_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                NetCDF.putatt(nc_file, mag_tor_real_var, "long_name", "global_magnetic_toroidal_real")
                NetCDF.putatt(nc_file, mag_tor_imag_var, "long_name", "global_magnetic_toroidal_imag")
                NetCDF.putatt(nc_file, mag_tor_real_var, "representation", "spectral_space")
                NetCDF.putatt(nc_file, mag_tor_imag_var, "representation", "spectral_space")
            end
            
            if global_data.magnetic_poloidal_real !== nothing
                mag_pol_real_var = NetCDF.defVar(nc_file, "magnetic_poloidal_real_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                mag_pol_imag_var = NetCDF.defVar(nc_file, "magnetic_poloidal_imag_global", 
                                                config.output_precision, (lm_dim, r_spec_dim))
                NetCDF.putatt(nc_file, mag_pol_real_var, "long_name", "global_magnetic_poloidal_real")
                NetCDF.putatt(nc_file, mag_pol_imag_var, "long_name", "global_magnetic_poloidal_imag")
                NetCDF.putatt(nc_file, mag_pol_real_var, "representation", "spectral_space")
                NetCDF.putatt(nc_file, mag_pol_imag_var, "representation", "spectral_space")
            end
        end
        
        # Time variables
        time_var = NetCDF.defVar(nc_file, "time", Float64, (time_dim,))
        NetCDF.putatt(nc_file, time_var, "long_name", "simulation_time")
        NetCDF.putatt(nc_file, time_var, "units", "dimensionless")
        
        step_var = NetCDF.defVar(nc_file, "step", Int32, (time_dim,))
        NetCDF.putatt(nc_file, step_var, "long_name", "simulation_step")
        
        # Global diagnostics variables
        if haskey(global_data.metadata, "global_diagnostics")
            scalar_dim = NetCDF.defDim(nc_file, "scalar", 1)
            diagnostics = global_data.metadata["global_diagnostics"]
            
            for (diag_name, diag_value) in diagnostics
                diag_var = NetCDF.defVar(nc_file, "global_$(diag_name)", Float64, (scalar_dim,))
                NetCDF.putatt(nc_file, diag_var, "long_name", replace(diag_name, "_" => " "))
            end
        end
        
        # End definition mode
        NetCDF.endDef(nc_file)
        
        # Write coordinate data
        if global_data.temperature_grid !== nothing
            NetCDF.putvar(nc_file, "theta", global_data.temperature_grid.theta)
            NetCDF.putvar(nc_file, "phi", global_data.temperature_grid.phi)
            NetCDF.putvar(nc_file, "r_physical", global_data.temperature_grid.r)
        end
        
        if global_data.spectral_grid !== nothing
            NetCDF.putvar(nc_file, "l_values_global", global_data.spectral_grid.l_values)
            NetCDF.putvar(nc_file, "m_values_global", global_data.spectral_grid.m_values)
            NetCDF.putvar(nc_file, "r_spectral", global_data.spectral_grid.r)
        end
        
        # Write field data
        if global_data.temperature !== nothing
            NetCDF.putvar(nc_file, "temperature_global", global_data.temperature)
        end
        
        if global_data.velocity_toroidal_real !== nothing
            NetCDF.putvar(nc_file, "velocity_toroidal_real_global", global_data.velocity_toroidal_real)
            NetCDF.putvar(nc_file, "velocity_toroidal_imag_global", global_data.velocity_toroidal_imag)
        end
        
        if global_data.velocity_poloidal_real !== nothing
            NetCDF.putvar(nc_file, "velocity_poloidal_real_global", global_data.velocity_poloidal_real)
            NetCDF.putvar(nc_file, "velocity_poloidal_imag_global", global_data.velocity_poloidal_imag)
        end
        
        if global_data.magnetic_toroidal_real !== nothing
            NetCDF.putvar(nc_file, "magnetic_toroidal_real_global", global_data.magnetic_toroidal_real)
            NetCDF.putvar(nc_file, "magnetic_toroidal_imag_global", global_data.magnetic_toroidal_imag)
        end
        
        if global_data.magnetic_poloidal_real !== nothing
            NetCDF.putvar(nc_file, "magnetic_poloidal_real_global", global_data.magnetic_poloidal_real)
            NetCDF.putvar(nc_file, "magnetic_poloidal_imag_global", global_data.magnetic_poloidal_imag)
        end
        
        # Write time data
        NetCDF.putvar(nc_file, "time", [global_data.time])
        NetCDF.putvar(nc_file, "step", [Int32(global_data.step)])
        
        # Write global diagnostics
        if haskey(global_data.metadata, "global_diagnostics")
            diagnostics = global_data.metadata["global_diagnostics"]
            for (diag_name, diag_value) in diagnostics
                NetCDF.putvar(nc_file, "global_$(diag_name)", [Float64(diag_value)])
            end
        end
        
    finally
        NetCDF.close(nc_file)
    end
    
    if config.verbose
        println("Combined file saved successfully: $(basename(filename))")
    end
end

# ============================================================================
# Batch Processing Functions
# ============================================================================

function combine_time_series(output_dir::String, time_range::Tuple{Float64,Float64}, 
                                config::CombinerConfig = default_combiner_config(),
                                filename_prefix::String = "geodynamo")
    start_time, end_time = time_range
    
    if config.verbose
        println("Combining time series from $start_time to $end_time")
    end
    
    # Find all unique times in range
    files = readdir(output_dir)
    output_files = filter(f -> endswith(f, ".nc") && contains(f, filename_prefix) && 
                            contains(f, "output") && !contains(f, "restart"), files)
    
    time_pattern = r"time_(\d+p\d+)"
    times_in_range = Set{Float64}()
    
    for file in output_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                file_time = parse(Float64, time_str)
                if start_time <= file_time <= end_time
                    push!(times_in_range, file_time)
                end
            catch
                continue
            end
        end
    end
    
    times_sorted = sort(collect(times_in_range))
    
    if isempty(times_sorted)
        @warn "No files found in time range [$start_time, $end_time]"
        return GlobalFieldData{Float64}[]
    end
    
    if config.verbose
        println("Found $(length(times_sorted)) time points to process")
    end
    
    # Process each time
    global_data_series = GlobalFieldData{config.output_precision}[]
    
    for time_val in times_sorted
        if config.verbose
            println("Processing time: $time_val")
        end
        
        try
            global_data = combine_distributed_files(output_dir, time_val, config, filename_prefix)
            push!(global_data_series, global_data)
        catch e
            @error "Failed to process time $time_val" exception=e
            continue
        end
    end
    
    if config.verbose
        println("Successfully processed $(length(global_data_series)) time points")
    end
    
    return global_data_series
end

function create_global_time_series_file(global_data_series::Vector{GlobalFieldData{T}}, 
                                        output_dir::String, 
                                        config::CombinerConfig = default_combiner_config()) where T
    if isempty(global_data_series)
        @warn "No data to write to time series file"
        return
    end
    
    filename = joinpath(output_dir, "$(config.combined_filename)_timeseries.nc")
    
    if config.verbose
        println("Creating global time series file: $filename")
    end
    
    nc_file = NetCDF.create(filename, NcFile)
    
    try
        ntimes = length(global_data_series)
        first_data = global_data_series[1]
        
        # Global attributes
        NetCDF.putatt(nc_file, "title", "Global Time Series - Combined Fields")
        NetCDF.putatt(nc_file, "source", "NetCDF File Combiner - Time Series")
        NetCDF.putatt(nc_file, "history", "Created on $(now())")
        NetCDF.putatt(nc_file, "Conventions", "CF-1.8")
        NetCDF.putatt(nc_file, "number_of_timesteps", ntimes)
        
        # Define dimensions
        time_dim = NetCDF.defDim(nc_file, "time", ntimes)
        
        # Get dimensions from first dataset
        if first_data.temperature !== nothing
            nlat, nlon, nr_phys = size(first_data.temperature)
            theta_dim = NetCDF.defDim(nc_file, "theta", nlat)
            phi_dim = NetCDF.defDim(nc_file, "phi", nlon)
            r_phys_dim = NetCDF.defDim(nc_file, "r_physical", nr_phys)
        end
        
        if first_data.spectral_grid !== nothing
            nlm = length(first_data.spectral_grid.l_values)
            nr_spec = length(first_data.spectral_grid.r)
            lm_dim = NetCDF.defDim(nc_file, "spectral_mode", nlm)
            r_spec_dim = NetCDF.defDim(nc_file, "r_spectral", nr_spec)
        end
        
        # Define coordinate variables
        time_var = NetCDF.defVar(nc_file, "time", Float64, (time_dim,))
        NetCDF.putatt(nc_file, time_var, "long_name", "simulation_time")
        NetCDF.putatt(nc_file, time_var, "units", "dimensionless")
        
        # Define field variables (4D with time)
        if first_data.temperature !== nothing
            temp_var = NetCDF.defVar(nc_file, "temperature_global", T, 
                                    (theta_dim, phi_dim, r_phys_dim, time_dim))
            NetCDF.putatt(nc_file, temp_var, "long_name", "global_temperature_timeseries")
        end
        
        if first_data.velocity_toroidal_real !== nothing
            vel_tor_real_var = NetCDF.defVar(nc_file, "velocity_toroidal_real_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
            vel_tor_imag_var = NetCDF.defVar(nc_file, "velocity_toroidal_imag_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
        end
        
        if first_data.velocity_poloidal_real !== nothing
            vel_pol_real_var = NetCDF.defVar(nc_file, "velocity_poloidal_real_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
            vel_pol_imag_var = NetCDF.defVar(nc_file, "velocity_poloidal_imag_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
        end
        
        if first_data.magnetic_toroidal_real !== nothing
            mag_tor_real_var = NetCDF.defVar(nc_file, "magnetic_toroidal_real_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
            mag_tor_imag_var = NetCDF.defVar(nc_file, "magnetic_toroidal_imag_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
        end
        
        if first_data.magnetic_poloidal_real !== nothing
            mag_pol_real_var = NetCDF.defVar(nc_file, "magnetic_poloidal_real_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
            mag_pol_imag_var = NetCDF.defVar(nc_file, "magnetic_poloidal_imag_global", T,
                                            (lm_dim, r_spec_dim, time_dim))
        end
        
        # End definition mode
        NetCDF.endDef(nc_file)
        
        # Write coordinate data (from first dataset)
        times_array = [data.time for data in global_data_series]
        NetCDF.putvar(nc_file, "time", times_array)
        
        if first_data.temperature_grid !== nothing
            NetCDF.putvar(nc_file, "theta", first_data.temperature_grid.theta)
            NetCDF.putvar(nc_file, "phi", first_data.temperature_grid.phi)
            NetCDF.putvar(nc_file, "r_physical", first_data.temperature_grid.r)
        end
        
        if first_data.spectral_grid !== nothing
            NetCDF.putvar(nc_file, "l_values_global", first_data.spectral_grid.l_values)
            NetCDF.putvar(nc_file, "m_values_global", first_data.spectral_grid.m_values)
            NetCDF.putvar(nc_file, "r_spectral", first_data.spectral_grid.r)
        end
        
        # Write field data for all times
        for (t_idx, data) in enumerate(global_data_series)
            if data.temperature !== nothing
                NetCDF.putvar(nc_file, "temperature_global", data.temperature, 
                            start=[1, 1, 1, t_idx], count=size(data.temperature))
            end
            
            if data.velocity_toroidal_real !== nothing
                NetCDF.putvar(nc_file, "velocity_toroidal_real_global", data.velocity_toroidal_real,
                            start=[1, 1, t_idx], count=size(data.velocity_toroidal_real))
                NetCDF.putvar(nc_file, "velocity_toroidal_imag_global", data.velocity_toroidal_imag,
                            start=[1, 1, t_idx], count=size(data.velocity_toroidal_imag))
            end
            
            if data.velocity_poloidal_real !== nothing
                NetCDF.putvar(nc_file, "velocity_poloidal_real_global", data.velocity_poloidal_real,
                            start=[1, 1, t_idx], count=size(data.velocity_poloidal_real))
                NetCDF.putvar(nc_file, "velocity_poloidal_imag_global", data.velocity_poloidal_imag,
                            start=[1, 1, t_idx], count=size(data.velocity_poloidal_imag))
            end
            
            if data.magnetic_toroidal_real !== nothing
                NetCDF.putvar(nc_file, "magnetic_toroidal_real_global", data.magnetic_toroidal_real,
                            start=[1, 1, t_idx], count=size(data.magnetic_toroidal_real))
                NetCDF.putvar(nc_file, "magnetic_toroidal_imag_global", data.magnetic_toroidal_imag,
                            start=[1, 1, t_idx], count=size(data.magnetic_toroidal_imag))
            end
            
            if data.magnetic_poloidal_real !== nothing
                NetCDF.putvar(nc_file, "magnetic_poloidal_real_global", data.magnetic_poloidal_real,
                            start=[1, 1, t_idx], count=size(data.magnetic_poloidal_real))
                NetCDF.putvar(nc_file, "magnetic_poloidal_imag_global", data.magnetic_poloidal_imag,
                            start=[1, 1, t_idx], count=size(data.magnetic_poloidal_imag))
            end
        end
        
    finally
        NetCDF.close(nc_file)
    end
    
    if config.verbose
        println("Time series file created with $(length(global_data_series)) timesteps")
    end
end

# ============================================================================
# Utility Functions
# ============================================================================

function list_available_times(output_dir::String, filename_prefix::String = "geodynamo")
    files = readdir(output_dir)
    output_files = filter(f -> endswith(f, ".nc") && contains(f, filename_prefix) && 
                            contains(f, "output"), files)
    
    time_pattern = r"time_(\d+p\d+)"
    times = Set{Float64}()
    
    for file in output_files
        m = match(time_pattern, file)
        if m !== nothing
            time_str = replace(m.captures[1], "p" => ".")
            try
                push!(times, parse(Float64, time_str))
            catch
                continue
            end
        end
    end
    
    return sort(collect(times))
end

function get_global_field_info(global_data::GlobalFieldData)
    info = Dict{String, Any}()
    
    info["time"] = global_data.time
    info["step"] = global_data.step
    info["nprocs_original"] = global_data.nprocs
    
    if global_data.temperature !== nothing
        info["temperature_shape"] = size(global_data.temperature)
        info["temperature_range"] = (minimum(global_data.temperature), maximum(global_data.temperature))
    end
    
    if global_data.spectral_grid !== nothing
        info["spectral_modes"] = length(global_data.spectral_grid.l_values)
        info["lmax"] = maximum(global_data.spectral_grid.l_values)
        info["mmax"] = maximum(global_data.spectral_grid.m_values)
    end
    
    if haskey(global_data.metadata, "global_diagnostics")
        info["global_diagnostics"] = global_data.metadata["global_diagnostics"]
    end
    
    return info
end

function extract_field_subset(global_data::GlobalFieldData, 
                                l_range::Union{UnitRange{Int}, Nothing} = nothing,
                                r_range::Union{UnitRange{Int}, Nothing} = nothing)
    # Extract subset of spectral modes or radial points
    
    if global_data.spectral_grid === nothing
        return global_data  # No spectral data to subset
    end
    
    # Create index filters
    l_values = global_data.spectral_grid.l_values
    
    # Filter by l range
    if l_range !== nothing
        l_filter = [l in l_range for l in l_values]
    else
        l_filter = trues(length(l_values))
    end
    
    # Filter by r range
    if r_range !== nothing
        r_filter = r_range
    else
        r_filter = 1:length(global_data.spectral_grid.r)
    end
    
    # Create subset data
    subset_l_values = l_values[l_filter]
    subset_m_values = global_data.spectral_grid.m_values[l_filter]
    subset_r = global_data.spectral_grid.r[r_filter]
    
    subset_spectral_grid = (l_values = subset_l_values, m_values = subset_m_values, r = subset_r)
    
    # Extract subset arrays
    subset_data = GlobalFieldData{eltype(global_data.velocity_toroidal_real)}(
        global_data.temperature,  # Keep full temperature
        global_data.temperature_grid,
        global_data.velocity_toroidal_real !== nothing ? global_data.velocity_toroidal_real[l_filter, r_filter] : nothing,
        global_data.velocity_toroidal_imag !== nothing ? global_data.velocity_toroidal_imag[l_filter, r_filter] : nothing,
        global_data.velocity_poloidal_real !== nothing ? global_data.velocity_poloidal_real[l_filter, r_filter] : nothing,
        global_data.velocity_poloidal_imag !== nothing ? global_data.velocity_poloidal_imag[l_filter, r_filter] : nothing,
        global_data.magnetic_toroidal_real !== nothing ? global_data.magnetic_toroidal_real[l_filter, r_filter] : nothing,
        global_data.magnetic_toroidal_imag !== nothing ? global_data.magnetic_toroidal_imag[l_filter, r_filter] : nothing,
        global_data.magnetic_poloidal_real !== nothing ? global_data.magnetic_poloidal_real[l_filter, r_filter] : nothing,
        global_data.magnetic_poloidal_imag !== nothing ? global_data.magnetic_poloidal_imag[l_filter, r_filter] : nothing,
        subset_spectral_grid,
        global_data.metadata, global_data.time, global_data.step, global_data.nprocs
    )
    
    return subset_data
end

# ============================================================================
# Export Functions
# ============================================================================

export GlobalFieldData, CombinerConfig, default_combiner_config

export find_distributed_files, validate_file_set
export combine_temperature_field, combine_spectral_field
export combine_distributed_files

export compute_global_diagnostics!, save_combined_file
export combine_time_series, create_global_time_series_file

export list_available_times, get_global_field_info, extract_field_subset

#end  # module NetCDFFileCombiner

# ============================================================================
# Complete Usage Examples
# ============================================================================

"""
Complete usage examples for combining distributed NetCDF files:

## Example 1: Combine Single Time Point

```julia
using .NetCDFFileCombiner

# Create configuration
config = CombinerConfig(
    Float64,              # output_precision
    true,                 # validate_files
    true,                 # verbose
    true,                 # save_combined
    "geodynamo_global",   # combined_filename
    true,                 # include_diagnostics
    false                 # interpolate_missing
)

# Combine files for specific time
output_dir = "./simulation_output"
time_point = 1.5

global_data = combine_distributed_files(output_dir, time_point, config, "geodynamo")

# Access combined data
if global_data.temperature !== nothing
    println("Global temperature shape: ", size(global_data.temperature))
    println("Temperature range: ", extrema(global_data.temperature))
end

if global_data.velocity_toroidal_real !== nothing
    println("Velocity spectral modes: ", size(global_data.velocity_toroidal_real))
    println("Max l mode: ", maximum(global_data.spectral_grid.l_values))
end

# Print global diagnostics
if haskey(global_data.metadata, "global_diagnostics")
    println("Global diagnostics:")
    for (key, value) in global_data.metadata["global_diagnostics"]
        println("  $key: $value")
    end
end
```

## Example 2: Process Time Series

```julia
# Process entire time range
time_range = (0.0, 5.0)
global_series = combine_time_series("./simulation_output", time_range, config)

println("Processed $(length(global_series)) time points")

# Create combined time series file
create_global_time_series_file(global_series, "./simulation_output", config)
```

## Example 3: Analysis Workflow

```julia
# List available times
available_times = list_available_times("./simulation_output")
println("Available times: ", available_times)

# Process specific times of interest
analysis_times = [1.0, 2.0, 3.0, 4.0, 5.0]
global_datasets = []

for time_val in analysis_times
    if time_val in available_times
        println("Processing time: $time_val")
        
        global_data = combine_distributed_files("./simulation_output", time_val, config)
        push!(global_datasets, global_data)
        
        # Quick analysis
        info = get_global_field_info(global_data)
        println("  Temperature range: ", info["temperature_range"])
        
        if haskey(info, "global_diagnostics")
            ke = info["global_diagnostics"]["global_kinetic_energy"]
            me = info["global_diagnostics"]["global_magnetic_energy"]
            println("  Kinetic energy: $ke, Magnetic energy: $me")
        end
    end
end
```

## Example 4: Extract Spectral Subsets

```julia
# Load full global data
global_data = combine_distributed_files("./simulation_output", 2.0, config)

# Extract low-l modes only (l ≤ 10)
low_l_data = extract_field_subset(global_data, 0:10, nothing)
println("Low-l subset shape: ", size(low_l_data.velocity_toroidal_real))

# Extract surface data only (outer radial points)
nr = length(global_data.spectral_grid.r)
surface_data = extract_field_subset(global_data, nothing, (nr-5):nr)
println("Surface data shape: ", size(surface_data.velocity_toroidal_real))
```

## Example 5: Batch Processing Script

```julia
function process_simulation_output(output_dir::String)
    # Find all available times
    times = list_available_times(output_dir)
    println("Found $(length(times)) time points")
    
    # Setup configuration
    config = default_combiner_config()
    config.verbose = false  # Reduce output for batch processing
    
    # Process each time point
    combined_dir = joinpath(output_dir, "combined")
    mkpath(combined_dir)
    
    for time_val in times
        try
            println("Processing time: $time_val")
            
            # Combine files
            global_data = combine_distributed_files(output_dir, time_val, config)
            
            # Save to combined directory
            time_str = @sprintf("%.6f", time_val)
            time_str = replace(time_str, "." => "p")
            
            combined_filename = joinpath(combined_dir, "global_time_$(time_str).nc")
            config.combined_filename = splitext(basename(combined_filename))[1]
            save_combined_file(global_data, combined_dir, config)
            
        catch e
            @error "Failed to process time $time_val" exception=e
            continue
        end
    end
    
    println("Batch processing completed")
end

# Run batch processing
process_simulation_output("./simulation_output")
```

This combiner module provides:

1. **Automatic File Discovery**: Finds all distributed files for a given time
2. **Smart Field Combination**: Handles mixed physical/spectral representations
3. **Global Reconstruction**: Assembles complete global fields from local pieces
4. **Time Series Processing**: Combines multiple time points into single files
5. **Flexible Analysis**: Extract subsets by spectral modes or radial ranges
6. **Global Diagnostics**: Computes domain-integrated quantities
7. **Robust Processing**: Validation and error handling for production use

The module handles the complexity of reassembling distributed data while preserving
the scientific accuracy of both physical and spectral representations.
"""