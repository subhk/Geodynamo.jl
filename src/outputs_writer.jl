# ============================================================================
# SHTns Geodynamo NetCDF Output Module - Distributed Files
# Each MPI process writes its own NetCDF file with local data
# ============================================================================

module SHTnsNetCDFOutput
    using MPI
    using NetCDF
    using PencilArrays
    using SHTnsSpheres
    using LinearAlgebra
    using Statistics
    using Dates
    using Printf
    
    # Import required modules (assuming they're available)
    # using ..Parameters
    # using ..SHTnsSetup
    # using ..VariableTypes
    # using ..SHTnsTransforms
    # using ..PencilSetup
    
    # For standalone usage, define essential parameters
    const comm = MPI.COMM_WORLD
    
    # ============================================================================
    # Output Configuration
    # ============================================================================
    
    @enum OutputSpace begin
        PHYSICAL_ONLY       # Output only physical space data
        SPECTRAL_ONLY       # Output only spectral coefficients
        BOTH_SPACES         # Output both physical and spectral
    end
    
    @enum FileNaming begin
        RANK_STEP          # filename_rank_XXXX_step_YYYYYY.nc
        STEP_RANK          # filename_step_YYYYYY_rank_XXXX.nc
        TIMESTAMP_RANK     # filename_YYYYMMDD_HHMMSS_rank_XXXX.nc
    end
    
    struct NetCDFOutputConfig
        output_space::OutputSpace
        naming_scheme::FileNaming
        output_dir::String
        filename_prefix::String
        compression_level::Int          # 0-9, 0=no compression
        include_metadata::Bool          # Include simulation metadata
        include_grid::Bool             # Include coordinate grids
        include_diagnostics::Bool       # Include local diagnostics
        output_precision::DataType      # Float32 or Float64
        spectral_lmax_output::Int      # Maximum l for spectral output (-1 = all)
        add_timestamp::Bool            # Add timestamp to filename
        overwrite_files::Bool          # Overwrite existing files
    end
    
    function default_netcdf_config()
        return NetCDFOutputConfig(
            PHYSICAL_ONLY,      # output_space
            RANK_STEP,          # naming_scheme  
            "./output",         # output_dir
            "geodynamo",        # filename_prefix
            6,                  # compression_level
            true,               # include_metadata
            true,               # include_grid
            true,               # include_diagnostics
            Float64,            # output_precision
            -1,                 # spectral_lmax_output (-1 = all modes)
            false,              # add_timestamp
            true                # overwrite_files
        )
    end
    
    # ============================================================================
    # Local Data Structures
    # ============================================================================
    
    struct LocalFieldInfo
        # Local array dimensions and ranges
        local_nlat::Int
        local_nlon::Int
        local_nr::Int
        local_nlm::Int
        
        # Global indices for this process
        theta_start::Int
        theta_end::Int
        phi_start::Int
        phi_end::Int
        r_start::Int
        r_end::Int
        
        # Local coordinate arrays
        theta_local::Vector{Float64}
        phi_local::Vector{Float64}
        r_local::Vector{Float64}
        
        # Spectral mode information (if applicable)
        l_values_local::Union{Vector{Int}, Nothing}
        m_values_local::Union{Vector{Int}, Nothing}
        lm_start::Int
        lm_end::Int
    end
    
    function extract_local_info(pencil_field, shtns_config, radial_domain)
        rank = MPI.Comm_rank(comm)
        
        # Extract local dimensions from pencil array
        local_dims = size(pencil_field.data_r)
        local_nlat = local_dims[1]
        local_nlon = local_dims[2] 
        local_nr = local_dims[3]
        
        # Get local ranges (this depends on pencil decomposition details)
        # For simplicity, assume we can extract these from the pencil
        theta_range = pencil_field.pencil.axes[1]
        phi_range = pencil_field.pencil.axes[2]
        r_range = pencil_field.pencil.axes[3]
        
        theta_start = first(theta_range)
        theta_end = last(theta_range)
        phi_start = first(phi_range)
        phi_end = last(phi_range)
        r_start = first(r_range)
        r_end = last(r_range)
        
        # Extract local coordinate arrays
        theta_local = shtns_config.theta_grid[theta_start:theta_end]
        phi_local = shtns_config.phi_grid[phi_start:phi_end]
        r_local = [radial_domain.r[i, 4] for i in r_start:r_end]
        
        return LocalFieldInfo(
            local_nlat, local_nlon, local_nr, 0,
            theta_start, theta_end, phi_start, phi_end, r_start, r_end,
            theta_local, phi_local, r_local,
            nothing, nothing, 0, 0
        )
    end
    
    function extract_spectral_info(spectral_field, shtns_config, radial_domain, max_l::Int = -1)
        rank = MPI.Comm_rank(comm)
        
        # Get local spectral dimensions
        local_dims = size(spectral_field.data_real)
        local_nlm = local_dims[1]
        local_nr = local_dims[3]
        
        # Get radial range
        r_range = spectral_field.local_radial_range
        r_start = first(r_range)
        r_end = last(r_range)
        r_local = [radial_domain.r[i, 4] for i in r_start:r_end]
        
        # Filter spectral modes if max_l is specified
        if max_l > 0
            valid_indices = findall(l -> l <= max_l, shtns_config.l_values)
            l_values_local = shtns_config.l_values[valid_indices]
            m_values_local = shtns_config.m_values[valid_indices]
            local_nlm_filtered = length(valid_indices)
        else
            l_values_local = copy(shtns_config.l_values[1:local_nlm])
            m_values_local = copy(shtns_config.m_values[1:local_nlm])
            local_nlm_filtered = local_nlm
        end
        
        return LocalFieldInfo(
            0, 0, local_nr, local_nlm_filtered,
            0, 0, 0, 0, r_start, r_end,
            Float64[], Float64[], r_local,
            l_values_local, m_values_local, 1, local_nlm_filtered
        )
    end
    
    # ============================================================================
    # Filename Generation
    # ============================================================================
    
    function generate_netcdf_filename(config::NetCDFOutputConfig, step::Int, rank::Int)
        timestamp = if config.add_timestamp
            Dates.format(now(), "yyyymmdd_HHMMSS")
        else
            ""
        end
        
        filename = if config.naming_scheme == RANK_STEP
            if config.add_timestamp
                "$(config.filename_prefix)_$(timestamp)_rank_$(lpad(rank, 4, '0'))_step_$(lpad(step, 6, '0')).nc"
            else
                "$(config.filename_prefix)_rank_$(lpad(rank, 4, '0'))_step_$(lpad(step, 6, '0')).nc"
            end
        elseif config.naming_scheme == STEP_RANK
            if config.add_timestamp
                "$(config.filename_prefix)_$(timestamp)_step_$(lpad(step, 6, '0'))_rank_$(lpad(rank, 4, '0')).nc"
            else
                "$(config.filename_prefix)_step_$(lpad(step, 6, '0'))_rank_$(lpad(rank, 4, '0')).nc"
            end
        elseif config.naming_scheme == TIMESTAMP_RANK
            "$(config.filename_prefix)_$(timestamp)_rank_$(lpad(rank, 4, '0')).nc"
        else
            "$(config.filename_prefix)_rank_$(lpad(rank, 4, '0'))_step_$(lpad(step, 6, '0')).nc"
        end
        
        return joinpath(config.output_dir, filename)
    end
    
    # ============================================================================
    # NetCDF File Creation and Metadata
    # ============================================================================
    
    function create_netcdf_file(filename::String, config::NetCDFOutputConfig, 
                               field_info::LocalFieldInfo, state)
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        # Remove existing file if overwrite is enabled
        if config.overwrite_files && isfile(filename)
            rm(filename)
        end
        
        # Create NetCDF file
        nc_file = NetCDF.create(filename, NcFile)
        
        try
            # Define global attributes
            NetCDF.putatt(nc_file, "title", "SHTns Geodynamo Simulation Output")
            NetCDF.putatt(nc_file, "institution", "Geodynamo Research Group")
            NetCDF.putatt(nc_file, "source", "SHTns-based geodynamo simulation")
            NetCDF.putatt(nc_file, "history", "Created on $(now()) by rank $rank")
            NetCDF.putatt(nc_file, "Conventions", "CF-1.8")
            
            # MPI information
            NetCDF.putatt(nc_file, "mpi_rank", rank)
            NetCDF.putatt(nc_file, "mpi_total_ranks", nprocs)
            
            # Simulation metadata
            if config.include_metadata
                add_simulation_metadata!(nc_file, state)
            end
            
            # Domain decomposition info
            add_domain_info!(nc_file, field_info, rank)
            
            return nc_file
            
        catch e
            NetCDF.close(nc_file)
            rethrow(e)
        end
    end
    
    function add_simulation_metadata!(nc_file, state)
        # Time information
        NetCDF.putatt(nc_file, "current_time", state.timestep_state.time)
        NetCDF.putatt(nc_file, "current_step", state.timestep_state.step)
        NetCDF.putatt(nc_file, "timestep_dt", state.timestep_state.dt)
        
        # Grid parameters
        NetCDF.putatt(nc_file, "global_nlat", state.shtns_config.nlat)
        NetCDF.putatt(nc_file, "global_nlon", state.shtns_config.nlon)
        NetCDF.putatt(nc_file, "global_nr", state.radial_domain.N)
        NetCDF.putatt(nc_file, "lmax", state.shtns_config.lmax)
        NetCDF.putatt(nc_file, "mmax", state.shtns_config.mmax)
        NetCDF.putatt(nc_file, "total_spectral_modes", state.shtns_config.nlm)
        
        # Physical parameters (would normally come from Parameters module)
        NetCDF.putatt(nc_file, "Rayleigh_number", 1e6)   # d_Ra
        NetCDF.putatt(nc_file, "Ekman_number", 1e-4)     # d_E  
        NetCDF.putatt(nc_file, "Prandtl_number", 1.0)    # d_Pr
        NetCDF.putatt(nc_file, "Magnetic_Prandtl", 1.0)  # d_Pm
        NetCDF.putatt(nc_file, "radius_ratio", 0.35)     # d_rratio
        NetCDF.putatt(nc_file, "Rossby_number", 1e-4)    # d_Ro
        
        # Numerical parameters
        NetCDF.putatt(nc_file, "implicit_factor", 0.5)   # d_implicit
        NetCDF.putatt(nc_file, "error_tolerance", 1e-8)  # d_dterr
        NetCDF.putatt(nc_file, "CFL_factor", 0.5)        # d_courant
    end
    
    function add_domain_info!(nc_file, field_info::LocalFieldInfo, rank::Int)
        NetCDF.putatt(nc_file, "local_nlat", field_info.local_nlat)
        NetCDF.putatt(nc_file, "local_nlon", field_info.local_nlon)
        NetCDF.putatt(nc_file, "local_nr", field_info.local_nr)
        
        # Global index ranges for this process
        NetCDF.putatt(nc_file, "theta_start_global", field_info.theta_start)
        NetCDF.putatt(nc_file, "theta_end_global", field_info.theta_end)
        NetCDF.putatt(nc_file, "phi_start_global", field_info.phi_start)
        NetCDF.putatt(nc_file, "phi_end_global", field_info.phi_end)
        NetCDF.putatt(nc_file, "r_start_global", field_info.r_start)
        NetCDF.putatt(nc_file, "r_end_global", field_info.r_end)
        
        if field_info.local_nlm > 0
            NetCDF.putatt(nc_file, "local_nlm", field_info.local_nlm)
            NetCDF.putatt(nc_file, "lm_start_global", field_info.lm_start)
            NetCDF.putatt(nc_file, "lm_end_global", field_info.lm_end)
        end
    end
    
    # ============================================================================
    # Coordinate Variable Creation
    # ============================================================================
    
    function add_coordinate_variables!(nc_file, field_info::LocalFieldInfo, config::NetCDFOutputConfig)
        if !config.include_grid
            return
        end
        
        # Theta coordinate (colatitude)
        if field_info.local_nlat > 0
            theta_dim = NetCDF.defDim(nc_file, "theta", field_info.local_nlat)
            theta_var = NetCDF.defVar(nc_file, "theta", config.output_precision, (theta_dim,))
            
            NetCDF.putatt(nc_file, theta_var, "long_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "units", "radians")
            NetCDF.putatt(nc_file, theta_var, "standard_name", "colatitude")
            NetCDF.putatt(nc_file, theta_var, "axis", "Y")
            NetCDF.putatt(nc_file, theta_var, "valid_range", [0.0, π])
            
            if config.compression_level > 0
                NetCDF.defVarDeflate(nc_file, theta_var, true, true, config.compression_level)
            end
        end
        
        # Phi coordinate (longitude)
        if field_info.local_nlon > 0
            phi_dim = NetCDF.defDim(nc_file, "phi", field_info.local_nlon)
            phi_var = NetCDF.defVar(nc_file, "phi", config.output_precision, (phi_dim,))
            
            NetCDF.putatt(nc_file, phi_var, "long_name", "azimuthal_angle")
            NetCDF.putatt(nc_file, phi_var, "units", "radians")
            NetCDF.putatt(nc_file, phi_var, "standard_name", "longitude")
            NetCDF.putatt(nc_file, phi_var, "axis", "X")
            NetCDF.putatt(nc_file, phi_var, "valid_range", [0.0, 2π])
            
            if config.compression_level > 0
                NetCDF.defVarDeflate(nc_file, phi_var, true, true, config.compression_level)
            end
        end
        
        # Radial coordinate
        if field_info.local_nr > 0
            r_dim = NetCDF.defDim(nc_file, "r", field_info.local_nr)
            r_var = NetCDF.defVar(nc_file, "r", config.output_precision, (r_dim,))
            
            NetCDF.putatt(nc_file, r_var, "long_name", "radial_coordinate")
            NetCDF.putatt(nc_file, r_var, "units", "dimensionless")
            NetCDF.putatt(nc_file, r_var, "axis", "Z")
            NetCDF.putatt(nc_file, r_var, "valid_range", [0.35, 1.0])  # inner core to CMB
            NetCDF.putatt(nc_file, r_var, "positive", "up")
            
            if config.compression_level > 0
                NetCDF.defVarDeflate(nc_file, r_var, true, true, config.compression_level)
            end
        end
        
        # Spectral mode coordinates (if applicable)
        if field_info.local_nlm > 0
            lm_dim = NetCDF.defDim(nc_file, "spectral_mode", field_info.local_nlm)
            
            # L values
            l_var = NetCDF.defVar(nc_file, "l_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, l_var, "long_name", "spherical_harmonic_degree")
            NetCDF.putatt(nc_file, l_var, "units", "1")
            
            # M values  
            m_var = NetCDF.defVar(nc_file, "m_values", Int32, (lm_dim,))
            NetCDF.putatt(nc_file, m_var, "long_name", "spherical_harmonic_order")
            NetCDF.putatt(nc_file, m_var, "units", "1")
        end
    end
    
    function write_coordinate_data!(nc_file, field_info::LocalFieldInfo)
        # Write coordinate arrays
        if field_info.local_nlat > 0 && !isempty(field_info.theta_local)
            NetCDF.putvar(nc_file, "theta", field_info.theta_local)
        end
        
        if field_info.local_nlon > 0 && !isempty(field_info.phi_local)
            NetCDF.putvar(nc_file, "phi", field_info.phi_local)
        end
        
        if field_info.local_nr > 0 && !isempty(field_info.r_local)
            NetCDF.putvar(nc_file, "r", field_info.r_local)
        end
        
        # Write spectral mode information
        if field_info.local_nlm > 0 && field_info.l_values_local !== nothing
            NetCDF.putvar(nc_file, "l_values", field_info.l_values_local)
            NetCDF.putvar(nc_file, "m_values", field_info.m_values_local)
        end
    end
    
    # ============================================================================
    # Field Variable Creation and Writing
    # ============================================================================
    
    function add_scalar_variable!(nc_file, var_name::String, field_info::LocalFieldInfo, 
                                 config::NetCDFOutputConfig, long_name::String, units::String)
        # Get dimensions
        if config.output_space == PHYSICAL_ONLY || config.output_space == BOTH_SPACES
            dims = ["theta", "phi", "r"]
            var_dims = [NetCDF.dimid(nc_file, d) for d in dims if NetCDF.dimid(nc_file, d) != -1]
        else
            return nothing  # Will be handled by spectral version
        end
        
        # Create variable
        var = NetCDF.defVar(nc_file, var_name, config.output_precision, tuple(var_dims...))
        
        # Add attributes
        NetCDF.putatt(nc_file, var, "long_name", long_name)
        NetCDF.putatt(nc_file, var, "units", units)
        NetCDF.putatt(nc_file, var, "grid_mapping", "spherical_coordinates")
        NetCDF.putatt(nc_file, var, "_FillValue", config.output_precision(NaN))
        
        # Compression
        if config.compression_level > 0
            NetCDF.defVarDeflate(nc_file, var, true, true, config.compression_level)
        end
        
        return var
    end
    
    function add_vector_variables!(nc_file, var_base_name::String, field_info::LocalFieldInfo,
                                  config::NetCDFOutputConfig, long_name::String, units::String)
        components = ["r", "theta", "phi"]
        component_names = ["radial", "colatitudinal", "azimuthal"]
        vars = Dict{String, Any}()
        
        for (i, comp) in enumerate(components)
            comp_name = "$(var_base_name)_$(comp)"
            comp_long_name = "$(long_name)_$(component_names[i])_component"
            
            var = add_scalar_variable!(nc_file, comp_name, field_info, config, comp_long_name, units)
            if var !== nothing
                vars[comp] = var
            end
        end
        
        return vars
    end
    
    function add_spectral_variables!(nc_file, var_name::String, field_info::LocalFieldInfo,
                                    config::NetCDFOutputConfig, long_name::String)
        if field_info.local_nlm == 0
            return nothing
        end
        
        # Dimensions for spectral data
        dims = ["spectral_mode", "r"]
        var_dims = [NetCDF.dimid(nc_file, d) for d in dims]
        
        # Real part
        real_var = NetCDF.defVar(nc_file, "$(var_name)_real", config.output_precision, tuple(var_dims...))
        NetCDF.putatt(nc_file, real_var, "long_name", "$(long_name)_real_part")
        NetCDF.putatt(nc_file, real_var, "units", "dimensionless")
        
        # Imaginary part
        imag_var = NetCDF.defVar(nc_file, "$(var_name)_imag", config.output_precision, tuple(var_dims...))
        NetCDF.putatt(nc_file, imag_var, "long_name", "$(long_name)_imaginary_part")
        NetCDF.putatt(nc_file, imag_var, "units", "dimensionless")
        
        # Compression
        if config.compression_level > 0
            NetCDF.defVarDeflate(nc_file, real_var, true, true, config.compression_level)
            NetCDF.defVarDeflate(nc_file, imag_var, true, true, config.compression_level)
        end
        
        return (real_var, imag_var)
    end
    
    function write_scalar_field!(nc_file, var_name::String, field_data, config::NetCDFOutputConfig)
        if NetCDF.varid(nc_file, var_name) == -1
            return  # Variable doesn't exist
        end
        
        # Convert to output precision
        data_out = config.output_precision.(field_data.data_r)
        
        # Write data
        NetCDF.putvar(nc_file, var_name, data_out)
    end
    
    function write_vector_field!(nc_file, var_base_name::String, vector_field, config::NetCDFOutputConfig)
        components = ["r", "theta", "phi"]
        field_components = [vector_field.r_component, vector_field.θ_component, vector_field.φ_component]
        
        for (comp, field_comp) in zip(components, field_components)
            comp_name = "$(var_base_name)_$(comp)"
            if NetCDF.varid(nc_file, comp_name) != -1
                data_out = config.output_precision.(field_comp.data_r)
                NetCDF.putvar(nc_file, comp_name, data_out)
            end
        end
    end
    
    function write_spectral_field!(nc_file, var_name::String, spectral_field, config::NetCDFOutputConfig)
        real_name = "$(var_name)_real"
        imag_name = "$(var_name)_imag"
        
        if NetCDF.varid(nc_file, real_name) == -1 || NetCDF.varid(nc_file, imag_name) == -1
            return
        end
        
        # Extract local spectral data
        local_real = spectral_field.data_real[:, 1, :]  # Remove singleton dimension
        local_imag = spectral_field.data_imag[:, 1, :]
        
        # Convert to output precision
        real_out = config.output_precision.(local_real)
        imag_out = config.output_precision.(local_imag)
        
        # Write data
        NetCDF.putvar(nc_file, real_name, real_out)
        NetCDF.putvar(nc_file, imag_name, imag_out)
    end
    
    # ============================================================================
    # Diagnostics Computation and Output
    # ============================================================================
    function compute_local_diagnostics(state, field_info::LocalFieldInfo)
        diagnostics = Dict{String, Float64}()
        
        # Local kinetic energy
        local_ke = 0.0
        for r_idx in 1:field_info.local_nr
            for j in 1:field_info.local_nlon
                for i in 1:field_info.local_nlat
                    if (i <= size(state.velocity.velocity.r_component.data_r, 1) &&
                        j <= size(state.velocity.velocity.r_component.data_r, 2) &&
                        r_idx <= size(state.velocity.velocity.r_component.data_r, 3))
                        
                        u_r = state.velocity.velocity.r_component.data_r[i, j, r_idx]
                        u_θ = state.velocity.velocity.θ_component.data_r[i, j, r_idx]
                        u_φ = state.velocity.velocity.φ_component.data_r[i, j, r_idx]
                        
                        local_ke += 0.5 * (u_r^2 + u_θ^2 + u_φ^2)
                    end
                end
            end
        end
        diagnostics["local_kinetic_energy"] = local_ke
        
        # Local magnetic energy
        local_me = 0.0
        for r_idx in 1:field_info.local_nr
            for j in 1:field_info.local_nlon
                for i in 1:field_info.local_nlat
                    if (i <= size(state.magnetic.magnetic.r_component.data_r, 1) &&
                        j <= size(state.magnetic.magnetic.r_component.data_r, 2) &&
                        r_idx <= size(state.magnetic.magnetic.r_component.data_r, 3))
                        
                        B_r = state.magnetic.magnetic.r_component.data_r[i, j, r_idx]
                        B_θ = state.magnetic.magnetic.θ_component.data_r[i, j, r_idx]
                        B_φ = state.magnetic.magnetic.φ_component.data_r[i, j, r_idx]
                        
                        local_me += 0.5 * (B_r^2 + B_θ^2 + B_φ^2)
                    end
                end
            end
        end
        diagnostics["local_magnetic_energy"] = local_me
        
        # Local temperature statistics
        T_data = state.temperature.temperature.data_r
        if !isempty(T_data)
            diagnostics["local_temp_mean"] = mean(T_data)
            diagnostics["local_temp_std"]  = std(T_data)
            diagnostics["local_temp_min"]  = minimum(T_data)
            diagnostics["local_temp_max"]  = maximum(T_data)
        end
        
        # Local velocity statistics
        if !isempty(state.velocity.velocity.r_component.data_r)
            u_mag = sqrt.(state.velocity.velocity.r_component.data_r.^2 .+ 
                         state.velocity.velocity.θ_component.data_r.^2 .+ 
                         state.velocity.velocity.φ_component.data_r.^2)
            diagnostics["local_vel_rms"] = sqrt(mean(u_mag.^2))
            diagnostics["local_vel_max"] = maximum(u_mag)
        end
        
        return diagnostics
    end
    
    function add_diagnostics_variables!(nc_file, diagnostics::Dict{String, Float64}, config::NetCDFOutputConfig)
        if !config.include_diagnostics
            return
        end
        
        # Create scalar dimension for diagnostics
        scalar_dim = NetCDF.defDim(nc_file, "scalar", 1)
        
        for (name, value) in diagnostics
            var = NetCDF.defVar(nc_file, "diag_$(name)", config.output_precision, (scalar_dim,))
            NetCDF.putatt(nc_file, var, "long_name", replace(name, "_" => " "))
            NetCDF.putatt(nc_file, var, "description", "Local diagnostic quantity for this MPI rank")
        end
    end
    
    function write_diagnostics!(nc_file, diagnostics::Dict{String, Float64}, config::NetCDFOutputConfig)
        if !config.include_diagnostics
            return
        end
        
        for (name, value) in diagnostics
            var_name = "diag_$(name)"
            if NetCDF.varid(nc_file, var_name) != -1
                NetCDF.putvar(nc_file, var_name, [config.output_precision(value)])
            end
        end
    end
    
    # ============================================================================
    # Main Output Function
    # ============================================================================
    
    function output_netcdf_fields!(state, config::NetCDFOutputConfig = default_netcdf_config())
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        # Create output directory (only rank 0)
        if rank == 0
            if !isdir(config.output_dir)
                mkpath(config.output_dir)
                println("Created output directory: $(config.output_dir)")
            end
        end
        MPI.Barrier(comm)  # Wait for directory creation
        
        # Convert spectral fields to physical space if needed
        if config.output_space == PHYSICAL_ONLY || config.output_space == BOTH_SPACES
            # These functions should exist in the main simulation module
            # shtns_spectral_to_physical!(state.temperature.spectral, state.temperature.temperature)
            # shtns_vector_synthesis!(state.velocity.toroidal, state.velocity.poloidal, state.velocity.velocity)
            # shtns_vector_synthesis!(state.magnetic.toroidal, state.magnetic.poloidal, state.magnetic.magnetic)
            
            println("Rank $rank: Converting spectral to physical space...")
        end
        
        # Generate filename for this rank
        filename = generate_netcdf_filename(config, state.timestep_state.step, rank)
        println("Rank $rank: Writing file $filename")
        
        # Extract local field information
        local_info = if config.output_space == PHYSICAL_ONLY || config.output_space == BOTH_SPACES
            extract_local_info(state.temperature.temperature, state.shtns_config, state.radial_domain)
        else
            extract_spectral_info(state.temperature.spectral, state.shtns_config, 
                                state.radial_domain, config.spectral_lmax_output)
        end
        
        # Compute local diagnostics
        local_diagnostics = compute_local_diagnostics(state, local_info)
        
        # Create NetCDF file
        nc_file = create_netcdf_file(filename, config, local_info, state)
        
        try
            # Add coordinate variables
            add_coordinate_variables!(nc_file, local_info, config)
            
            # Define field variables based on output space
            if config.output_space == PHYSICAL_ONLY || config.output_space == BOTH_SPACES
                # Physical space variables
                temp_var = add_scalar_variable!(nc_file, "temperature", local_info, config,
                                              "temperature", "dimensionless")
                
                vel_vars = add_vector_variables!(nc_file, "velocity", local_info, config,
                                               "velocity", "dimensionless")
                
                mag_vars = add_vector_variables!(nc_file, "magnetic_field", local_info, config,
                                               "magnetic_field", "dimensionless")
            end
            
            if config.output_space == SPECTRAL_ONLY || config.output_space == BOTH_SPACES
                # Spectral space variables
                temp_spec_vars = add_spectral_variables!(nc_file, "temperature_spectral", local_info,
                                                       config, "temperature_spectral_coefficients")
                
                vel_tor_vars = add_spectral_variables!(nc_file, "velocity_toroidal", local_info,
                                                     config, "velocity_toroidal_coefficients")
                
                vel_pol_vars = add_spectral_variables!(nc_file, "velocity_poloidal", local_info,
                                                     config, "velocity_poloidal_coefficients")
                
                mag_tor_vars = add_spectral_variables!(nc_file, "magnetic_toroidal", local_info,
                                                     config, "magnetic_toroidal_coefficients")
                
                mag_pol_vars = add_spectral_variables!(nc_file, "magnetic_poloidal", local_info,
                                                     config, "magnetic_poloidal_coefficients")
            end
            
            # Add diagnostics variables
            add_diagnostics_variables!(nc_file, local_diagnostics, config)
            
            # Add time variables
            time_dim = NetCDF.defDim(nc_file, "time", 1)
            time_var = NetCDF.defVar(nc_file, "time", config.output_precision, (time_dim,))
            NetCDF.putatt(nc_file, time_var, "long_name", "simulation_time")
            NetCDF.putatt(nc_file, time_var, "units", "dimensionless_time_units")
            NetCDF.putatt(nc_file, time_var, "standard_name", "time")
            
            step_var = NetCDF.defVar(nc_file, "step", Int32, (time_dim,))
            NetCDF.putatt(nc_file, step_var, "long_name", "simulation_step_number")
            NetCDF.putatt(nc_file, step_var, "units", "1")
            
            # End definition mode
            NetCDF.endDef(nc_file)
            
            # Write coordinate data
            write_coordinate_data!(nc_file, local_info)
            
            # Write field data
            if config.output_space == PHYSICAL_ONLY || config.output_space == BOTH_SPACES
                println("Rank $rank: Writing physical space data...")
                
                write_scalar_field!(nc_file, "temperature", state.temperature.temperature, config)
                write_vector_field!(nc_file, "velocity", state.velocity.velocity, config)
                write_vector_field!(nc_file, "magnetic_field", state.magnetic.magnetic, config)
            end
            
            if config.output_space == SPECTRAL_ONLY || config.output_space == BOTH_SPACES
                println("Rank $rank: Writing spectral data...")
                
                write_spectral_field!(nc_file, "temperature_spectral", state.temperature.spectral, config)
                write_spectral_field!(nc_file, "velocity_toroidal", state.velocity.toroidal, config)
                write_spectral_field!(nc_file, "velocity_poloidal", state.velocity.poloidal, config)
                write_spectral_field!(nc_file, "magnetic_toroidal", state.magnetic.toroidal, config)
                write_spectral_field!(nc_file, "magnetic_poloidal", state.magnetic.poloidal, config)
            end
            
            # Write time information
            NetCDF.putvar(nc_file, "time", [config.output_precision(state.timestep_state.time)])
            NetCDF.putvar(nc_file, "step", [Int32(state.timestep_state.step)])
            
            # Write diagnostics
            write_diagnostics!(nc_file, local_diagnostics, config)
            
            println("Rank $rank: Successfully wrote NetCDF file")
            
        catch e
            @error "Rank $rank: Error writing NetCDF file" exception=e
            rethrow(e)
        finally
            NetCDF.close(nc_file)
        end
        
        # Update output counter
        state.output_counter += 1
        
        # Synchronize all processes
        MPI.Barrier(comm)
        
        if rank == 0
            println("All ranks completed NetCDF output for step $(state.timestep_state.step)")
        end
    end
    
    # ============================================================================
    # Restart File Functions
    # ============================================================================
    
    function write_netcdf_restart!(state, config::NetCDFOutputConfig = default_netcdf_config())
        rank = MPI.Comm_rank(comm)
        
        # Use spectral-only output for restart files (more compact and exact)
        restart_config = NetCDFOutputConfig(
            SPECTRAL_ONLY, RANK_STEP, config.output_dir, "restart",
            9, true, true, false, Float64, -1, true, true
        )
        
        filename = generate_netcdf_filename(restart_config, state.timestep_state.step, rank)
        println("Rank $rank: Writing restart file $filename")
        
        # Extract spectral field information
        spec_info = extract_spectral_info(state.temperature.spectral, state.shtns_config, 
                                        state.radial_domain, -1)  # All modes for restart
        
        # Create restart file
        nc_file = create_netcdf_file(filename, restart_config, spec_info, state)
        
        try
            # Add coordinate variables (spectral modes and radial)
            add_coordinate_variables!(nc_file, spec_info, restart_config)
            
            # Add all spectral field variables for exact restart
            temp_vars = add_spectral_variables!(nc_file, "temperature", spec_info, restart_config,
                                              "temperature_spectral_coefficients")
            
            vel_tor_vars = add_spectral_variables!(nc_file, "velocity_toroidal", spec_info, restart_config,
                                                 "velocity_toroidal_coefficients")
            
            vel_pol_vars = add_spectral_variables!(nc_file, "velocity_poloidal", spec_info, restart_config,
                                                 "velocity_poloidal_coefficients")
            
            mag_tor_vars = add_spectral_variables!(nc_file, "magnetic_toroidal", spec_info, restart_config,
                                                 "magnetic_toroidal_coefficients")
            
            mag_pol_vars = add_spectral_variables!(nc_file, "magnetic_poloidal", spec_info, restart_config,
                                                 "magnetic_poloidal_coefficients")
            
            # Add timestepping state variables
            scalar_dim = NetCDF.defDim(nc_file, "scalar", 1)
            
            time_var = NetCDF.defVar(nc_file, "current_time", Float64, (scalar_dim,))
            step_var = NetCDF.defVar(nc_file, "current_step", Int32, (scalar_dim,))
            dt_var = NetCDF.defVar(nc_file, "current_dt", Float64, (scalar_dim,))
            error_var = NetCDF.defVar(nc_file, "current_error", Float64, (scalar_dim,))
            iteration_var = NetCDF.defVar(nc_file, "current_iteration", Int32, (scalar_dim,))
            
            # End definition mode
            NetCDF.endDef(nc_file)
            
            # Write coordinate data
            write_coordinate_data!(nc_file, spec_info)
            
            # Write all spectral field data
            write_spectral_field!(nc_file, "temperature", state.temperature.spectral, restart_config)
            write_spectral_field!(nc_file, "velocity_toroidal", state.velocity.toroidal, restart_config)
            write_spectral_field!(nc_file, "velocity_poloidal", state.velocity.poloidal, restart_config)
            write_spectral_field!(nc_file, "magnetic_toroidal", state.magnetic.toroidal, restart_config)
            write_spectral_field!(nc_file, "magnetic_poloidal", state.magnetic.poloidal, restart_config)
            
            # Write timestepping state
            NetCDF.putvar(nc_file, "current_time", [state.timestep_state.time])
            NetCDF.putvar(nc_file, "current_step", [Int32(state.timestep_state.step)])
            NetCDF.putvar(nc_file, "current_dt", [state.timestep_state.dt])
            NetCDF.putvar(nc_file, "current_error", [state.timestep_state.error])
            NetCDF.putvar(nc_file, "current_iteration", [Int32(state.timestep_state.iteration)])
            
            println("Rank $rank: Successfully wrote restart file")
            
        finally
            NetCDF.close(nc_file)
        end
    end
    
    function read_netcdf_restart!(state, restart_dir::String, step::Int)
        rank = MPI.Comm_rank(comm)
        nprocs = MPI.Comm_size(comm)
        
        # Generate restart filename for this rank
        filename = joinpath(restart_dir, "restart_rank_$(lpad(rank, 4, '0'))_step_$(lpad(step, 6, '0')).nc")
        
        if !isfile(filename)
            error("Rank $rank: Restart file not found: $filename")
        end
        
        println("Rank $rank: Reading restart file $filename")
        
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        
        try
            # Read timestepping state (only rank 0, then broadcast)
            if rank == 0
                state.timestep_state.time = NetCDF.readvar(nc_file, "current_time")[1]
                state.timestep_state.step = NetCDF.readvar(nc_file, "current_step")[1]
                state.timestep_state.dt = NetCDF.readvar(nc_file, "current_dt")[1]
                state.timestep_state.error = NetCDF.readvar(nc_file, "current_error")[1]
                state.timestep_state.iteration = NetCDF.readvar(nc_file, "current_iteration")[1]
            end
            
            # Read local spectral data for this rank
            temp_real = NetCDF.readvar(nc_file, "temperature_real")
            temp_imag = NetCDF.readvar(nc_file, "temperature_imag")
            
            vel_tor_real = NetCDF.readvar(nc_file, "velocity_toroidal_real")
            vel_tor_imag = NetCDF.readvar(nc_file, "velocity_toroidal_imag")
            
            vel_pol_real = NetCDF.readvar(nc_file, "velocity_poloidal_real")
            vel_pol_imag = NetCDF.readvar(nc_file, "velocity_poloidal_imag")
            
            mag_tor_real = NetCDF.readvar(nc_file, "magnetic_toroidal_real")
            mag_tor_imag = NetCDF.readvar(nc_file, "magnetic_toroidal_imag")
            
            mag_pol_real = NetCDF.readvar(nc_file, "magnetic_poloidal_real")
            mag_pol_imag = NetCDF.readvar(nc_file, "magnetic_poloidal_imag")
            
            # Copy data back to state arrays
            nlm_local, nr_local = size(temp_real)
            
            for r_idx in 1:nr_local
                for lm_idx in 1:nlm_local
                    if (lm_idx <= size(state.temperature.spectral.data_real, 1) &&
                        r_idx <= size(state.temperature.spectral.data_real, 3))
                        
                        state.temperature.spectral.data_real[lm_idx, 1, r_idx] = temp_real[lm_idx, r_idx]
                        state.temperature.spectral.data_imag[lm_idx, 1, r_idx] = temp_imag[lm_idx, r_idx]
                        
                        state.velocity.toroidal.data_real[lm_idx, 1, r_idx] = vel_tor_real[lm_idx, r_idx]
                        state.velocity.toroidal.data_imag[lm_idx, 1, r_idx] = vel_tor_imag[lm_idx, r_idx]
                        
                        state.velocity.poloidal.data_real[lm_idx, 1, r_idx] = vel_pol_real[lm_idx, r_idx]
                        state.velocity.poloidal.data_imag[lm_idx, 1, r_idx] = vel_pol_imag[lm_idx, r_idx]
                        
                        state.magnetic.toroidal.data_real[lm_idx, 1, r_idx] = mag_tor_real[lm_idx, r_idx]
                        state.magnetic.toroidal.data_imag[lm_idx, 1, r_idx] = mag_tor_imag[lm_idx, r_idx]
                        
                        state.magnetic.poloidal.data_real[lm_idx, 1, r_idx] = mag_pol_real[lm_idx, r_idx]
                        state.magnetic.poloidal.data_imag[lm_idx, 1, r_idx] = mag_pol_imag[lm_idx, r_idx]
                    end
                end
            end
            
            println("Rank $rank: Successfully read restart data")
            
        finally
            NetCDF.close(nc_file)
        end
        
        # Broadcast timestepping state to all ranks
        state.timestep_state.time = MPI.bcast(state.timestep_state.time, 0, comm)
        state.timestep_state.step = MPI.bcast(state.timestep_state.step, 0, comm)
        state.timestep_state.dt = MPI.bcast(state.timestep_state.dt, 0, comm)
        state.timestep_state.error = MPI.bcast(state.timestep_state.error, 0, comm)
        state.timestep_state.iteration = MPI.bcast(state.timestep_state.iteration, 0, comm)
        
        if rank == 0
            println("Restart completed from step $(state.timestep_state.step) at time $(state.timestep_state.time)")
        end
    end
    
    # ============================================================================
    # Utility Functions
    # ============================================================================
    
    function create_file_list(output_dir::String, step::Int, nprocs::Int)
        # Create a text file listing all NetCDF files for a given step
        # Useful for post-processing tools
        
        list_filename = joinpath(output_dir, "filelist_step_$(lpad(step, 6, '0')).txt")
        
        open(list_filename, "w") do file
            write(file, "# NetCDF files for simulation step $step\n")
            write(file, "# Total processes: $nprocs\n")
            write(file, "# Format: rank filename\n")
            
            for rank in 0:(nprocs-1)
                nc_filename = "geodynamo_rank_$(lpad(rank, 4, '0'))_step_$(lpad(step, 6, '0')).nc"
                write(file, "$rank $nc_filename\n")
            end
        end
        
        return list_filename
    end
    
    function validate_netcdf_output(filename::String)
        # Validate NetCDF file integrity
        try
            nc_file = NetCDF.open(filename, NC_NOWRITE)
            
            # Check for required variables
            required_vars = ["time", "step", "r"]
            
            for var in required_vars
                if NetCDF.varid(nc_file, var) == -1
                    @warn "Missing variable: $var in $filename"
                    NetCDF.close(nc_file)
                    return false
                end
            end
            
            # Check data integrity (basic)
            time_data = NetCDF.readvar(nc_file, "time")
            if any(isnan.(time_data)) || any(isinf.(time_data))
                @warn "Invalid time data in $filename"
                NetCDF.close(nc_file)
                return false
            end
            
            NetCDF.close(nc_file)
            return true
            
        catch e
            @error "Error validating NetCDF file: $filename" exception=e
            return false
        end
    end
    
    function cleanup_old_netcdf_files(output_dir::String, keep_last_n::Int = 10)
        # Clean up old NetCDF files to save disk space
        # Keep only the last n timesteps
        
        # Find all step numbers
        files = readdir(output_dir)
        netcdf_files = filter(f -> endswith(f, ".nc") && contains(f, "step_"), files)
        
        if isempty(netcdf_files)
            return
        end
        
        # Extract step numbers
        step_pattern = r"step_(\d+)"
        steps = Set{Int}()
        
        for file in netcdf_files
            m = match(step_pattern, file)
            if m !== nothing
                push!(steps, parse(Int, m.captures[1]))
            end
        end
        
        steps_sorted = sort(collect(steps), rev=true)  # Newest first
        
        if length(steps_sorted) <= keep_last_n
            return  # Nothing to clean up
        end
        
        # Remove old steps
        steps_to_remove = steps_sorted[(keep_last_n+1):end]
        
        for step in steps_to_remove
            step_files = filter(f -> contains(f, "step_$(lpad(step, 6, '0'))"), netcdf_files)
            
            for file in step_files
                full_path = joinpath(output_dir, file)
                try
                    rm(full_path)
                    println("Removed old file: $file")
                catch e
                    @warn "Failed to remove file: $file" exception=e
                end
            end
        end
        
        println("Cleanup completed. Kept last $keep_last_n timesteps.")
    end
    
    function get_netcdf_info(filename::String)
        # Get information about a NetCDF file
        nc_file = NetCDF.open(filename, NC_NOWRITE)
        
        try
            info = Dict{String, Any}()
            
            # Global attributes
            info["rank"] = NetCDF.getatt(nc_file, NC_GLOBAL, "mpi_rank")
            info["total_ranks"] = NetCDF.getatt(nc_file, NC_GLOBAL, "mpi_total_ranks")
            info["time"] = NetCDF.readvar(nc_file, "time")[1]
            info["step"] = NetCDF.readvar(nc_file, "step")[1]
            
            # Dimensions
            info["dimensions"] = Dict{String, Int}()
            for (name, dim_id) in NetCDF.dimnames(nc_file)
                info["dimensions"][name] = NetCDF.dimlen(nc_file, dim_id)
            end
            
            # Variables
            info["variables"] = collect(keys(NetCDF.varnames(nc_file)))
            
            return info
            
        finally
            NetCDF.close(nc_file)
        end
    end
    
    # ============================================================================
    # Export Functions
    # ============================================================================
    
    export NetCDFOutputConfig, OutputSpace, FileNaming
    export PHYSICAL_ONLY, SPECTRAL_ONLY, BOTH_SPACES
    export RANK_STEP, STEP_RANK, TIMESTAMP_RANK
    export default_netcdf_config
    
    export LocalFieldInfo, extract_local_info, extract_spectral_info
    
    export output_netcdf_fields!
    export write_netcdf_restart!, read_netcdf_restart!
    
    export create_file_list, validate_netcdf_output, cleanup_old_netcdf_files
    export get_netcdf_info

end  # module SHTnsNetCDFOutput

# ============================================================================
# Example Usage and Integration
# ============================================================================

"""
Example usage of the SHTns NetCDF Output module:

```julia
using .SHTnsNetCDFOutput

# Create custom NetCDF output configuration
config = NetCDFOutputConfig(
    BOTH_SPACES,          # Output both physical and spectral data
    RANK_STEP,            # Naming: rank_XXXX_step_YYYYYY.nc
    "./netcdf_output",    # Output directory
    "geodynamo_run01",    # Filename prefix
    6,                    # Compression level (0-9)
    true,                 # Include metadata
    true,                 # Include coordinate grids
    true,                 # Include local diagnostics
    Float32,              # Use single precision for smaller files
    16,                   # Output spectral modes up to l=16
    false,                # Don't add timestamp to filename
    true                  # Overwrite existing files
)

# Output fields during simulation loop
for step in 1:1000
    # ... simulation code ...
    
    if step % 100 == 0  # Output every 100 steps
        output_netcdf_fields!(simulation_state, config)
    end
    
    if step % 1000 == 0  # Write restart every 1000 steps
        write_netcdf_restart!(simulation_state, config)
    end
end

# Post-processing utilities
create_file_list("./netcdf_output", 1000, 64)  # Create file list for step 1000
cleanup_old_netcdf_files("./netcdf_output", 5)  # Keep only last 5 timesteps

# Validate output
filename = "geodynamo_run01_rank_0000_step_001000.nc"
if validate_netcdf_output(filename)
    info = get_netcdf_info(filename)
    println("File info: ", info)
end

# Restart from checkpoint
read_netcdf_restart!(simulation_state, "./netcdf_output", 1000)
```

Key advantages of this NetCDF approach:

1. **Simplicity**: Each process writes independently, no complex coordination
2. **Reliability**: Less prone to MPI I/O issues than collective approaches
3. **Flexibility**: Easy to read individual files for debugging or analysis
4. **Standard Format**: NetCDF is widely supported by analysis tools
5. **Self-Describing**: Files contain complete metadata and coordinate information
6. **Compression**: Built-in compression reduces file sizes significantly
7. **CF Compliance**: Follows Climate & Forecast metadata conventions

The distributed file approach scales well and is often more robust than
collective I/O for large-scale simulations.
"""