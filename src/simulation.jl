# ============================================================================
# Main Simulation Driver with SHTns
# ============================================================================

# module GeodynamoSimulation
# using MPI
# using PencilArrays
# using PencilFFTs
# using HDF5
# using SHTnsSpheres
# using ..Parameters
# using ..PencilSetup
# using ..SHTnsSetup
# using ..VariableTypes
# using ..SHTnsTransforms
# using ..Timestepping
# using ..Velocity
# using ..MagneticField
# using ..Temperature
    
# Main simulation state with SHTns
struct SHTnsSimulationState{T}
    # Field variables
    velocity::SHTnsVelocityFields{T}
    magnetic::SHTnsMagneticFields{T}
    temperature::SHTnsTemperatureField{T}
    
    # Geometric data
    shtns_config::SHTnsConfig
    radial_domain::RadialDomain
    inner_core_domain::RadialDomain
    
    # Pencil decomposition
    pencils::Tuple{Pencil{3}, Pencil{3}, Pencil{3}}
    pencil_spec::Pencil{3}
    transforms::NamedTuple
    
    # Timestepping
    timestep_state::TimestepState
    implicit_matrices::Dict{Symbol, SHTnsImplicitMatrices{T}}
    
    # I/O
    output_counter::Int
end

function initialize_shtns_simulation(::Type{T} = Float64) where T
    # Initialize MPI first
    MPI.Init()
    
    # Create SHTns configuration
    shtns_config = create_parallel_shtns_config(comm)
    
    # Initialize pencil decomposition with SHTns grid
    pencil_θ, pencil_φ, pencil_r, pencil_spec = create_pencil_topology(shtns_config)
    
    # Create transform operations
    transforms = create_transforms(pencil_θ, pencil_φ, pencil_r, pencil_spec)
    
    # Initialize geometric data
    radial_domain = create_radial_domain(pencil_r)
    inner_core_domain = create_radial_domain(pencil_r)  # Modify for inner core
    
    # Create field variables
    pencils = (pencil_θ, pencil_φ, pencil_r)
    velocity = create_shtns_velocity_fields(T, shtns_config, radial_domain, pencils, pencil_spec)
    magnetic = create_shtns_magnetic_fields(T, shtns_config, radial_domain, 
                                            inner_core_domain, pencils, pencil_spec)
    temperature = create_shtns_temperature_field(T, shtns_config, radial_domain, pencils, pencil_spec)
    
    # Initialize timestepping
    timestep_state = TimestepState(d_time, d_timestep, 0, 0, Inf, false)
    
    # Create implicit matrices for each equation
    implicit_matrices = Dict{Symbol, SHTnsImplicitMatrices{T}}()
    implicit_matrices[:velocity] = create_shtns_timestepping_matrices(
        shtns_config, radial_domain, d_E, d_timestep)
    implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(
        shtns_config, radial_domain, 1.0/d_Pm, d_timestep)
    implicit_matrices[:temperature] = create_shtns_timestepping_matrices(
        shtns_config, radial_domain, 1.0/d_Pr, d_timestep)
    
    return SHTnsSimulationState{T}(velocity, magnetic, temperature, shtns_config,
                                    radial_domain, inner_core_domain,
                                    pencils, pencil_spec, transforms, timestep_state,
                                    implicit_matrices, 0)
end

function run_shtns_simulation!(state::SHTnsSimulationState{T}) where T
    rank = MPI.Comm_rank(comm)
    
    if rank == 0
        println("Starting geodynamo simulation with SHTnsSpheres...")
        println("SHTns Grid: $(state.shtns_config.nlat) × $(state.shtns_config.nlon) × $(i_N)")
        println("Spectral modes: $(state.shtns_config.nlm)")
        println("lmax: $(state.shtns_config.lmax), mmax: $(state.shtns_config.mmax)")
    end
    
    # Initialize fields with some perturbation
    initialize_fields!(state)
    
    # Main timestepping loop
    while state.timestep_state.step < i_maxtstep && 
            state.timestep_state.time < 1.0
        
        # Predictor-corrector iterations
        state.timestep_state.iteration = 1
        state.timestep_state.error = Inf
        state.timestep_state.converged = false
        
        # Store previous state for error computation
        prev_velocity = deepcopy(state.velocity.toroidal)
        prev_magnetic = deepcopy(state.magnetic.toroidal)
        prev_temperature = deepcopy(state.temperature.spectral)
        
        while state.timestep_state.iteration <= 10 && 
                state.timestep_state.error > d_dterr
            
            # Compute nonlinear terms
            compute_all_nonlinear_terms!(state)
            
            # Timestepping (predictor or corrector)
            if state.timestep_state.iteration == 1
                predictor_step!(state)
            else
                corrector_step!(state)
            end
            
            # Compute convergence error
            error_vel  = compute_timestep_error(state.velocity.toroidal, prev_velocity)
            error_mag  = compute_timestep_error(state.magnetic.toroidal, prev_magnetic)
            error_temp = compute_timestep_error(state.temperature.spectral, prev_temperature)
            
            state.timestep_state.error = max(error_vel, error_mag, error_temp)
            
            if state.timestep_state.error < d_dterr
                state.timestep_state.converged = true
                break
            end
            
            # Update previous state
            prev_velocity .= state.velocity.toroidal
            prev_magnetic .= state.magnetic.toroidal
            prev_temperature .= state.temperature.spectral
            
            state.timestep_state.iteration += 1
        end
        
        # Advance time
        state.timestep_state.time += state.timestep_state.dt
        state.timestep_state.step += 1
        
        # Adaptive timestepping
        compute_cfl_timestep!(state)
        
        # Update implicit matrices if timestep changed
        if abs(state.timestep_state.dt - d_timestep) > 1e-10
            update_implicit_matrices!(state)
        end
        
        # Output
        if state.timestep_state.step % i_save_rate2 == 0
            output_shtns_fields!(state)
            
            if rank == 0
                println("Step: $(state.timestep_state.step), " *
                        "Time: $(state.timestep_state.time), " *
                        "dt: $(state.timestep_state.dt), " *
                        "Error: $(state.timestep_state.error)")
            end
        end
    end
    
    if rank == 0
        println("SHTns simulation completed!")
    end
end

function initialize_fields!(state::SHTnsSimulationState{T}) where T
    # Initialize with some random perturbation for onset
    
    # Temperature: conductive profile + small perturbation
    for lm_idx in 1:state.temperature.spectral.nlm
        l = state.temperature.spectral.config.l_values[lm_idx]
        m = state.temperature.spectral.config.m_values[lm_idx]
        
        for r_idx in state.temperature.spectral.local_radial_range
            if r_idx <= size(state.temperature.spectral.data_real, 3)
                if l == 0 && m == 0
                    # Conductive profile
                    r = state.radial_domain.r[r_idx, 4]
                    state.temperature.spectral.data_real[lm_idx, 1, r_idx] = 1.0 - r
                elseif l <= 4 && l >= 1
                    # Small perturbation
                    state.temperature.spectral.data_real[lm_idx, 1, r_idx] = 1e-3 * (rand() - 0.5)
                    if m > 0
                        state.temperature.spectral.data_imag[lm_idx, 1, r_idx] = 1e-3 * (rand() - 0.5)
                    end
                end
            end
        end
    end
    
    # Velocity: start with zero
    fill!(state.velocity.toroidal.data_real, zero(T))
    fill!(state.velocity.toroidal.data_imag, zero(T))
    fill!(state.velocity.poloidal.data_real, zero(T))
    fill!(state.velocity.poloidal.data_imag, zero(T))
    
    # Magnetic field: dipole + small perturbation
    for lm_idx in 1:state.magnetic.toroidal.nlm
        l = state.magnetic.toroidal.config.l_values[lm_idx]
        m = state.magnetic.toroidal.config.m_values[lm_idx]
        
        for r_idx in state.magnetic.toroidal.local_radial_range
            if r_idx <= size(state.magnetic.toroidal.data_real, 3)
                if l == 1 && m == 0
                    # Dipole field
                    r = state.radial_domain.r[r_idx, 4]
                    state.magnetic.poloidal.data_real[lm_idx, 1, r_idx] = r^2 * (1.0 - r)
                elseif l <= 3 && l >= 1
                    # Small perturbation
                    state.magnetic.toroidal.data_real[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    state.magnetic.poloidal.data_real[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    if m > 0
                        state.magnetic.toroidal.data_imag[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                        state.magnetic.poloidal.data_imag[lm_idx, 1, r_idx] = 1e-4 * (rand() - 0.5)
                    end
                end
            end
        end
    end
end

function compute_all_nonlinear_terms!(state::SHTnsSimulationState{T}) where T
    # Compute nonlinear terms for all equations using SHTns transforms
    compute_velocity_nonlinear!(state.velocity, state.temperature, 
                                nothing, state.magnetic)  # No composition for now
    
    compute_magnetic_nonlinear!(state.magnetic, state.velocity, 0.0)  # No inner core rotation for now
    
    compute_temperature_nonlinear!(state.temperature, state.velocity)
end

function predictor_step!(state::SHTnsSimulationState{T}) where T
    # Predictor step for all fields using SHTns matrices
    
    # Velocity (toroidal component)
    rhs_tor = similar(state.velocity.toroidal)
    apply_explicit_operator!(rhs_tor, state.velocity.toroidal, 
                            state.velocity.nl_toroidal, state.radial_domain,
                            d_E, state.timestep_state.dt)
    solve_implicit_step!(state.velocity.toroidal, rhs_tor, 
                        state.implicit_matrices[:velocity])
    
    # Velocity (poloidal component)
    rhs_pol = similar(state.velocity.poloidal)
    apply_explicit_operator!(rhs_pol, state.velocity.poloidal, 
                            state.velocity.nl_poloidal, state.radial_domain,
                            d_E, state.timestep_state.dt)
    solve_implicit_step!(state.velocity.poloidal, rhs_pol, 
                        state.implicit_matrices[:velocity])
    
    # Magnetic field (toroidal)
    rhs_mag_tor = similar(state.magnetic.toroidal)
    apply_explicit_operator!(rhs_mag_tor, state.magnetic.toroidal, 
                            state.magnetic.nl_toroidal, state.radial_domain,
                            1.0/d_Pm, state.timestep_state.dt)
    solve_implicit_step!(state.magnetic.toroidal, rhs_mag_tor, 
                        state.implicit_matrices[:magnetic])
    
    # Magnetic field (poloidal)
    rhs_mag_pol = similar(state.magnetic.poloidal)
    apply_explicit_operator!(rhs_mag_pol, state.magnetic.poloidal, 
                            state.magnetic.nl_poloidal, state.radial_domain,
                            1.0/d_Pm, state.timestep_state.dt)
    solve_implicit_step!(state.magnetic.poloidal, rhs_mag_pol, 
                        state.implicit_matrices[:magnetic])
    
    # Temperature
    rhs_temp = similar(state.temperature.spectral)
    apply_explicit_operator!(rhs_temp, state.temperature.spectral, 
                            state.temperature.nonlinear, state.radial_domain,
                            1.0/d_Pr, state.timestep_state.dt)
    solve_implicit_step!(state.temperature.spectral, rhs_temp, 
                        state.implicit_matrices[:temperature])
end

function corrector_step!(state::SHTnsSimulationState{T}) where T
    # Corrector step - similar to predictor but with time-averaged nonlinear terms
    # For simplicity, using same implementation as predictor
    predictor_step!(state)
end

function compute_cfl_timestep!(state::SHTnsSimulationState{T}) where T
    # Compute CFL-limited timestep based on velocity magnitudes
    
    # Convert velocity to physical space for analysis
    shtns_vector_synthesis!(state.velocity.toroidal, state.velocity.poloidal, state.velocity.velocity)
    
    max_velocity = zero(Float64)
    
    # Find maximum velocity
    for r_idx in state.velocity.velocity.r_component.local_radial_range
        if r_idx <= size(state.velocity.velocity.r_component.data_r, 3)
            for j_phi in 1:state.velocity.velocity.r_component.nlon
                for i_theta in 1:state.velocity.velocity.r_component.nlat
                    if (i_theta <= size(state.velocity.velocity.r_component.data_r, 1) && 
                        j_phi <= size(state.velocity.velocity.r_component.data_r, 2))
                        
                        u_r = state.velocity.velocity.r_component.data_r[i_theta, j_phi, r_idx]
                        u_θ = state.velocity.velocity.θ_component.data_r[i_theta, j_phi, r_idx]
                        u_φ = state.velocity.velocity.φ_component.data_r[i_theta, j_phi, r_idx]
                        
                        u_mag = sqrt(u_r^2 + u_θ^2 + u_φ^2)
                        max_velocity = max(max_velocity, u_mag)
                    end
                end
            end
        end
    end
    
    # Global maximum across all processes
    global_max_vel = MPI.Allreduce(max_velocity, MPI.MAX, comm)
    
    # Compute grid spacing (approximate)
    dr_min = minimum(diff(state.radial_domain.r[:, 4]))
    dtheta_min = π / state.shtns_config.nlat
    dphi_min = 2π / state.shtns_config.nlon
    dx_min = min(dr_min, dtheta_min, dphi_min)
    
    # CFL condition
    if global_max_vel > 1e-10
        dt_cfl = d_courant * dx_min / global_max_vel
        state.timestep_state.dt = min(state.timestep_state.dt, dt_cfl, d_timestep)
    end
end

function update_implicit_matrices!(state::SHTnsSimulationState{T}) where T
    # Update implicit matrices with new timestep
    dt = state.timestep_state.dt
    
    state.implicit_matrices[:velocity] = create_shtns_timestepping_matrices(
        state.shtns_config, state.radial_domain, d_E, dt)
    state.implicit_matrices[:magnetic] = create_shtns_timestepping_matrices(
        state.shtns_config, state.radial_domain, 1.0/d_Pm, dt)
    state.implicit_matrices[:temperature] = create_shtns_timestepping_matrices(
        state.shtns_config, state.radial_domain, 1.0/d_Pr, dt)
end

# function output_shtns_fields!(state::SHTnsSimulationState{T}) where T
#     # Output fields to HDF5 files using collective I/O
#     filename = "geodynamo_shtns_step_$(lpad(state.timestep_state.step, 6, '0')).h5"
    
#     rank = MPI.Comm_rank(comm)
    
#     # Convert spectral fields to physical space for output
#     shtns_spectral_to_physical!(state.temperature.spectral, state.temperature.temperature)
#     shtns_vector_synthesis!(state.velocity.toroidal, state.velocity.poloidal, state.velocity.velocity)
#     shtns_vector_synthesis!(state.magnetic.toroidal, state.magnetic.poloidal, state.magnetic.magnetic)
    
#     if rank == 0
#         println("Writing output file: $filename")
#     end
    
#     # Use HDF5 collective I/O for parallel output
#     # This is a placeholder - would implement proper parallel HDF5 output
    
#     state.output_counter += 1
# end

# Main entry point
function run_shtns_geodynamo_simulation()
    state = initialize_shtns_simulation(Float64)
    run_shtns_simulation!(state)
    MPI.Finalize()
end

export SHTnsSimulationState, initialize_shtns_simulation, run_shtns_simulation!
export run_shtns_geodynamo_simulation
