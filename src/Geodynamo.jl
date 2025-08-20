module Geodynamo

    using LinearAlgebra
    using SparseArrays
    using MPI
    using PencilArrays
    using PencilFFTs
    using HDF5
    using StaticArrays
    using SHTnsKit

    # exports shtns_config.jl
    export SHTnsConfig, create_shtns_config
    export get_mode_index, is_mode_local, get_local_modes
    export validate_config, print_shtns_config_summary
    export create_pencil_topology_shtns, create_shtns_pencils

    # exports pencil_decomps.jl
    export get_comm, get_rank, get_nprocs
    export create_pencil_topology, create_transpose_plans
    export transpose_with_timer!, print_transpose_statistics
    export analyze_load_balance, estimate_memory_usage
    export create_pencil_array, synchronize_halos!
    export print_pencil_info, optimize_communication_order
    export ENABLE_TIMING


    # exports field.jl
    export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
    export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
    export create_shtns_vector_field, create_radial_domain
    export get_local_range, get_local_indices, local_data_size, get_local_data

    # export shtns_transforms.jl
    export shtns_spectral_to_physical!, shtns_physical_to_spectral!
    export shtns_vector_synthesis!, shtns_vector_analysis!
    export batch_spectral_to_physical!
    export shtns_compute_gradient!
    export get_transform_statistics, print_transform_statistics
    export clear_transform_cache!
    export SHTnsTransformManager, get_transform_manager

    # exports linear_algebra.jl
    export BandedMatrix, create_derivative_matrix, create_radial_laplacian
    export apply_derivative_matrix!, apply_banded_matrix!

    # exports timestep.jl
    export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
    export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error

    # exports velocity.jl
    export SHTnsVelocityFields, create_shtns_velocity_fields
    export compute_velocity_nonlinear!, compute_vorticity_spectral_full!
    export compute_kinetic_energy, compute_reynolds_stress
    export zero_velocity_work_arrays!
    export apply_velocity_boundary_conditions!, add_thermal_buoyancy_force!
    export add_buoyancy_force!, add_lorentz_force!, validate_velocity_configuration

    # exports magnetic.jl
    export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!
    export compute_current_density_spectral!

    # exports thermal.jl
    export SHTnsTemperatureField, create_shtns_temperature_field
    export compute_temperature_nonlinear!
    export compute_nusselt_number, compute_thermal_energy
    export compute_surface_flux, get_temperature_statistics
    export zero_temperature_work_arrays!
    export set_temperature_ic!, set_boundary_conditions!, set_internal_heating!
    export batch_transform_to_physical!, apply_temperature_boundary_conditions_spectral!

    # exports compositional.jl
    export SHTnsCompositionField, create_shtns_composition_field
    export compute_composition_nonlinear!
    export compute_composition_rms, compute_composition_energy
    export get_composition_statistics, zero_composition_work_arrays!
    export set_composition_ic!, set_composition_boundary_conditions!
    export apply_composition_boundary_conditions_spectral!

    # exports simulation.jl
    export SHTnsSimulationState, initialize_shtns_simulation, run_shtns_simulation!
    export run_shtns_geodynamo_simulation

    # exports outputs_writer.jl
    export OutputConfig, FieldInfo, TimeTracker
    export default_config, create_time_tracker, should_output_now, should_restart_now
    export write_fields!, write_restart!, read_restart!
    export create_shtns_aware_output_config, validate_output_compatibility
    export get_time_series, find_files_in_time_range, cleanup_old_files

    # exports spectral_to_physical.jl (from extras)
    export SpectralToPhysicalConverter
    export create_spectral_converter, load_spectral_data!, convert_to_physical!
    export compute_global_diagnostics, save_physical_fields
    export convert_spectral_file, batch_convert_directory
    export main_convert_file, main_batch_convert

    # exports optimizations.jl (unified parallelization system)
    export AdvancedThreadManager, ThreadingAccelerator, SIMDOptimizer, TaskGraph, MemoryOptimizer
    export AsyncCommManager, DynamicLoadBalancer, ParallelIOOptimizer, PerformanceMonitor
    export HybridParallelizer, CPUParallelizer, MasterParallelizer
    export create_advanced_thread_manager, create_threading_accelerator, create_simd_optimizer
    export create_task_graph, create_memory_optimizer, create_async_comm_manager
    export create_dynamic_load_balancer, create_parallel_io_optimizer, create_performance_monitor
    export create_hybrid_parallelizer, create_cpu_parallelizer, create_master_parallelizer
    export hybrid_compute_nonlinear!, compute_nonlinear!, add_task!, execute_task_graph!
    export async_write_fields!, analyze_parallel_performance, adaptive_rebalance!
    export allocate_aligned_array, deallocate_aligned_array, optimize_memory_layout!

    # exports simulation.jl (includes basic and enhanced simulation)
    export EnhancedSimulationState, initialize_enhanced_simulation, run_enhanced_simulation!
    export MasterSimulationState, initialize_master_simulation, run_master_simulation!
    export run_enhanced_geodynamo_simulation, run_master_geodynamo_simulation

    # exports parameters.jl
    export GeodynamoParameters, load_parameters, save_parameters, create_parameter_template
    export get_parameters, set_parameters!, initialize_parameters
    export @param  # Deprecated - use direct variable access instead

    # Include Parameters system first
    include("parameters.jl")

    # Include base modules in dependency order
    include("pencil_decomps.jl")
    include("shtns_config.jl")
    include("fields.jl")
    include("linear_algebra.jl")
    include("shtns_transforms.jl")
    include("timestep.jl")
    include("velocity.jl")
    include("magnetic.jl")
    include("thermal.jl")
    include("compositional.jl")
    include("outputs_writer.jl")
    include("optimizations.jl")
    include("simulation.jl")
    include("../extras/spectral_to_physical.jl")

    # Initialize parameters when module is loaded
    function __init__()
        try
            initialize_parameters()
            @info "Geodynamo.jl parameters loaded successfully"
        catch e
            @warn "Could not load parameters: $e. Using defaults."
            set_parameters!(GeodynamoParameters())
        end
    end

end
