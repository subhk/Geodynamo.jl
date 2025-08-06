module Geodynamo

    using LinearAlgebra
    using SparseArrays
    using MPI
    using PencilArrays
    using PencilFFTs
    using HDF5
    using StaticArrays
    using SHTnsSpheres


    # exports shtns_config.jl
    export SHTnsConfig, create_shtns_config, create_parallel_shtns_config

    # exports pencil_decomps.jl
    export get_comm, create_pencil_topology, create_transpose_plans


    # exports field.jl
    export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField
    export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
    export create_shtns_vector_field, create_radial_domain
    export get_local_range, get_local_indices, local_data_size, get_local_data


    # export shtns_transforms.jl
    export shtns_spectral_to_physical!, shtns_physical_to_spectral!
    export shtns_vector_synthesis!, shtns_vector_analysis!

    # exports linear_algebra.jl
    export BandedMatrix, create_derivative_matrix, create_radial_laplacian, apply_banded_matrix!

    # exports timestep.jl
    export TimestepState, SHTnsImplicitMatrices, create_shtns_timestepping_matrices
    export apply_explicit_operator!, solve_implicit_step!, compute_timestep_error

    # exports velocity.jl
    export SHTnsVelocityFields, create_shtns_velocity_fields, compute_velocity_nonlinear!

    # exports magnetic.jl
    export SHTnsMagneticFields, create_shtns_magnetic_fields, compute_magnetic_nonlinear!

    # exports thermal.jl
    export SHTnsTemperatureField, create_shtns_temperature_field, compute_temperature_nonlinear!




    include("fields.jl")
    include("shtns_config.jl")
    include("shtns_transforms.jl")
    include("linear_algebra.jl")
    include("pencil_decomps.jl")
    include("timestep.jl")
    include("velocity.jl")
    include("magnetic.jl")
    include("thermal.jl")

    

end
