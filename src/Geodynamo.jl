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
    export create_pencil_topology, create_transforms, comm


    # exports field.jl
    export SHTnsSpectralField, SHTnsPhysicalField, SHTnsVectorField, SHTnsTorPolField
    export RadialDomain, create_shtns_spectral_field, create_shtns_physical_field
    export create_shtns_vector_field, create_radial_domain 


    # exports linear_algebra.jl
    export BandedMatrix, create_derivative_matrix, create_radial_laplacian, apply_banded_matrix!



    include("fields.jl")
    include("shtns_config.jl")
    include("shtns_transforms.jl")
    include("linear_algebra.jl")
    include("pencil_decomps.jl")
    

end
