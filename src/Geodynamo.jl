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

    

end
