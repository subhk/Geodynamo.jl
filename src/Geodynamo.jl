module Geodynamo

    using LinearAlgebra
    using SparseArrays
    using MPI
    using PencilArrays
    using PencilFFTs
    using HDF5
    using StaticArrays
    using SHTnsSpheres


    # export shtns_config.jl
    export SHTnsConfig, create_shtns_config, create_parallel_shtns_config

    

end
