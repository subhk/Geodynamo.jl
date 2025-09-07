#!/usr/bin/env julia

# ============================================================================
# Test Script for MPI/PencilArrays/PencilFFTs Integration in Time Stepping
# ============================================================================

using Test
using MPI

# Include the necessary modules from the local source
push!(LOAD_PATH, "src")

using Geodynamo
using SHTnsKit
using PencilArrays
using PencilFFTs

"""
    test_mpi_initialization()

Test basic MPI initialization and communication.
"""
function test_mpi_initialization()
    @testset "MPI Initialization" begin
        # Test MPI setup
        if !MPI.Initialized()
            MPI.Init()
        end
        
        comm = get_comm()
        rank = get_rank() 
        nprocs = get_nprocs()
        
        @test comm isa MPI.Comm
        @test rank >= 0
        @test nprocs >= 1
        
        println("MPI Test: Rank $rank of $nprocs processes")
    end
end

"""
    test_pencil_decomposition()

Test PencilArrays decomposition setup.
"""
function test_pencil_decomposition()
    @testset "PencilArrays Decomposition" begin
        # Create a small test configuration
        lmax = 4
        nlat = 8
        nlon = 12
        nr = 10
        
        config = create_shtnskit_config(lmax=lmax, nlat=nlat, nlon=nlon)
        
        @test config isa SHTnsKitConfig
        @test config.nlat == nlat
        @test config.nlon == nlon
        @test config.lmax == lmax
        
        # Validate pencil decomposition
        validate_pencil_decomposition(config)
        
        println("Pencil decomposition test passed")
    end
end

"""
    test_timestepping_functions()

Test key timestepping functions for MPI compatibility.
"""
function test_timestepping_functions()
    @testset "Timestepping Functions" begin
        # Create minimal test setup
        T = Float64
        lmax = 3
        nlat = 8 
        nlon = 8
        nr = 6
        
        config = create_shtnskit_config(lmax=lmax, nlat=nlat, nlon=nlon)
        domain = create_radial_domain(nr)
        
        # Create test spectral fields
        spec_field1 = create_shtns_spectral_field(T, config, domain, config.pencils.spec)
        spec_field2 = create_shtns_spectral_field(T, config, domain, config.pencils.spec)
        
        # Initialize with some test data
        fill!(parent(spec_field1.data_real), 1.0)
        fill!(parent(spec_field1.data_imag), 0.5)
        fill!(parent(spec_field2.data_real), 1.1)
        fill!(parent(spec_field2.data_imag), 0.6)
        
        # Test error computation
        error = compute_timestep_error(spec_field1, spec_field2)
        @test error >= 0.0
        @test isfinite(error)
        
        # Test synchronization functions
        synchronize_pencil_transforms!(spec_field1)
        validate_mpi_consistency!(spec_field1)
        
        println("Timestepping functions test passed")
    end
end

"""
    test_pencilffts_integration()

Test PencilFFTs integration.
"""
function test_pencilffts_integration()
    @testset "PencilFFTs Integration" begin
        # Create test configuration
        lmax = 3
        nlat = 8
        nlon = 8
        
        config = create_shtnskit_config(lmax=lmax, nlat=nlat, nlon=nlon)
        
        # Test FFT plan creation
        @test haskey(config.fft_plans, :phi_forward) || haskey(config.fft_plans, :fallback)
        
        # Test FFT optimization
        optimize_fft_performance!(config)
        
        println("PencilFFTs integration test passed")
    end
end

"""
    run_all_tests()

Run all tests with proper MPI setup.
"""
function run_all_tests()
    try
        test_mpi_initialization()
        test_pencil_decomposition() 
        test_timestepping_functions()
        test_pencilffts_integration()
        
        rank = get_rank()
        if rank == 0
            println("\n✅ All tests passed successfully!")
            println("MPI, PencilArrays, and PencilFFTs integration is working correctly.")
        end
        
    catch e
        rank = get_rank()
        println("❌ Test failed on rank $rank: $e")
        rethrow(e)
    finally
        if MPI.Initialized() 
            MPI.Finalize()
        end
    end
end

# Run tests if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end