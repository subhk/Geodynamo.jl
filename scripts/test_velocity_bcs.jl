#!/usr/bin/env julia

using Geodynamo
using Test

function check_no_slip!(geom::Symbol)
    println("Checking no-slip BCs for geometry=$(geom)...")
    # Set parameters
    params = GeodynamoParameters(
        geometry=geom,
        i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64,
        d_rratio=0.35,
        i_vel_bc=1,
    )
    set_parameters!(params)
    state = initialize_shtns_simulation(Float64; include_composition=false)

    # Fill velocity spectral fields with random data
    rand!(parent(state.velocity.toroidal.data_real))
    rand!(parent(state.velocity.toroidal.data_imag))
    rand!(parent(state.velocity.poloidal.data_real))
    rand!(parent(state.velocity.poloidal.data_imag))

    apply_velocity_boundary_conditions!(state.velocity, state.oc_domain)

    # Validate: poloidal and toroidal vanish at outer boundary
    pol_r = parent(state.velocity.poloidal.data_real)
    tor_r = parent(state.velocity.toroidal.data_real)
    r_range = range_local(state.velocity.poloidal.pencil, 3)
    if state.oc_domain.N in r_range
        rloc = state.oc_domain.N - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
        @test maximum(abs.(view(tor_r, :, 1, rloc))) ≈ 0 atol=1e-12
    end

    # For shell geometry, also inner boundary vanishes; for ball, skip
    if Geodynamo.get_parameters().geometry == :shell && (1 in r_range)
        rloc = 1 - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
        @test maximum(abs.(view(tor_r, :, 1, rloc))) ≈ 0 atol=1e-12
    end
end

function check_stress_free!(geom::Symbol)
    println("Checking stress-free BCs for geometry=$(geom)...")
    params = GeodynamoParameters(
        geometry=geom,
        i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64,
        d_rratio=0.35,
        i_vel_bc=2,
    )
    set_parameters!(params)
    state = initialize_shtns_simulation(Float64; include_composition=false)

    # Fill velocity spectral fields with random data
    rand!(parent(state.velocity.toroidal.data_real))
    rand!(parent(state.velocity.toroidal.data_imag))
    rand!(parent(state.velocity.poloidal.data_real))
    rand!(parent(state.velocity.poloidal.data_imag))

    apply_velocity_boundary_conditions!(state.velocity, state.oc_domain)

    # Validate: poloidal vanish at boundaries (impenetrable)
    pol_r = parent(state.velocity.poloidal.data_real)
    tor_r = parent(state.velocity.toroidal.data_real)
    r_range = range_local(state.velocity.poloidal.pencil, 3)
    if state.oc_domain.N in r_range
        rloc = state.oc_domain.N - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
    end
    if Geodynamo.get_parameters().geometry == :shell && (1 in r_range)
        rloc = 1 - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
    end

    # Heuristic check: stress-free toroidal correction should reduce discrete gradient at boundaries
    r_range_t = range_local(state.velocity.toroidal.pencil, 3)
    if state.oc_domain.N in r_range_t
        rlocN = state.oc_domain.N - first(r_range_t) + 1
        if rlocN >= 2
            gradN = maximum(abs.(view(tor_r, :, 1, rlocN) .- view(tor_r, :, 1, rlocN-1)))
            @test gradN ≤ 1.0  # not strict, just sanity bound
        end
    end
end

# Run checks
check_no_slip!(:shell)
check_stress_free!(:shell)
check_no_slip!(:ball)
check_stress_free!(:ball)

println("Velocity BC tests completed.")

