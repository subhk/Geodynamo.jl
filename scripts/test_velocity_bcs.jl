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

    # Validate: poloidal vanish at boundaries (impenetrable) and stress metric small
    pol_r = parent(state.velocity.poloidal.data_real)
    tor_r = parent(state.velocity.toroidal.data_real)
    r_range = range_local(state.velocity.poloidal.pencil, 3)
    # Helper to compute S metric
    function stress_metric_at_boundary(pol_arr)
        lm_range = range_local(state.velocity.poloidal.pencil, 1)
        r_range  = range_local(state.velocity.poloidal.pencil, 3)
        nr = state.oc_domain.N
        # Aggregate maximum absolute S across local modes
        S_inner = 0.0
        S_outer = 0.0
        for lm_idx in lm_range
            if lm_idx <= state.velocity.poloidal.nlm
                local_lm = lm_idx - first(lm_range) + 1
                prof = extract_local_radial_profile(pol_arr, local_lm, nr, r_range)
                # dP/dr
                dP = similar(prof)
                Geodynamo.apply_derivative_matrix!(dP, state.velocity.dr_matrix, prof)
                # Q = r dP/dr
                Q = similar(prof)
                for i in 1:nr
                    Q[i] = state.oc_domain.r[i,4] * dP[i]
                end
                dQ = similar(prof)
                Geodynamo.apply_derivative_matrix!(dQ, state.velocity.dr_matrix, Q)
                S_in  = nr >= 1 ? (state.oc_domain.r[1,4] > 0 ? dQ[1] / state.oc_domain.r[1,4] : dQ[1]) : 0.0
                S_out = dQ[nr] / state.oc_domain.r[nr,4]
                S_inner = max(S_inner, abs(S_in))
                S_outer = max(S_outer, abs(S_out))
            end
        end
        return S_inner, S_outer
    end
    if state.oc_domain.N in r_range
        rloc = state.oc_domain.N - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
        Sin, Sout = stress_metric_at_boundary(pol_r)
        @test Sout ≤ 1e-6
    end
    if Geodynamo.get_parameters().geometry == :shell && (1 in r_range)
        rloc = 1 - first(r_range) + 1
        @test maximum(abs.(view(pol_r, :, 1, rloc))) ≈ 0 atol=1e-12
        Sin, Sout = stress_metric_at_boundary(pol_r)
        @test Sin ≤ 1e-6
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
