#!/usr/bin/env julia

using Geodynamo
using Test

function main()
    # Minimal setup
    params = GeodynamoParameters(geometry=:ball, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64,
                                 d_E=1e-4, d_Pr=1.0, d_Pm=1.0, d_Ra=1e6, i_B=1)
    set_parameters!(params)

    state = initialize_simulation(Float64; include_composition=false)
    initialize_fields!(state)

    # Baseline compute
    compute_vorticity_spectral_full!(state.velocity, state.oc_domain)
    base_tor = copy(parent(state.velocity.vort_toroidal.data_real))
    base_pol = copy(parent(state.velocity.vort_poloidal.data_real))

    # Zero and recompute with workspace
    fill!(parent(state.velocity.vort_toroidal.data_real), 0.0)
    fill!(parent(state.velocity.vort_toroidal.data_imag), 0.0)
    fill!(parent(state.velocity.vort_poloidal.data_real), 0.0)
    fill!(parent(state.velocity.vort_poloidal.data_imag), 0.0)

    ws = create_velocity_workspace(Float64, state.oc_domain.N)
    set_velocity_workspace!(ws)
    compute_vorticity_spectral_full!(state.velocity, state.oc_domain)

    # Compare
    @test isapprox(parent(state.velocity.vort_toroidal.data_real), base_tor; rtol=1e-10, atol=1e-10)
    @test isapprox(parent(state.velocity.vort_poloidal.data_real), base_pol; rtol=1e-10, atol=1e-10)

    println("Workspace equivalence test passed.")
end

main()
