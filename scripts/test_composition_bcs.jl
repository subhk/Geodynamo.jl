#!/usr/bin/env julia

using Geodynamo
using Test

function test_fixed_and_no_flux()
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    dom = Geodynamo.create_radial_domain(i_N)
    comp = Geodynamo.create_shtns_composition_field(Float64, cfg, dom)

    # Set BCs: inner fixed=1.0, outer no-flux
    Geodynamo.set_composition_boundary_conditions!(comp, :fixed, :no_flux; value_inner=1.0, value_outer=0.0)
    Geodynamo.apply_composition_boundary_conditions_spectral!(comp, dom)

    spec_r = parent(comp.spectral.data_real)
    r_range = Geodynamo.get_local_range(comp.spectral.pencil, 3)
    if 1 in r_range
        lr = 1 - first(r_range) + 1
        @test spec_r[1,1,lr] ≈ 1.0 atol=1e-12
    end
    if dom.N in r_range
        # Check derivative ~ 0 at outer boundary
        dr = Geodynamo.create_derivative_matrix(1, dom)
        prof = zeros(Float64, dom.N)
        for r in 1:dom.N
            if r in r_range
                lr2 = r - first(r_range) + 1
                if lr2 <= size(spec_r, 3)
                    prof[r] = spec_r[1,1,lr2]
                end
            end
        end
        d = zeros(Float64, dom.N)
        Geodynamo.apply_derivative_matrix!(d, dr, prof)
        @test abs(d[end]) ≤ 1e-8
    end
end

function test_no_flux_both_shell()
    params = GeodynamoParameters(geometry=:shell, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    dom = Geodynamo.create_radial_domain(i_N)
    comp = Geodynamo.create_shtns_composition_field(Float64, cfg, dom)

    # Both boundaries no-flux
    Geodynamo.set_composition_boundary_conditions!(comp, :no_flux, :no_flux)
    Geodynamo.apply_composition_boundary_conditions_spectral!(comp, dom)

    # Check derivatives at both boundaries are ~0 for l=0,m=0
    spec_r = parent(comp.spectral.data_real)
    r_range = Geodynamo.get_local_range(comp.spectral.pencil, 3)
    dr = Geodynamo.create_derivative_matrix(1, dom)
    prof = zeros(Float64, dom.N)
    for r in 1:dom.N
        if r in r_range
            lr = r - first(r_range) + 1
            if lr <= size(spec_r, 3)
                prof[r] = spec_r[1,1,lr]
            end
        end
    end
    if MPI.Comm_size(Geodynamo.get_comm()) > 1
        MPI.Allreduce!(prof, MPI.SUM, Geodynamo.get_comm())
    end
    d = zeros(Float64, dom.N)
    Geodynamo.apply_derivative_matrix!(d, dr, prof)
    @test abs(d[1]) ≤ 1e-8
    @test abs(d[end]) ≤ 1e-8
end

function test_ball_outer_no_flux()
    params = GeodynamoParameters(geometry=:ball, i_N=32, i_L=16, i_M=16, i_Th=32, i_Ph=64)
    set_parameters!(params)
    cfg = Geodynamo.create_shtnskit_config(lmax=i_L, mmax=i_M, nlat=i_Th, nlon=i_Ph)
    # Ball uses outer radius only; inner r=0 is a regularity plane
    dom = GeodynamoBall.create_ball_radial_domain(i_N)
    comp = Geodynamo.create_shtns_composition_field(Float64, cfg, dom)

    # Inner: no-flux (skipped at r=0); Outer: no-flux enforced
    Geodynamo.set_composition_boundary_conditions!(comp, :no_flux, :no_flux)
    Geodynamo.apply_composition_boundary_conditions_spectral!(comp, dom)

    spec_r = parent(comp.spectral.data_real)
    r_range = Geodynamo.get_local_range(comp.spectral.pencil, 3)
    dr = Geodynamo.create_derivative_matrix(1, dom)
    prof = zeros(Float64, dom.N)
    for r in 1:dom.N
        if r in r_range
            lr = r - first(r_range) + 1
            if lr <= size(spec_r, 3)
                prof[r] = spec_r[1,1,lr]
            end
        end
    end
    if MPI.Comm_size(Geodynamo.get_comm()) > 1
        MPI.Allreduce!(prof, MPI.SUM, Geodynamo.get_comm())
    end
    d = zeros(Float64, dom.N)
    Geodynamo.apply_derivative_matrix!(d, dr, prof)
    @test abs(d[end]) ≤ 1e-8
end

test_fixed_and_no_flux()
test_no_flux_both_shell()
test_ball_outer_no_flux()
println("Composition BC tests completed.")
