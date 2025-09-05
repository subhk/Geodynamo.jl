#!/usr/bin/env julia

# Rotating MHD convection in a ball (solid sphere)
# - Geometry: :ball with outer radius = 1.0
# - Ekman = 1e-4, Pr = 1, Pm = 1, Ra = 1e6
# - Magnetic field enabled (i_B = 1)
#
# Run:
#   julia --project examples/ball_mhd_demo.jl
# or with threads:
#   JULIA_NUM_THREADS=8 julia --project examples/ball_mhd_demo.jl

using Geodynamo

# 1) Set parameters (ball geometry and physics)
params = GeodynamoParameters(
    # Geometry + grid
    geometry   = :ball,   # solid sphere geometry
    i_N        = 64,      # radial points
    i_L        = 32,      # lmax
    i_M        = 32,      # mmax
    i_Th       = 64,      # nlat
    i_Ph       = 128,     # nlon
    i_KL       = 4,       # FD bandwidth

    # Physics
    d_E        = 1e-4,    # Ekman number
    d_Pr       = 1.0,     # Prandtl number
    d_Pm       = 1.0,     # Magnetic Prandtl number
    d_Ra       = 1e6,     # Rayleigh number

    # Magnetic field on
    i_B        = 1,

    # Ball size (outer radius)
    d_R_outer  = 1.0,

    # Timestepping / runtime controls (tune for your machine)
    d_timestep = 1e-4,
    i_maxtstep = 500,
    i_save_rate2 = 50,
)

set_parameters!(params)

# 2) Initialize basic SHTns simulation (thermal + magnetic, no composition)
state = initialize_shtns_simulation(Float64; include_composition=false)

# Debug: print pencil layouts (axes_in) to verify decomposition
Geodynamo.print_pencil_axes(state.shtns_config.pencils)

# Optional: register a shared VelocityWorkspace to reduce allocations
if get(ENV, "GEODYNAMO_USE_WS", "1") == "1"
    ws = Geodynamo.create_velocity_workspace(Float64, state.oc_domain.N)
    Geodynamo.set_velocity_workspace!(ws)
end

# 3) Temperature boundary conditions (Dirichlet inner/outer)
#    Options:
#    - Dirichlet (fixed T): inner_bc_type=1, outer_bc_type=1, values below
#    - Neumann (flux): use inner_bc_type=2/outer_bc_type=2; flux profile is built-in for l=0
set_boundary_conditions!(state.temperature;
    inner_bc_type=1, inner_value=1.0,
    outer_bc_type=1, outer_value=0.0,
)

# Alternative: programmatic uniform boundaries via hybrid API
# using GeodynamoBall
# temp_bc = GeodynamoBall.create_ball_hybrid_temperature_boundaries((:uniform, 1.0), (:uniform, 0.0), state.shtns_config)
# apply_netcdf_temperature_boundaries!(state.temperature, temp_bc)

# 4) Initial conditions
#    - Temperature: conductive profile + small perturbation
set_temperature_ic!(state.temperature, state.oc_domain; perturbation_amplitude=1e-3)

#    - Overwrite l=0,m=0 with conductive profile matching BC values
function _find_mode_index(config, l_target::Int, m_target::Int)
    for i in 1:config.nlm
        if config.l_values[i] == l_target && config.m_values[i] == m_target
            return i
        end
    end
    return 0
end

function set_conductive_ic!(temp_field, domain; T_in=1.0, T_out=0.0)
    spec_r = parent(temp_field.spectral.data_real)
    spec_i = parent(temp_field.spectral.data_imag)
    lm_rng = Geodynamo.get_local_range(temp_field.spectral.pencil, 1)
    r_rng  = Geodynamo.get_local_range(temp_field.spectral.pencil, 3)
    l0m0 = _find_mode_index(temp_field.config, 0, 0)
    if l0m0 != 0 && (first(lm_rng) <= l0m0 <= last(lm_rng))
        ll = l0m0 - first(lm_rng) + 1
        ri = domain.r[1, 4]
        ro = domain.r[end, 4]
        for r_idx in r_rng
            rr = r_idx - first(r_rng) + 1
            if rr <= size(spec_r, 3)
                r = domain.r[r_idx, 4]
                spec_r[ll, 1, rr] = T_in + (T_out - T_in) * (r - ri) / (ro - ri)
                spec_i[ll, 1, rr] = 0.0
            end
        end
    end
end

set_conductive_ic!(state.temperature, state.oc_domain; T_in=1.0, T_out=0.0)

#    - Velocity: starts at zero (recommended). Uncomment to add tiny perturbations:
# using Random
# Random.seed!(1234)
# tor = parent(state.velocity.toroidal.data_real); 
# pol = parent(state.velocity.poloidal.data_real)
# lm_range = range_local(state.velocity.toroidal.pencil, 1)
# r_range  = range_local(state.velocity.toroidal.pencil, 3)
# @inbounds for lm_idx in lm_range
#     l = state.velocity.toroidal.config.l_values[lm_idx]
#     if l <= 3
#         ll = lm_idx - first(lm_range) + 1
#         for r_idx in r_range
#             rr = r_idx - first(r_range) + 1
#             if rr <= size(tor,3)
#                 tor[ll,1,rr] = 1e-6 * (rand()-0.5)
#                 pol[ll,1,rr] = 1e-6 * (rand()-0.5)
#             end
#         end
#     end
# end

# 5) Run
run_shtns_simulation!(state)
