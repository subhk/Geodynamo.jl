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

# 3) Run
run_shtns_simulation!(state)

