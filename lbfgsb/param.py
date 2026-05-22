class Param:
    # time
    timesteps = 200
    dt = 0.02

    # loss weights
    ergodic_weight = 1e5      # 1e6
    surface_weight = 10.0     # 50.0
    accel_l2_weight = 1e-1    # 2e-2
    jerk_l2_weight = 0      # 1e-7
    vel_l2_weight = 0.0       # 1e-8

    # point cloud + spectrum
    alpha = 100
    voxel_size = 0.003
    nb_eigen = 400

    nb_max_neighbors = 300
    nb_minimum_neighbors = 150
    agent_radius = 4.17e-5
    spectral_mix = 0.9

    # optimizer
    method = "L-BFGS-B"
    max_iterations = 100
    ftol = 1e-6
    gtol = 1e-6
    maxcor = 20
    debug = True
