# RL-Track-Simulator

Patch Note

#2018.9.3
1. Fixed a bug where reward did not work properly after reset function.
2. Make an episode terminate after deviating outer yellow line.

#2018.9.7
1. No need to fix camera setting anymore
2. environment with obstacles
 - 8 cars on straight 4 lane
 - reward : previous reward + collsion reward + repulsive reward




### How to use the environment

1. straight_4lane
 - roslaunch dbw_runner straight_4lane_rl.launch
 - import rlsim_env
 - rlsim_env.make("straight_4lane") or rlsim_env.make("straight_4lane_cam")
 - observation : 8 dim (pose, velocity, direction) or (800,800,120) dim (4 stacked rgb)
 - action : steering (-8.0~8.0), accel (0.0~1.0), brake (0.0~1.0)

2. straight_4lane with obstacles
 - roslaunch dbw_runner straight_4lane_obs_rl.launch
 - import rlsim_env
 - rlsim_env.make("straight_4lane_obs") or rlsim_env.make("straight_4lane_obs_cam")
 - observation : 8+16 dim (car state + relative pos to other cars) or (800,800,120) dim (4 stacked rgb)
 - action : steering (-8.0~8.0), accel (0.0~1.0), brake (0.0~1.0)
