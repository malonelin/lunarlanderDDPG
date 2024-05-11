# lunarlanderDDPG
fine turn version of DDPG for LunarLanderContinuous-v2
test_cnt:100 gt200_cnt:(100/100) avg_steps:165.95 avg_reward: 286.00 w_file./w/actor_best.pth_ep32514_rw292.77_st177

2024-05-12 00:36:54.213 | INFO     | __main__:<module>:56 - hyparam:
2024-05-12 00:36:54.213 | INFO     | __main__:<module>:57 - Namespace(mode='test', render=False, env_name='LunarLanderContinuous-v2', tau=0.001, target_update_interval=1, test_iteration=100, lr_actor=0.0001, lr_critic=0.001, gamma=0.99, capacity=100000.0, batch_size=64, seed=False, random_seed=9527, max_length_of_trajectory=250, log_interval=20, load=False, exploration_noise=0.1, max_episode=100000, best_w_file='./w/actor_best.pth_ep32514_rw292.77_st177', best_w_file_name='actor_best.pth*rw29*', update_iteration=20)
2024-05-12 00:36:55.993 | INFO     | __main__:<module>:60 - using:cuda


2024-05-12 00:36:58.037 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  0 steps:203 reward: 318.89
2024-05-12 00:36:58.144 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  1 steps:183 reward: 302.41
2024-05-12 00:36:58.221 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  2 steps:132 reward: 285.26
2024-05-12 00:36:58.334 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  3 steps:193 reward: 306.34
2024-05-12 00:36:58.447 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  4 steps:195 reward: 296.55
2024-05-12 00:36:58.546 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  5 steps:169 reward: 293.00
2024-05-12 00:36:58.631 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  6 steps:146 reward: 277.44
2024-05-12 00:36:58.717 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  7 steps:148 reward: 276.50
2024-05-12 00:36:58.800 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  8 steps:142 reward: 284.73
2024-05-12 00:36:58.900 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep  9 steps:174 reward: 255.09
2024-05-12 00:36:58.978 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 10 steps:136 reward: 268.56
2024-05-12 00:36:59.075 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 11 steps:164 reward: 310.10
2024-05-12 00:36:59.193 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 12 steps:205 reward: 279.30
2024-05-12 00:36:59.325 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 13 steps:226 reward: 308.35
2024-05-12 00:36:59.418 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 14 steps:164 reward: 278.01
2024-05-12 00:36:59.550 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 15 steps:226 reward: 298.23
2024-05-12 00:36:59.647 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 16 steps:167 reward: 270.99
2024-05-12 00:36:59.742 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 17 steps:164 reward: 282.01
2024-05-12 00:36:59.842 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 18 steps:174 reward: 276.17
2024-05-12 00:36:59.929 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 19 steps:148 reward: 268.26
2024-05-12 00:37:00.027 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 20 steps:167 reward: 288.56
2024-05-12 00:37:00.120 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 21 steps:159 reward: 297.59
2024-05-12 00:37:00.214 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 22 steps:163 reward: 275.77
2024-05-12 00:37:00.305 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 23 steps:155 reward: 286.20
2024-05-12 00:37:00.385 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 24 steps:138 reward: 294.62
2024-05-12 00:37:00.468 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 25 steps:141 reward: 295.68
2024-05-12 00:37:00.576 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 26 steps:184 reward: 302.10
2024-05-12 00:37:00.696 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 27 steps:206 reward: 288.87
2024-05-12 00:37:00.799 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 28 steps:175 reward: 299.36
2024-05-12 00:37:00.887 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 29 steps:148 reward: 236.79
2024-05-12 00:37:00.992 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 30 steps:177 reward: 301.82
2024-05-12 00:37:01.094 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 31 steps:176 reward: 252.79
2024-05-12 00:37:01.168 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 32 steps:127 reward: 281.60
2024-05-12 00:37:01.255 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 33 steps:150 reward: 282.21
2024-05-12 00:37:01.344 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 34 steps:150 reward: 310.59
2024-05-12 00:37:01.417 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 35 steps:126 reward: 282.62
2024-05-12 00:37:01.505 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 36 steps:151 reward: 291.01
2024-05-12 00:37:01.594 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 37 steps:155 reward: 257.73
2024-05-12 00:37:01.689 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 38 steps:164 reward: 310.92
2024-05-12 00:37:01.815 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 39 steps:214 reward: 305.62
2024-05-12 00:37:01.897 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 40 steps:145 reward: 286.02
2024-05-12 00:37:01.996 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 41 steps:168 reward: 312.72
2024-05-12 00:37:02.079 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 42 steps:142 reward: 289.51
2024-05-12 00:37:02.172 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 43 steps:158 reward: 268.23
2024-05-12 00:37:02.258 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 44 steps:148 reward: 250.31
2024-05-12 00:37:02.337 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 45 steps:138 reward: 289.61
2024-05-12 00:37:02.414 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 46 steps:133 reward: 271.19
2024-05-12 00:37:02.494 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 47 steps:136 reward: 294.50
2024-05-12 00:37:02.620 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 48 steps:214 reward: 295.35
2024-05-12 00:37:02.748 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 49 steps:221 reward: 293.44
2024-05-12 00:37:02.865 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 50 steps:202 reward: 295.39
2024-05-12 00:37:02.962 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 51 steps:167 reward: 269.26
2024-05-12 00:37:03.063 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 52 steps:169 reward: 296.99
2024-05-12 00:37:03.181 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 53 steps:202 reward: 297.51
2024-05-12 00:37:03.280 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 54 steps:168 reward: 272.96
2024-05-12 00:37:03.377 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 55 steps:164 reward: 260.36
2024-05-12 00:37:03.474 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 56 steps:167 reward: 261.10
2024-05-12 00:37:03.563 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 57 steps:151 reward: 299.47
2024-05-12 00:37:03.653 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 58 steps:153 reward: 271.37
2024-05-12 00:37:03.736 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 59 steps:142 reward: 272.24
2024-05-12 00:37:03.825 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 60 steps:152 reward: 294.12
2024-05-12 00:37:03.915 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 61 steps:154 reward: 251.74
2024-05-12 00:37:04.003 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 62 steps:150 reward: 299.39
2024-05-12 00:37:04.098 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 63 steps:159 reward: 299.50
2024-05-12 00:37:04.204 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 64 steps:182 reward: 286.11
2024-05-12 00:37:04.300 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 65 steps:162 reward: 308.23
2024-05-12 00:37:04.395 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 66 steps:164 reward: 267.48
2024-05-12 00:37:04.476 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 67 steps:138 reward: 290.13
2024-05-12 00:37:04.564 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 68 steps:148 reward: 315.43
2024-05-12 00:37:04.656 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 69 steps:159 reward: 269.47
2024-05-12 00:37:04.775 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 70 steps:204 reward: 289.11
2024-05-12 00:37:04.863 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 71 steps:150 reward: 288.06
2024-05-12 00:37:04.969 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 72 steps:181 reward: 310.59
2024-05-12 00:37:05.083 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 73 steps:165 reward: 270.80
2024-05-12 00:37:05.190 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 74 steps:177 reward: 312.73
2024-05-12 00:37:05.293 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 75 steps:175 reward: 283.08
2024-05-12 00:37:05.386 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 76 steps:157 reward: 285.79
2024-05-12 00:37:05.491 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 77 steps:183 reward: 278.91
2024-05-12 00:37:05.595 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 78 steps:180 reward: 281.00
2024-05-12 00:37:05.674 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 79 steps:138 reward: 290.88
2024-05-12 00:37:05.755 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 80 steps:141 reward: 255.16
2024-05-12 00:37:05.852 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 81 steps:170 reward: 273.22
2024-05-12 00:37:05.938 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 82 steps:148 reward: 278.18
2024-05-12 00:37:06.014 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 83 steps:132 reward: 279.86
2024-05-12 00:37:06.100 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 84 steps:146 reward: 250.65
2024-05-12 00:37:06.203 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 85 steps:177 reward: 311.15
2024-05-12 00:37:06.298 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 86 steps:159 reward: 294.63
2024-05-12 00:37:06.377 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 87 steps:137 reward: 274.91
2024-05-12 00:37:06.498 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 88 steps:209 reward: 305.88
2024-05-12 00:37:06.587 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 89 steps:152 reward: 305.91
2024-05-12 00:37:06.696 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 90 steps:189 reward: 302.81
2024-05-12 00:37:06.837 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 91 steps:243 reward: 314.77
2024-05-12 00:37:06.919 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 92 steps:141 reward: 278.12
2024-05-12 00:37:07.026 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 93 steps:186 reward: 281.74
2024-05-12 00:37:07.127 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 94 steps:174 reward: 279.69
2024-05-12 00:37:07.233 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 95 steps:184 reward: 249.85
2024-05-12 00:37:07.328 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 96 steps:163 reward: 292.28
2024-05-12 00:37:07.425 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 97 steps:166 reward: 289.01
2024-05-12 00:37:07.516 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 98 steps:155 reward: 296.79
2024-05-12 00:37:07.616 | INFO     | __main__:test_mode_one_ep:338 - test mode. ep 99 steps:172 reward: 286.90
2024-05-12 00:37:07.616 | INFO     | __main__:test_mode_one_ep:345 - test mode. test_cnt:100 gt200_cnt:(100/100) avg_steps:165.95 avg_reward: 286.00 w_file./w/actor_best.pth_ep32514_rw292.77_st177

modify --mode in args, to switch between training mode and testing mode
modify --render in args, to switch between render mode and backend mode
