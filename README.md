# lunarlanderDDPG
fine turn version of DDPG for LunarLanderContinuous-v2

get the best result at episode 4800. with avg reward 256.31 by testing 10 times. the log is:
2024-05-11 00:16:32.788 | INFO     | __main__:main:340 - training steps:1430454 ep:4800 landStep:158 reward:  153.57
2024-05-11 00:16:34.715 | INFO     | __main__:val_test:279 - validate test:10. avg_steps:278.9 avg_reward: 256.31

modify --mode in args, to switch between training mode and testing mode
modify --render in args, to switch between render mode and backend mode
