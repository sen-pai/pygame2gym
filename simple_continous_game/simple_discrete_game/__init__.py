from gym.envs.registration import register

register(id="GoalGrid-v0", entry_point="simple_discrete_game.envs:GoalGridEnv")

# register(id="BubbleShooter-v0", entry_point="gym_bubbleshooter.envs:BubbleShooterEnv")
