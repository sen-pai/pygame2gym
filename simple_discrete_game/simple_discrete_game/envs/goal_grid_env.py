# KidsCanCode - Game Development with Pygame video series
# Tile-based game - Part 1
# Project setup
# Video link: https://youtu.be/3UxnelT9aCo
import pygame as pg
from gym import spaces, error
import gym
import numpy as np
import matplotlib.pyplot as plt
import math, random
import os
import sys
from .settings import *
from .sprites import *

os.putenv("SDL_VIDEODRIVER", "fbcon")
os.environ["SDL_VIDEODRIVER"] = "dummy"


class GoalGridEnv(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": FPS}

    def __init__(self):
        pg.init()
        pg.display.init()
        pg.display.set_mode((1, 1))
        self.screen = pg.Surface((WIDTH, HEIGHT), pg.SRCALPHA, 32)
        self.clock = pg.time.Clock()

        # self.config = config
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=256.0, shape=(128, 128, 3))
        self.sparce_reward = False
        # use to maximize reward and not maximize distance
        self.max_distance = 1
        self.reset()

    def generate_new_map(self):
        """wall = 1
        player = 2
        goal = 3"""

        map = np.zeros((GRIDWIDTH, GRIDHEIGHT)).T

        # border walls
        for index, tile in np.ndenumerate(map):
            row, col = index
            if col == 0 or col == GRIDWIDTH - 1:
                map[row][col] = 1
            if row == 0 or row == GRIDHEIGHT - 1:
                map[row][col] = 1

        # add player and goal in a random cell
        # possible_x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        # possible_y = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        #
        possible_x = [2, 3, 4, 5]
        possible_y = [2, 3, 4, 5]

        map[random.sample(possible_x, 1)[0]][random.sample(possible_y, 1)[0]] = 2
        map[random.sample(possible_x, 1)[0]][random.sample(possible_y, 1)[0]] = 3

        # fixed player and Goal

        # map[1][1] = 2
        # map[14][14] = 3
        return map

    def reset(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.goals = pg.sprite.Group()
        map = self.generate_new_map()
        for index, tile in np.ndenumerate(map):
            row, col = index
            if tile == 1:
                Wall(self, col, row)
            if tile == 2:
                self.player = Player(self, col, row)
            if tile == 3:
                self.goal = Goal(self, col, row)

        self.all_sprites.update()
        self.draw()

        return self._get_obs()

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.draw_grid()
        self.all_sprites.draw(self.screen)
        pg.display.flip()

    def _get_obs(self):
        return pg.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _reward_func(self):
        self.goal_visited_reward = self.max_distance
        # reward is the distance between goal and player
        dist = round(-math.hypot(self.goal.x - self.player.x, self.goal.y - self.player.y), 2,)

        # dist = -0.1
        if dist == 0:
            return self.goal_visited_reward
        if self.sparce_reward:
            return 0
        return -0.1

    def _check_done(self):
        if self._reward_func() == self.goal_visited_reward:
            return True
        return False

    def step(self, action):

        if action == 0:
            self.player.move(dx=-1)
        if action == 1:
            self.player.move(dx=1)
        if action == 2:
            self.player.move(dy=-1)
        if action == 3:
            self.player.move(dy=1)

        self.all_sprites.update()
        self.draw()

        obs = self._get_obs()
        reward = self._reward_func()
        done = self._check_done()
        info = {}

        return obs, reward, done, info

    def render(self, mode="human", close=False):
        pass

    def save_obs(self, save_name):
        obs = self._get_obs()
        plt.imsave(save_name, obs)

    def close(self):
        pg.quit()
        sys.exit()
