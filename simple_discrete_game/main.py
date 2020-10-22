# KidsCanCode - Game Development with Pygame video series
# Tile-based game - Part 1
# Project setup
# Video link: https://youtu.be/3UxnelT9aCo
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import sys
from settings import *
from sprites import *

# os.putenv("SDL_VIDEODRIVER", "fbcon")
# os.environ["SDL_VIDEODRIVER"] = "dummy"


class Game:
    def __init__(self):
        pg.init()
        pg.display.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))

        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.max_reward = 20

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
        map[1][2] = 2
        map[10][12] = 3
        return map

    def new(self):
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
        # self.save_obs("wqeewqqwqerq1.png")

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.maunal_actions()
            self.update()
            print(self._get_obs())
            # print("reward", self._reward_func())
            self.save_obs("1221321321.png")
            self.draw()
            if self._check_done():
                self.quit()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

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

    def maunal_actions(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_LEFT:
                    self.player.move(dx=-1)
                if event.key == pg.K_RIGHT:
                    self.player.move(dx=1)
                if event.key == pg.K_UP:
                    self.player.move(dy=-1)
                if event.key == pg.K_DOWN:
                    self.player.move(dy=1)

    def _get_obs(self):
        return pg.surfarray.array3d(self.screen).swapaxes(0, 1)

    def _reward_func(self):
        # rewaard is the distance between goal and player
        return self.max_reward - (
            round(math.hypot(self.goal.x - self.player.x, self.goal.y - self.player.y), 2)
        )

    def _check_done(self):
        if self._reward_func() == self.max_reward:
            return True
        return False

    def save_obs(self, save_name):
        obs = self._get_obs()
        plt.imsave(save_name, obs)


# create the game object
g = Game()
while True:
    g.new()
    g.run()
