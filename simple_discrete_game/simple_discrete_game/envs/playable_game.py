# KidsCanCode - Game Development with Pygame video series
# Tile-based game - Part 1
# Project setup
# Video link: https://youtu.be/3UxnelT9aCo
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
import math, random
import sys
from settings import *
from sprites import *


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
        possible_x = [2,3,4,5,6,7,8]
        possible_y = [2,3,4,5,6,7,8, 9, 10]

        map[random.sample(possible_x, 1)[0]][random.sample(possible_y, 1)[0]] = 2
        map[random.sample(possible_x, 1)[0]][random.sample(possible_y, 1)[0]] = 3
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

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.maunal_actions()
            self.update()
            self.draw()

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
                    self.new()
                if event.key == pg.K_LEFT:
                    self.player.move(dx=-1)
                if event.key == pg.K_RIGHT:
                    self.player.move(dx=1)
                if event.key == pg.K_UP:
                    self.player.move(dy=-1)
                if event.key == pg.K_DOWN:
                    self.player.move(dy=1)

# create the game object
g = Game()
while True:
    g.new()
    g.run()
