import os
import numpy as np
import copy


class Ball:

    def __init__(self, col):
        self.col = col
        self.row = 0

    def update(self):
        self.row += 1

    def isDroped(self, n_rows):
        return True if self.row >= n_rows else False


class CatchBall:

    def __init__(self, time_limit=True):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 16
        self.screen_n_cols = 16
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5
        self.ball_post_interval = 8
        self.ball_past_time = 0
        self.past_time = 0
        self.balls = []
        self.time_limit = time_limit

        # variables
        self.reset()

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1, self.screen_n_cols - self.player_length)
        else:
            # do nothing
            pass

        # update ball position
        for b in self.balls:
            b.row += 1

        if self.ball_past_time == self.ball_post_interval:
            self.ball_past_time = 0
            new_pos = np.random.randint(self.screen_n_cols)
            while abs(new_pos - self.balls[-1].col) > self.ball_post_interval + self.player_length - 1:
                new_pos = np.random.randint(self.screen_n_cols)
            self.balls.append(Ball(new_pos))
        else:
            self.ball_past_time += 1

        # collision detection
        self.reward = 0
        self.terminal = False

        self.past_time += 1
        if self.time_limit and self.past_time > 200:
            self.terminal = True

        if self.balls[0].row == self.screen_n_rows - 1:
            if self.player_col <= self.balls[0].col < self.player_col + self.player_length:
                # catch
                self.reward = 1
            else:
                # drop
                self.reward = -1
                self.terminal = True

        new_balls = []
        for b in self.balls:
            if not b.isDroped(self.screen_n_rows):
                new_balls.append(b)
        self.balls = copy.copy(new_balls)

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw player
        self.screen[self.player_row, self.player_col:self.player_col + self.player_length] = 1

        # draw ball
        for b in self.balls:
            self.screen[b.row, b.col] = 0.5

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(self.screen_n_cols - self.player_length)

        # reset ball position
        self.balls = []
        self.balls.append(Ball(np.random.randint(self.screen_n_cols)))

        # reset other variables
        self.reward = 0
        self.terminal = False
        self.past_time = 0
        self.ball_past_time = 0
