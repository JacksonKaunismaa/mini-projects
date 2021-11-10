import numpy as np
import math
import random
import pickle
"""Not 100% if this exactly classifies as Q-Learning, but I think it essentially fits the mold, as it builds up a table of what moves lead to high scores
in a given position of an implementation of a Game class and tries to balance exploration and exploitation, and is trained by doing a ton of self-play games"""

class QLearn(object):
    def __init__(self, c, max_moves):
        self.Q = {}
        self.N = {}
        self.visited = set()
        self.c = c
        self.max_moves = max_moves

    def search(self, game_state, c_val):
        h_game = game_state.__hash__()
        if h_game not in self.visited:
            self.visited.add(h_game)
            self.Q[h_game] = np.zeros(self.max_moves)
            self.N[h_game] = np.zeros(self.max_moves)
            return np.random.choice(game_state.get_legal()), h_game
        else:
            max_score = -float("inf")
            b_move = -1.212
            sum_sqrt = math.sqrt(sum(self.N[h_game]))
            for m in game_state.get_legal():
                ucb = self.Q[h_game][m] + c_val * np.log(1+sum_sqrt/(1+self.N[h_game][m]))
                if ucb >= max_score:
                    max_score = ucb
                    b_move = m
            return b_move, h_game

    def get_next_max(self, game_state, m):
        try:
            game_state.move(m)
            g_hash = hash(game_state)
            game_state.unmove(m)
            return max(self.Q[g_hash])
        except KeyError:
            return 0

    def update_table(self, score, history, gamma):
        s = score
        for h, m in history[::-1]:
            try:
                self.Q[h][m] = (s + self.Q[h][m] * self.N[h][m]) / (1+self.N[h][m])
                self.N[h][m] += 1
            except Exception:
                raise
            s *= -gamma


    def get_scores(self, pos):
        try:
            return self.Q[hash(pos)] + self.c * np.log(1+math.sqrt(sum(self.N[hash(pos)]))/(1+self.N[hash(pos)]))
        except KeyError:
            self.search(pos, self.c)
            return self.Q[hash(pos)] + self.c * np.log(1+math.sqrt(sum(self.N[hash(pos)]))/(1+self.N[hash(pos)]))

    def save(self, name):
        with open(name, "wb") as p:
            pickle.dump((self.Q, self.N, self.visited, self.c, self.max_moves), p)

    def load(self, name):
        with open(name, "rb") as p:
            self.Q, self.N, self.visited, self.c, self.max_moves = pickle.load(p)

    def learn(self, g, num_games, gamma=0.97):
        c_val = self.c
        discount = self.c / float(num_games)
        for n in range(num_games):
            g.reset()
            hist = []
            prev_move = 0
            while True:
                res = g.win_check(prev_move)
                if res:
                    break
                else:
                    next_move, h_game = self.search(g, c_val)
                    prev_move = next_move
                    hist.append((h_game, next_move))
                    g.move(next_move)
            self.update_table(res, hist, gamma)
            c_val -= discount
        g.reset()


    def h_play(self, game_state):
        res = self.search(game_state, 0.0)
        return res[0]
