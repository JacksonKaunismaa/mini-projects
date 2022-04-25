import rng
import numpy as np
import ttt
from tqdm import tqdm
import pickle
from collections import defaultdict
import os

class ValueDict(dict):
    def __init__(self):
        super().__init__()
        self.pool = rng.RNG(np.random.uniform, (1,), -1, 1)

    def __missing__(self, key):
        return self.pool.next()[0]


class ValueIteration():
    def __init__(self, weight=0.95, gamma=0.7, tau=1.5):
        self.values = ValueDict()
        self.expanded = {}
        self.weight = weight
        self.gamma = gamma
        self.train_tau = tau
        self.N = defaultdict(int)
        self.log = False
        self.train_mode = True
        self.train()

    def train(self):
        self.log = False#True
        self.train_mode = True
        self.tau = self.train_tau

    def eval(self):
        self.train_mode = False
        self.log = True
        self.tau = 0.01

    def update(self, reward, state):
        self.values[state] = self.values[state]*self.weight + (1-self.weight)*reward
        if self.log:
            print("updated values[state] to, reward, state", self.values[state], reward, state)

    def update_history(self, states, reward):
        if self.log:
            print("Updating histories...")
        reward /= self.gamma
        for state in states[::-1]:
            self.update(reward, state)
            reward *= self.gamma

    def softmax(self, x):
        e = np.exp(x/self.tau)
        return e/e.sum()

    def get_move(self, game, ghash):
        distrib = {}
        if self.train_mode:
            self.N[ghash] += 1
        if ghash in self.expanded:
            if self.log:
                print("In expanded, states, next is", self.expanded[ghash])
            for state,mv in self.expanded[ghash]:
                distrib[mv] = self.values[state]*(game.player)
        else:
            self.expanded[ghash] = []
            for mv in game.all_legal():
                mv_state = game.get_next(mv)
                self.expanded[ghash].append((mv_state, mv))
                distrib[mv] = self.values[mv_state]*(game.player)
        mvs = list(distrib.keys())
        probs = self.softmax(np.array(list(distrib.values())))
        if self.log:
            print("values, mvs and num_visists be:", mvs, distrib.values(), self.N[ghash])
        return mvs[np.random.choice(len(mvs), p=probs)]

    def save(self, path):
        with open(path, "wb") as p:
            pickle.dump((self.values,self.N), p)

    def load(self, path):
        with open(path, "rb") as p:
            self.values,self.N = pickle.load(p)

def get_rand(game):
    return game.all_legal()[np.random.choice(9-game.moves)]

def train(value_iter, num):
    if os.path.exists("ttt.vi"):
        value_iter.load("ttt.vi")
        return
    value_iter.train()
    game = ttt.TicTacToe()
    init_ghash = hash(game)
    for _ in tqdm(range(num)):
        hist = [init_ghash]
        while not (game.win() or game.draw()):
            #print(game, hist[-1], game.player)
            next_mv = value_iter.get_move(game, hist[-1])
            game.full_move(next_mv)
            ghash = hash(game)
            hist.append(ghash)
            #input()
        if game.win():
            value_iter.update_history(hist, -game.player)
        else:
            value_iter.update_history(hist, 0)
        game.reset()
    print(f"After training, value iter has {len(value_iter.expanded)} expanded states")
    value_iter.save("ttt.vi")


def play_rand(value_iter, games):
    value_iter.eval()
    value_iter.log = False
    game = ttt.TicTacToe()
    vi_wins = 0
    draws = 0
    rand_wins = 0
    for _ in range(games):
        player = np.random.randint(0,2)
        while not (game.win() or game.draw()):
            #print(game, hash(game))
            if player: # random
                #value_iter.get_move(game, hash(game))
                mv = get_rand(game) #game.human_move()
                game.full_move(mv)
            else:
                mv = value_iter.get_move(game, hash(game))
                game.full_move(mv)
            player = 1-player
        if game.draw():
            draws += 1#print("Draw")
        if player == 0:
            rand_wins += 1#print("Congrats you win")
        else:
            #print("You lose")
            vi_wins += 1
        game.reset()
    print("Draw pct:", draws/games)
    print("VI win pct:", vi_wins/games)
    print("rand win pct:", rand_wins/games)


def play(value_iter):
    value_iter.eval()
    game = ttt.TicTacToe()
    player = np.random.randint(0,2)
    while not (game.win() or game.draw()):
        print(game, hash(game))
        if player: # human
            value_iter.get_move(game, hash(game))
            mv = game.human_move()
            game.full_move(mv)
        else:
            mv = value_iter.get_move(game, hash(game))
            game.full_move(mv)
        player = 1-player
    if game.draw():
        print("Draw")
    elif player == 0:
        print("Congrats you win")
    else:
        print("You lose")


def main():
    value_iter = ValueIteration()
    train(value_iter, 500_000)
    #play(value_iter)
    play_rand(value_iter, 1000)

if __name__ == "__main__":
    main()
