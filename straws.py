import numpy as np
import q_learning as ql
import sys
import random
"""Simple game that I tested the Q-Learning agent on, not sure if it is a correct name for it, but it is essentially a game where you start with some number, say 120
and then you pick another number (<120), call it x. Each 'move' involves players taking turns calling out a number from 1 to x, which increases the running total. Whichever
player is able to call out the final number (eg. 120) wins the game."""

class Straws(object):
    def __init__(self, game_len, max_move):
        self.len = game_len
        self.mv_sz = max_move
        self.counter = 0

    def move(self, amount):
        self.counter += amount + 1

    def unmove(self, amount):
        self.counter -= amount

    def win_check(self, m):
        if self.counter == self.len:
            return 2
        return 0

    def __repr__(self):
        return f"{self.counter}/{self.len}"

    def get_legal(self):
        return list(range(min(self.len - self.counter, self.mv_sz)))

    def __hash__(self):
        return hash(f"{self.len - self.counter}|{self.mv_sz}")

    def reset(self):
        self.counter = 0



def main():
    game = Straws(1000, 50)
    agent = ql.QLearn(10.0, 50)
    try:
        agent.load("simple_nim.pickle")
    except FileNotFoundError:
        agent.learn(game, 3000)
    if len(sys.argv) > 1 and sys.argv[1] == '-t':
        agent.learn(game, 30000)
    agent.save("simple_nim.pickle")
    switch = bool(random.getrandbits(1))
    while True:
        if switch:
            print(game)
            hmove = int(input("move: "))
            print("Huuman move chose =>", hmove)
            while hmove not in game.get_legal():
                hmove = int(input("move: "))
            game.move(hmove)
            if game.win_check(5):
                print(game)
                print("wow you're good!")
                return 0
        else:
            print(game)
            qmove = agent.h_play(game)
            print("Q agent move chose =>", qmove)
            print("numbers =>", agent.N[hash(game)])
            game.move(qmove)
            if game.win_check(5):
                print(game)
                print("lol u suck!")
                return 0
        switch = not switch


if __name__ == "__main__":
    main()
