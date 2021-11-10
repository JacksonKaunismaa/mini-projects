#!/usr/bin/python3
import numpy as np
import random
import q_learning as ql
import sys
"""Another test game for the q learning agent (tic tac toe)"""

class TicTacToe(object):
    def __init__(self, init_board=None):
        if init_board is None:
            self.board = np.zeros((2, 3, 3))
        else:
            self.board = init_board.copy()

    def __repr__(self):
        border = "+" + "—" * (4 * 3 - 1) + "+"
        the_repr = ""
        the_repr += border
        for i in range(3):
            the_repr += "\n|"
            for j in range(3):
                if self.board[0, i, j]:
                    the_repr += " X "
                elif self.board[1, i, j]:
                    the_repr += " O "
                else:
                    the_repr += " - "
                the_repr += "|"
            the_repr += "\n" + border
        return the_repr


    def reverse(self):
        return repr(TicTacToe(init_board=self.board[::-1]))

    def reset(self):
        self.board = np.zeros((2, 3, 3))

    def move(self, idx):
        try:
            self.board[0, idx // 3, idx % 3] = 1.0
            self.board = self.board[::-1]
        except IndexError:
            print(f"Uh oh! Move {idx} is illegal!")
            raise

    def unmove(self, idx):
        try:
            self.board[1, idx // 3, idx % 3] = 0.0
            self.board = self.board[::-1]
        except IndexError:
            print(f"Uh oh! Move {idx} is illegal!")
            raise

    def get_legal(self):
        d_idx = np.where(self.board[0] + self.board[1] == 0)
        return [i*3 + j for i, j in zip(d_idx[0], d_idx[1])]

    def __hash__(self):
        return hash(self.board.tostring())

    def win_check(self, m):
        y = m // 3
        x = m % 3
        amount = 2
        if self.board[1, :, x].all():
            return amount
        elif self.board[1, y, :].all():
            return amount
        if self.board[1, 1, 1]:
            if y == 0:
                if x == 0 and self.board[1, 2, 2]:
                    return amount
                elif x == 2 and self.board[1, 2, 0]:
                    return amount
            elif y == 2:
                if x == 0 and self.board[1, 0, 2]:
                    return amount
                elif x == 2 and self.board[1, 0, 0]:
                    return amount
        if y == 1 and x == 1:
            if self.board[1, 0, 0] and self.board[1, 2, 2]:
                return amount
            elif self.board[1, 0, 2] and self.board[1, 2, 0]:
                return amount
        if len(self.get_legal()) == 0:
            return 1e-8
        return 0

def main():
    game = TicTacToe()
    agent = ql.QLearn(20.0, 9)
    print("get_scores", agent.get_scores(game))
    try:
        agent.load("simple_ttt.pickle")
    except FileNotFoundError:
        agent.learn(game, 100000)
        agent.save("simple_ttt.pickle")
    switch = bool(random.getrandbits(1))
    if len(sys.argv) > 1 and sys.argv[1] == "-t":
        agent.learn(game, 100000)
        agent.save("simple_ttt.pickle")
    prev_move = 0
    game.reset()
    print("get_scores:", agent.get_scores(game))
    print("get_q_values:", agent.Q[hash(game)])
    for n in range(9):
        try:
            print("get_n_values:", agent.N[hash(game)])
        except KeyError:
            print(game)
            raise
        if switch:
            print(game)
            hmove = int(input("move: "))
            prev_move = hmove
            game.move(hmove)
            if game.win_check(prev_move) == 2:
                print(game)
                print("Wow good job!")
                return 0
        else:
            qmove = agent.h_play(game)
            prev_move = qmove
            game.move(qmove)
            if game.win_check(prev_move) == 2:
                print(game)
                print("Wow you suck!")
                return 0
        switch = not switch
    print(game)
    print("Wow you're ok!")

if __name__ == "__main__":
    main()
