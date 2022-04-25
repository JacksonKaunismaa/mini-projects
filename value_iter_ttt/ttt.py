import numpy as np

class TicTacToe():
    def __init__(self):
        self.diag_map = {}
        self.init_legal = []
        for row in range(3):
            for col in range(3):
                diff = abs(row-col)
                if row == col == 1:
                    self.diag_map[row,col] = 0
                elif diff == 0:
                    self.diag_map[row,col] = 0
                elif diff == 2:
                    self.diag_map[row,col] = 1
                self.init_legal.append((row,col))
        self.reset()

    def reset(self):
        self.board = np.zeros((3,3))
        self.row_sums = np.zeros(3)
        self.col_sums = np.zeros(3)
        self.diag_sums = np.zeros(2)
        self.player = 1
        self.moves = 0
        self.win_flag = 0
        self.legal = self.init_legal.copy()

    def full_move(self, pos):
        self.board[pos] = self.player
        # update win states
        self.row_sums[pos[0]] += self.player
        self.win_flag += abs(self.row_sums[pos[0]]) == 3
        self.col_sums[pos[1]] += self.player
        self.win_flag += abs(self.col_sums[pos[1]]) == 3
        # diagonal updates
        if pos in self.diag_map:
            select = self.diag_map[pos]
            self.diag_sums[select] += self.player
            self.win_flag += abs(self.diag_sums[select]) == 3
            if pos[0] == 1:  # update both diagonals if in the center
                self.diag_sums[1] += self.player
                self.win_flag += abs(self.diag_sums[1]) == 3
        self.legal.remove(pos)
        self.moves += 1
        self.player = -self.player

    def get_next(self, pos):
        self.board[pos] = self.player
        res = hash(self)
        self.board[pos] = 0
        return res

    def win(self):
        return self.win_flag

    def draw(self):
        return not self.win_flag and self.moves == 9

    def is_legal(self, pos):
        return pos in self.legal

    def all_legal(self):
        return self.legal

    def human_move(self):
        while True:
            try:
                inpt = input("Move: ")
                mv = tuple([int(x) for x in inpt.split()])
                if not self.is_legal(mv):
                    print("Illegal move")
                else:
                    break
            except ValueError:
                print("Invalid move")
        return mv

    def __hash__(self):
        return hash(self.board.tobytes())

    def __repr__(self):
        s = "-"*9 + "\n| "
        for row in self.board:
            for elem in row:
                if elem == 0:
                    s += "- "
                elif elem == 1:
                    s += "X "
                elif elem == -1:
                    s += "O "
            s += "|\n| "
        s += "\b"*4 + "-"*9
        return s

def explore():
    game = TicTacToe()
    player = np.random.randint(0,2)
    while not (game.win() or game.draw()):
        print(game, hash(game))
        if player: # human
            mv = game.human_move()
            game.full_move(mv)
        else:
            mv = game.human_move()
            game.full_move(mv)
        player = 1-player
    if game.draw():
        print("Draw")
    elif player == 0:
        print("Congrats you win")
    else:
        print("You lose")


def main():
    times = 0
    wins = 0
    N = 100_000
    import time
    import rng
    from tqdm import tqdm
    generate = rng.RNG(np.random.randint, (2,), 0, 3)
    states = {}
    board = TicTacToe()
    for _ in tqdm(range(N)):
        board.reset()
        start = time.perf_counter()
        while True:
            #mv = tuple(np.random.randint(0,3,(2)))
            mv = tuple(generate.next())
            if mv in board.all_legal():
                board.move(mv)
            if board.win():
                #print(board, "win")
                wins += 1
                states[hash(board)] = True
                #print(board, board.__hash__())
                break
            if board.draw():
                #print(board, "drow")
                states[hash(board)] = True
                break
        times += time.perf_counter() - start
    print("avg time:", times/N)
    print("avg wins:", wins/N)
    print(len(states))
    print(list(states.keys())[:5])

if __name__ == "__main__":
    #main()
    explore()
