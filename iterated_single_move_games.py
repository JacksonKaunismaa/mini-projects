from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        pass

    @abstractmethod
    def reset(self):
        pass

    def get_name(self):
        try:
            return self.name
        except AttributeError:
            return type(self).__name__


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        pass

    def reset(self):
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        return 0

    def update_results(self, my_move, other_move):
        pass

    def reset(self):
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        return self.last_move

    def update_results(self, my_move, other_move):
        self.last_move = other_move

    def reset(self):
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        #print("player1, player2 played", move1, move2)
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]
    return p1_score, p2_score

def init_agent(player_tup):
    if type(player_tup) is tuple:
        return player_tup[0](game_matrix, **player_tup[1])
    else:
        return player_tup(game_matrix)

def play_match(player1, player2, game_matrix, N, num_rounds, i, j):
    us = init_agent(player1)
    them = init_agent(player2)
    score = np.zeros((2))
    for _ in range(N):
        them.reset()
        us.reset()
        score += play_game(us, them, game_matrix, num_rounds)  # only ever play 1000 games
    return i,j,score, us.get_name(), them.get_name()

def tournament(all_agents, game_matrix, N):
    import matplotlib.pyplot as plt
    import multiprocessing as mp
    num_agents = len(all_agents)
    scores = np.zeros((num_agents, num_agents))
    names = {}
    num_rounds = 1000
    run_list = []
    with mp.Pool(7) as p:
        for i, us in enumerate(all_agents):
            for j, them in enumerate(all_agents[i:]):
                run_list.append((us, them, game_matrix, N, num_rounds, i, j))
        results = p.starmap(play_match, run_list)

    for res in results:
        i,j,score,name,name_them = res
        scores[i,j+i] = score[0]
        scores[j+i,i] = score[1]
        names[i] = name
    fig = plt.figure()
    ax = fig.add_subplot()
    print(scores)
    scores = (scores+N*num_rounds)/(2*N)-(num_rounds/2)  # put scores in (0,num_rounds)
    #scores = 50*np.tanh(0.006*((scores+1000*N)/(2*N)-500))  # nonlinear transform to make colors nicer
    #scores = scores/10
    scores = 50*np.tanh(0.006*scores)
    pcm = ax.matshow(scores)
    fig.colorbar(pcm, ax=ax)
    lbl_names = [names[i] for i in range(len(names))]
    ax.set_xticks(np.arange(num_agents), labels=lbl_names, rotation="vertical")
    ax.set_yticks(np.arange(num_agents), labels=lbl_names)
    plt.show()


class NAntiMAPMarkov(IteratedGamePlayer):
    # simulates exactly what a NMAPMarkov model might do and then produces moves that beat that
    def __init__(self, game_matrix: np.ndarray, N=5):
        super().__init__(game_matrix)
        self.name = f"NAntiMAPMarkov{N}" # probably want a balance between N being large (able to model high order complexity)
        self.max_idx = self.n_moves**N-1 # and N being small (dont make the matrix too sparse)
        self.reset()

    def make_move(self) -> int:
        opp_move = np.random.choice(np.where(self.p_matrix[self.idx] == self.p_matrix[self.idx].max())[0])
        return (opp_move+2)%self.n_moves

    def update_results(self, my_move, other_move):
        self.p_matrix[self.idx, my_move] += 1
        self.idx = (self.idx*self.n_moves)%(self.max_idx+1) # wipes out the highest order digit
        self.idx += my_move  # pushes the new move onto the state stack

    def reset(self):
        self.p_matrix = np.ones((self.max_idx+1, self.n_moves)) # assume uniform prior
        self.idx = np.random.randint(self.max_idx)  # base n_moves repr of the last_n moves

class NKMarkovAgent(IteratedGamePlayer):
    # basically NKMAPMarkov but instead of doing a MAP estimate, just randomly samples
    # from the transition table and assumes that will be the opponents move
    def __init__(self, game_matrix: np.ndarray, N=2, K=2):
        super().__init__(game_matrix)
        self.name = f"NKMarkov{N},{K}"
        self.K = K
        self.N = N
        self.reset()

    def make_move(self) -> int:
        if self.full_hist in self.p_matrix:
            opp_distrib = self.p_matrix[self.full_hist]/(self.p_matrix[self.full_hist].sum())
        else:
            opp_distrib = np.ones((self.n_moves))/self.n_moves
        opp_move = np.random.choice(self.n_moves, p=opp_distrib)
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        if self.full_hist not in self.p_matrix: # if havent encountered, assume uniform
            self.p_matrix[self.full_hist] = np.ones((self.n_moves))
        self.p_matrix[self.full_hist][other_move] += 1
        self.our_hist = self.our_hist[1:] + str(my_move)
        self.their_hist = self.their_hist[1:] + str(other_move)
        self.full_hist = self.our_hist + self.their_hist

    def reset(self):
        self.p_matrix = {} # transition matrix. hopefully save some memory by not storing the full table
        self.our_hist = "".join(map(str, np.random.choice(self.n_moves, size=self.K))) # maps strings to distributions of the opponents next move
        self.their_hist = "".join(map(str, np.random.choice(self.n_moves, size=self.N)))
        self.full_hist = self.our_hist + self.their_hist

class NKMAPMarkovAgent(IteratedGamePlayer):
    # basically NMAPMarkovAgent but also tracks our own history. the hope is that it will be be able
    # to beat NMAPMarkovAgents since it can model how that opponent will do. K is the hyperparamter
    # corresponding to how much of our own history we should track. At each step both our move
    # history and their move history is updated. The first N trits of the number will correspond
    # to the opponents move history and the remaining K trits correspond to our move history.
    def __init__(self, game_matrix: np.ndarray, N=2, K=2):
        super().__init__(game_matrix)
        self.name = f"NKMAPMarkov{N},{K}"
        self.K = K
        self.N = N
        self.reset()

    def make_move(self) -> int:
        if self.full_hist in self.p_matrix:
            opp_distrib = self.p_matrix[self.full_hist]/(self.p_matrix[self.full_hist].sum())
        else:
            opp_distrib = np.ones((self.n_moves))/self.n_moves
        opp_move = np.random.choice(np.where(opp_distrib == opp_distrib.max())[0])
        #print(self.full_hist, "=> PREDICTING (distrib=", opp_distrib, ") ->", opp_move)
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        #print("my_move, their_move", my_move, other_move)
        if self.full_hist not in self.p_matrix: # if havent encountered, assume uniform
            #print("CACHE MISS", self.full_hist)
            self.p_matrix[self.full_hist] = np.ones((self.n_moves))
        #print(self.full_hist, self.p_matrix[self.full_hist])
        self.p_matrix[self.full_hist][other_move] += 1
        #print("UPDATED DISTRIB", self.full_hist, self.p_matrix[self.full_hist])
        self.our_hist = self.our_hist[1:] + str(my_move)
        self.their_hist = self.their_hist[1:] + str(other_move)
        self.full_hist = self.our_hist + self.their_hist

    def reset(self):
        self.p_matrix = {} # transition matrix. hopefully save some memory by not storing the full table
        self.our_hist = "".join(map(str, np.random.choice(self.n_moves, size=self.K))) # maps strings to distributions of the opponents next move
        self.their_hist = "".join(map(str, np.random.choice(self.n_moves, size=self.N)))
        self.full_hist = self.our_hist + self.their_hist
        #if self.N == 1 and self.K == 1:
        #    print("RESETTING")

class NMAPMarkovAgent(IteratedGamePlayer):
    # basically NMarkovAgent but it takes the MAP estimate of the distibution
    # rather just sampling from it
    def __init__(self, game_matrix: np.ndarray, N=5):
        super().__init__(game_matrix)
        self.name = f"NMAPMarkovAgent{N}" # probably want a balance between N being large (able to model high order complexity)
        self.max_idx = self.n_moves**N-1 # and N being small (dont make the matrix too sparse)
        self.reset()

    def make_move(self) -> int:
        opp_move = np.random.choice(np.where(self.p_matrix[self.idx] == self.p_matrix[self.idx].max())[0])
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        self.p_matrix[self.idx, other_move] += 1
        self.idx = (self.idx*self.n_moves)%(self.max_idx+1) # wipes out the highest order digit
        self.idx += other_move  # pushes the new move onto the state stack

    def reset(self):
        self.p_matrix = np.ones((self.max_idx+1, self.n_moves)) # assume uniform prior
        self.idx = np.random.randint(self.max_idx)  # base n_moves repr of the last_n moves

class NMarkovAgent(IteratedGamePlayer):
    # generalization of MarkovAgent that instead takes "state" as the last sequence
    # of n moves by the opponent. Then, given that the opponent is in some state,
    # it picks the most likely transition state. We reduce the size of transition
    # matrix by ignoring impossible state trasitions that would modify the history
    # of the opponents moves and instead model possible transition states as
    # being the next move the opponent will take. It samples randomly from all
    # moves that are considered to be most likely as the next move and then
    # plays the move that beats that move
    def __init__(self, game_matrix: np.ndarray, N=5):
        super().__init__(game_matrix)
        self.name = f"NMarkovAgent{N}" # probably want a balance between N being large (able to model high order complexity)
        self.max_idx = self.n_moves**N-1 # and N being small (dont make the matrix too sparse)
        self.reset()

    def make_move(self) -> int:
        opponent_distrib = self.p_matrix[self.idx]/(self.p_matrix[self.idx].sum()) # normalize into a distribution
        opp_move = np.random.choice(self.n_moves, p=opponent_distrib)
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        #print("state inc", self.idx, other_move)
        self.p_matrix[self.idx, other_move] += 1
        self.idx = (self.idx*self.n_moves)%(self.max_idx+1) # wipes out the highest order digit
        #print("state intermeditae", self.idx)
        self.idx += other_move  # pushes the new move onto the state stack
        #print("state post", self.idx)

    def reset(self):
        self.p_matrix = np.ones((self.max_idx+1, self.n_moves)) # assume uniform prior
        self.idx = np.random.randint(self.max_idx)  # base n_moves repr of the last_n moves

class MarkovAgent(IteratedGamePlayer):
    # more sophisticated version of the PredictorAgent that models transition
    # probabilities for its opponent, and then does whatever beats that.
    # Each row in the transition probability matrix corresponds to distribution
    # of what the opponents next move will be given they are in the current state,
    # where state is defined as being the last move they played. The predicted move
    # is chosen randomly, using the distribution of moves given the current state.
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.reset()
        #self.p_matrix = np.ones((self.n_moves, self.n_moves)) # assume uniform prior
        #self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        #opponent_move = np.random.choice(np.where(self.predictor_chances == self.predictor_chances.max())[0])
        opponent_distrib = self.p_matrix[self.last_move]/(self.p_matrix[self.last_move].sum())
        opp_move = np.random.choice(3, p=opponent_distrib)
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        self.p_matrix[self.last_move, other_move] += 1
        self.last_move = other_move

    def reset(self):
        self.p_matrix = np.ones((self.n_moves, self.n_moves)) # assume uniform prior
        self.last_move = np.random.randint(self.n_moves)

class MAPMarkovAgent(IteratedGamePlayer):
    # identical to the MarkovAgent but instead of randomly sampling from the predicted
    # opponents distribution, it takes the MAP estimate of it as the opponents move.
    # If there is a tie in terms of the next predicted move, then it randomly samples
    # from those
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.reset()
        #self.p_matrix = np.ones((self.n_moves, self.n_moves)) # assume uniform prior
        #self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        opp_move = np.random.choice(np.where(self.p_matrix[self.last_move] == self.p_matrix[self.last_move].max())[0])
        return (opp_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        self.p_matrix[self.last_move, other_move] += 1
        self.last_move = other_move

    def reset(self):
        self.p_matrix = np.ones((self.n_moves, self.n_moves)) # assume uniform prior
        self.last_move = np.random.randint(self.n_moves)

class AntiPredictor(IteratedGamePlayer):
    # Such a strategy might do well against an human. Essentially,
    # it tracks the same metrcis as PredictorAgent, but assumes the opponent will
    # play whatever move it has played the least, and then plays whatever
    # move beats that
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.reset()

    def make_move(self) -> int:
        opponent_move = np.random.choice(np.where(self.predictor_chances == self.predictor_chances.min())[0])
        return (opponent_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        self.predictor_chances[other_move] += 1

    def reset(self):
        self.predictor_chances = np.zeros((self.n_moves))


class PredictorPredictor(IteratedGamePlayer):
    # Simulates whatever PredictorAgent will do and then beats that.
    # Again, any ties that exist are broken randomly
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.predictor_chances = np.zeros((self.n_moves)) # simulates what PredictorAgent will do and then beats that

    def make_move(self) -> int:
        opponent_move = np.random.choice(np.where(self.predictor_chances == self.predictor_chances.max())[0])
        return (opponent_move+2)%self.n_moves

    def update_results(self, my_move, other_move):
        self.predictor_chances[my_move] += 1

    def reset(self):
        self.predictor_chances = np.zeros((self.n_moves))


class PredictorAgent(IteratedGamePlayer):
    # Keeps track of whatever the most commonly played moves of the opponent are
    # and assumes the opponent will keep playing whatever move its played the most.
    # If there are ties, it randomly picks from the most likely moves
    def __init__(self, game_matrix: np.ndarray):
        super(PredictorAgent, self).__init__(game_matrix)
        self.opponent_chances = np.zeros((self.n_moves)) # prior distrib

    def make_move(self) -> int:
        opponent_move = np.random.choice(np.where(self.opponent_chances == self.opponent_chances.max())[0])
        #print(np.where(opponent_move == opponent_move.max())[0])
        return (opponent_move+1)%self.n_moves

    def update_results(self, my_move, other_move):
        self.opponent_chances[other_move] += 1

    def reset(self):
        self.opponent_chances = np.zeros((self.n_moves))

class SlowResponder(IteratedGamePlayer):
    # Very similar to CopyCat but repeat the move that was played two
    # moves ago rather than  one move ago
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.reset()

    def make_move(self) -> int:
        return self.last_move_2

    def update_results(self, my_move, other_move):
        self.last_move_2 = self.last_move
        self.last_move = other_move

    def reset(self):
        self.last_move = np.random.randint(self.n_moves)
        self.last_move_2 = np.random.randint(self.n_moves)

class NSlowResponder(IteratedGamePlayer):
    # Generalization of SlowResponder but to N moves
    def __init__(self, game_matrix: np.ndarray, N=2):
        super().__init__(game_matrix)
        self.name = f"NSlowResponder{N}"
        self.max_size = self.n_moves**N-1
        #print("max", self.max_size)
        self.divisor = self.n_moves**(N-1)
        #print("div", self.divisor)
        self.reset()

    def make_move(self) -> int:
        #print(self.state, self.state//self.divisor)
        return self.state//self.divisor

    def update_results(self, my_move, other_move):
        #print("state now", self.state, "incoming mov", other_move)
        self.state = (self.state*self.n_moves)%(self.max_size+1)
        #print("state intermediate", self.state)
        self.state += other_move
        #print("state after", self.state)

    def reset(self):
        self.state = np.random.randint(self.max_size)

class CycleAgent(IteratedGamePlayer):
    # simple agent that just cycles through the possible moves
    # mostly just a test case to see that our algorithm can beat simple moves
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.next_move = 0

    def make_move(self) -> int:
        return self.next_move

    def update_results(self, my_move, other_move):
        self.next_move = (self.next_move+1)%(self.n_moves)

    def reset(self):
        self.next_move = 0

class EnsembleAgent(IteratedGamePlayer):
    # Combines the capabilites of many agents. Basically it simulates what a bunch of agents
    # would be doing and whichever agent is having the best success against the opponent is what
    # is chosen as the strategy for the agent. If it encounters some agent that it does really poorly
    # against (ie some strategy I didnt think of that can predict what my agents do very well) then
    # it switches to a random strategy to avoid losing too badly (since no agent can consistently
    # win against random agents). If you want to read what all the agents do, their documentation
    # is specified in each of their class definitions. The scocre of each agent is computed using
    # an exponential weighted moving average of score the agent would have received.
    def __init__(self, game_matrix: np.ndarray):
        super().__init__(game_matrix)
        self.gm = game_matrix
        self.agents = [ # the ensemble of agents
            UniformPlayer(game_matrix),                 #0
            NAntiMAPMarkov(game_matrix, N=1),           #1
            NAntiMAPMarkov(game_matrix, N=2),           #2
            NAntiMAPMarkov(game_matrix, N=3),           #3
            NAntiMAPMarkov(game_matrix, N=4),           #4
            NAntiMAPMarkov(game_matrix, N=5),           #5
            NAntiMAPMarkov(game_matrix, N=6),           #6
            NAntiMAPMarkov(game_matrix, N=7),           #7
            NAntiMAPMarkov(game_matrix, N=8),           #8
            NKMAPMarkovAgent(game_matrix, N=0, K=2),    #9
            NKMAPMarkovAgent(game_matrix, N=1, K=1),    #10
            NKMAPMarkovAgent(game_matrix, N=2, K=1),    #11
            NKMAPMarkovAgent(game_matrix, N=2, K=2),    #12
            NKMAPMarkovAgent(game_matrix, N=3, K=3),    #13
            NMAPMarkovAgent(game_matrix, N=2),          #14
            NMAPMarkovAgent(game_matrix, N=3),          #15
            NMAPMarkovAgent(game_matrix, N=4),          #16
            NMAPMarkovAgent(game_matrix, N=5),          #17
            NMAPMarkovAgent(game_matrix, N=6),          #18
            NMAPMarkovAgent(game_matrix, N=7),          #19
            NKMarkovAgent(game_matrix, N=1, K=1),       #20
            NKMarkovAgent(game_matrix, N=2, K=1),       #21
            NKMarkovAgent(game_matrix, N=1, K=2),       #22
            NKMarkovAgent(game_matrix, N=2, K=2),       #23
            NKMAPMarkovAgent(game_matrix, N=3, K=3),    #24
            NKMAPMarkovAgent(game_matrix, N=4, K=4),    #25
            NKMAPMarkovAgent(game_matrix, N=5, K=5),    #26
            NKMAPMarkovAgent(game_matrix, N=6, K=6),    #27
            NKMAPMarkovAgent(game_matrix, N=7, K=7),    #28
            NKMAPMarkovAgent(game_matrix, N=8, K=8),    #29
            NKMAPMarkovAgent(game_matrix, N=9, K=9),    #30
            NKMAPMarkovAgent(game_matrix, N=10, K=10),  #31
            NKMAPMarkovAgent(game_matrix, N=11, K=11),  #32
            NKMAPMarkovAgent(game_matrix, N=12, K=12),  #33
            NKMAPMarkovAgent(game_matrix, N=13, K=13),  #34
        ]
        self.reset()

    def make_move(self) -> int:
        self.agent_moves = [ag.make_move() for ag in self.agents]
        best_agent = np.random.choice(np.where(self.agent_scores == self.agent_scores.max())[0])
        return self.agent_moves[best_agent]

    def update_results(self, my_move, other_move):
        for i, agent in enumerate(self.agents):
            agent.update_results(my_move, other_move)
            self.agent_scores[i] *= 0.9 # decay factor
            self.agent_scores[i] += self.gm[self.agent_moves[i], other_move] # how the agent would have done

    def reset(self):
        for ag in self.agents:
            ag.reset()
        self.agent_scores = np.zeros(len(self.agents))
        self.agent_scores[10] += 1000  # bias the ensemble towards NKMAPMarkovAgents

if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    agents = [UniformPlayer,
              FirstMovePlayer,
              CopycatPlayer,
              PredictorAgent,
              SlowResponder,
              PredictorPredictor,
              MAPMarkovAgent,
              MarkovAgent,
              AntiPredictor,
              (NMarkovAgent, {"N":2}),
              (NMAPMarkovAgent, {"N":2}),
              (NMarkovAgent, {"N":3}),
              (NMAPMarkovAgent, {"N":3}),
              (NMAPMarkovAgent, {"N":4}),
              (NMAPMarkovAgent, {"N":5}),
              (NMarkovAgent, {"N":6}),
              (NMAPMarkovAgent, {"N":6}),
              (NMAPMarkovAgent, {"N":7}),
              (NMarkovAgent, {"N":8}),
              (NMAPMarkovAgent, {"N":8}),
              (NMAPMarkovAgent, {"N":9}),
              (NMarkovAgent, {"N":10}),
              (NMAPMarkovAgent, {"N":10}),
              (NSlowResponder,{"N":2}),
              (NSlowResponder,{"N":3}),
              (NSlowResponder,{"N":4}),
              (NSlowResponder,{"N":5}),
              (NSlowResponder,{"N":6}),
              (NSlowResponder,{"N":7}),
              (NKMarkovAgent, {"N":0, "K":2}),
              (NKMarkovAgent, {"N":1, "K":1}),
              (NKMarkovAgent, {"N":2, "K":1}),
              (NKMarkovAgent, {"N":1, "K":2}),
              (NKMarkovAgent, {"N":2, "K":2}),
              (NKMarkovAgent, {"N":3, "K":2}),
              (NKMarkovAgent, {"N":2, "K":3}),
              (NKMarkovAgent, {"N":3, "K":3}),
              (NKMAPMarkovAgent, {"N":0, "K":2}),
              (NKMAPMarkovAgent, {"N":1, "K":1}),
              (NKMAPMarkovAgent, {"N":2, "K":1}),
              (NKMAPMarkovAgent, {"N":1, "K":2}),
              (NKMAPMarkovAgent, {"N":2, "K":2}),
              (NKMAPMarkovAgent, {"N":3, "K":2}),
              (NKMAPMarkovAgent, {"N":2, "K":3}),
              (NKMAPMarkovAgent, {"N":3, "K":3}),
              (NAntiMAPMarkov, {"N":1}),
              (NAntiMAPMarkov, {"N":2}),
              (NAntiMAPMarkov, {"N":3}),
              (NAntiMAPMarkov, {"N":4}),
              (NAntiMAPMarkov, {"N":5}),
              (NAntiMAPMarkov, {"N":6}),
              (NAntiMAPMarkov, {"N":7}),
              (NAntiMAPMarkov, {"N":8}),
              EnsembleAgent,
              CycleAgent,
             ]
    tournament(agents, game_matrix, 10)
