import numpy as np
import random
import string
from tqdm import tqdm
import time
#import multiprocessing as mp

with open("allowed_words.txt", "r") as f:
    allowed_words = f.read().split("\n")

with open("answer_words.txt", "r") as f:
    answer_words = f.read().split("\n")
#answer_words = answer_words[:100]
#allowed_words = allowed_words[:200]
answer_words = answer_words[:700]
allowed_words = answer_words.copy()

class Profiler():
    def __init__(self, gamma=0.99):
        self.prev_time = 0
        self.gamma = gamma
        self.stats = {}  # tracks ewma running average
        self.profile = True
        self.verbose = False

    def benchmark(self, point=None): # not thread safe at all
        if not self.profile:
            return
        if point is not None:
            time_taken = time.perf_counter() - self.prev_time
            if point not in self.stats:
                self.stats[point] = [time_taken, 0]  # avg_time, num_times
            self.stats[point][1] += 1
            self.stats[point][0] = self.stats[point][0]*self.gamma + time_taken*(1-self.gamma)
            if self.verbose:
                print(f"took {time_taken} to reach {point}, ewma={self.stats[point]}")
        self.prev_time = time.perf_counter()

    def __repr__(self):
        sum_avgs = sum([x[0] for x in self.stats.values()])
        sum_time = sum([x[0]*x[1] for x in self.stats.values()])
        ret_str = "point\tpct_avg\tpct_cumulative"
        for point,stat in self.stats.items():
            ret_str += f"\n{point}\t{stat[0]/sum_avgs*100.:.3f}%\t{stat[0]*stat[1]/sum_time*100.:.3f}%"
        return ret_str

def bit_count(arr):
    return arr.bit_count()

def bitwise_not(arr):
    return ~arr

vectorized_count = np.vectorize(bit_count)
vectorized_not = np.vectorize(bitwise_not)

class FastWordList():
    def __init__(self, batched=False):
        self.batched = batched
        self.reset()

    def reset(self):
        if self.batched:
            self.remaining_words = vectorized_not(np.zeros(batched_arr_size).astype(np_type))
        else:
            self.remaining_words = np.zeros(arr_size).astype(np_type)

    def restore_state(self, prev_remaining=None):
        if prev_remaining is not None:
            self.remaining_words = prev_remaining.copy()
        else:
            self.remaining_words = self.prev_remaining.copy()

    def save_state(self):
        self.prev_remaining = self.remaining_words.copy()

    def update_list(self, green_letters, yellow_letters, grey_letters, curr_round):
        # want to do the updates that remove the most first
        g_profiler.benchmark()
        if green_letters:
            for let,pos in green_letters:
                if self.batched:
                    self.remaining_words &= batched_pos_let_to_words[pos][let]
                else:
                    self.remaining_words[pos_let_to_words[pos][let]] = 0  # remove words that dont have let in position pos
        g_profiler.benchmark(f"green_letters_{curr_round}")
        if yellow_letters:
            for let in yellow_letters:
                if self.batched:
                    self.remaining_words &= batched_letter_to_words[let]
                else:
                    self.remaining_words[letter_to_not_words[let]] = 0  # remove all words that don't contain the required letters
        g_profiler.benchmark(f"yellow_letters_{curr_round}")
        if grey_letters:
            for let in grey_letters:
                if self.batched:
                    self.remaining_words &= batched_letter_to_not_words[let]
                else:
                    self.remaining_words[letter_to_words[let]] = 0  # remove all words that contain greyed out letters
        g_profiler.benchmark(f"grey_letters_{curr_round}")

    #def pop(self):
    #    return self.

    #def rand_select(self):
    #    elem = self.remaining_words.pop()
    #    self.remaining_words.add(elem)
    #    return elem

    def __contains__(self, word):
        if self.batched:
            bucket,bit_mask = batched_word_to_idx[word]
            return self.remaining_words[bucket] & bit_mask
        else:
            idx = word_to_idx[word]
            return self.remaining_words[idx]

    def __len__(self):
        if self.batched:
            return vectorized_count(self.remaining_words).sum()
        else:
            return self.remaining_words.sum()

class AntiWordle():
    INVALID_WORD = -1
    CONTINUE = -2

    def __init__(self, batched=False):
        self.remaining_words = FastWordList(batched=batched)
        self.batched = batched
        self.reset()

    def reset(self):
        self.answer = random.choice(answer_words)
        self.remaining_words.reset()
        self.round = 0
        self.allowed_letters = set(string.ascii_lowercase)
        self.forced_letters = {}
        self.speculating = False

   # def guess_valid(self, word):
   #     #if word not in allowed_words:  # assume strategies don't do this, avoid an O(n) check
   #     #    return "Not in word list!"
   #     if set(word) - self.allowed_letters:
   #         return False #"Guess contains invalid letters"
   #     for let,positions in self.forced_letters.items():
   #         if positions is None:
   #             if let not in word:  # yellow letters
   #                 return False #f"Answer must contain {let}"
   #             continue
   #         for pos in positions:
   #             if word[pos] != let:  # green letters
   #                 return False #f"Answer must contain {let}"
   #     return True

    def fast_guess_valid(self, word):
        return word in self.remaining_words

    def speculate(self):  # for efficiency reasons
        self.speculating = True

    def unspeculate(self):
        self.speculating = False

    def save_state(self):
        self.remaining_words.save_state()
        return self.forced_letters.copy(), self.remaining_words.prev_remaining

    def restore_state(self, saved_forced_letters, prev_remaining):
        self.remaining_words.restore_state(prev_remaining=prev_remaining)
        self.forced_letters = self.saved_forced_letters.copy()

    def update_constraints(self, word):  # does not really work with double letters
        remove_these = set()
        green_letters = []
        yellow_letters = []
        grey_letters = []
        for i, (answer_let, guess_let) in enumerate(zip(self.answer, word)):
            if answer_let == guess_let:
                if answer_let not in self.forced_letters or self.forced_letters[answer_let] is None:
                    self.forced_letters[answer_let] = []
                if i not in self.forced_letters[answer_let]:
                    self.forced_letters[answer_let].append(i)
                    green_letters.append((answer_let, i))
            elif guess_let in self.answer:
                if guess_let not in self.forced_letters:
                    self.forced_letters[guess_let] = None
                    yellow_letters.append(guess_let)
            else:
                grey_letters.append(guess_let)
                remove_these.add(guess_let)
        if not self.speculating:
            self.allowed_letters = self.allowed_letters - remove_these
        if not self.speculating and not self.batched:
            self.remaining_words.update_list(green_letters, yellow_letters, grey_letters, self.round)
        return green_letters, yellow_letters, grey_letters

    def play_word(self, word):
        g_profiler.benchmark()
        if not self.fast_guess_valid(word): #self.guess_valid(word):
            g_profiler.benchmark("validating_word")
            return self.INVALID_WORD
        g_profiler.benchmark("validating_word")
        self.round += 1
        if word == self.answer:
            return self.round
        self.update_constraints(word)
        g_profiler.benchmark("updating_update_constraints")
        return self.CONTINUE

    def __repr__(self):
        answer_display = ["_"] * len(self.answer)
        must_contain = []
        for let,positions in self.forced_letters.items():
            if positions is None:
                must_contain.append(let)
                continue
            for pos in positions:
                answer_display[pos] = let.upper()
        good_letters = sorted(list(self.allowed_letters))
        return f"{self.round} {''.join(answer_display)}\nMust contain: {must_contain}\nRemaining Letters: {''.join(good_letters)}\nAnswer:{self.answer}"


class RandStrategy():
    def __init__(self):
        self.reset()

    def reset(self):
        pass #self.remaining_words = fast_allowed.copy()

    def guess(self, game):
        g_profiler.benchmark()
        return game.remaining_words.rand_select()

class GreedyStrategy():
    def __init__(self):
        pass

    def reset(self):
        self.word_to_expected_len = {w: 0 for w in allowed_words}

    def canonicalize_updates(self, green_letters, yellow_letters, grey_letters):
        #return "".join((f"{x[0]}{x[1]}" for x in green_letters)) + "|" + \
        #        "".join(sorted(yellow_letters)) + "|" + "".join(sorted(grey_letters))
        return f"{green_letters}|{yellow_letters}|{grey_letters}"


    def explore(self, game):  # use num remaining words as proxy for game length
        game.speculate()
        state_begin = game.save_state()
        tbl = {}
        for potential_answer in tqdm(answer_words):
            game.answer = potential_answer
            for potential_guess in allowed_words:
                guess_idx = word_to_idx[potential_guess]
                if not game.remaining_words.prev_contains(guess_idx):
                    continue
                update_rules = game.update_constraints(potential_guess)
                updates = self.canonicalize_updates(*update_rules)
                if updates not in tbl:
                    game.remaining_words.update_list(*update_rules, game.round)
                    tbl[updates] = len(game.remaining_words)
                #print(potential_guess, word_to_expected_len, tbl)
                self.word_to_expected_len[potential_guess] += tbl[updates]
                game.restore_state(*state_begin)
        game.unspeculate()

    def save_self(self):
        import pickle
        with open("word_to_expected_len.dict", "wb") as p:
            pickle.dump(self.word_to_expected_len, p)

    #def guess_mp(self, game):
    #    num_procs = 6
    #    batch_size = len(answer_words)//num_procs
    #    split_answers = [(answer_words[i:i+batch_size], AntiWordle()) for i in range(0,len(answer_words), batch_size)]
    #    split_answers[-1] = (answer_words[num_procs*batch_size:], AntiWordle())
    #    with mp.Pool(processes=num_procs) as pool:
    #        results = pool.map(self.guess_thr, split_answers)
    #    import pickle
    #    with open("results.pickle", "wb") as p:
    #        pickle.dump(results, p)
    #    merged_results = {}
    #    for res in results:
    #        for k,v in res.items():
    #            merged_results[k] = v
    #    best_word = max(merged_results, key=lambda x: x[1])
    #    print("best_word", best_word)
    #    return best_word

class GreedyExpected():
    def __init__(self, num_rollouts=30):
        self.num_rollouts = num_rollouts
        self.reset()

    def reset(self):
        self.word_to_game_lens = {w: [] for w in allowed_words}

    def rollout_game(self, game: AntiWordle):
        for word in tqdm(allowed_words):  # change to be priority list of these, and probably to be more explorative
            if word in game.remaining_words:
                break
        result = game.play_word(word)
        if result > 0:
            return result
        return self.rollout_game(game)

    def guess(self, game):
        game.speculate()
        start_state = game.save_state()
        for potential_answer in tqdm(answer_words):
            game.answer = potential_answer
            for potential_guess in allowed_words:
                guess_idx = word_to_idx[potential_guess]
                if not game.remaining_words.prev_contains(guess_idx):
                    continue
                state_now = game.speculate()
                game.play_word(potential_guess) # we are now in a specific (answer, [first_word]) state
                for _ in range(self.num_rollouts): # num rollouts
                    self.word_to_game_lens[potential_guess].append(self.rollout_game(game))
                    game.restore_state(*state_now)
                game.restore_state(*start_state)

    def save_self(self):
        import pickle
        with open("distributions.dict", "wb") as p:
            pickle.dump(self.word_to_game_lens, p)

class HumanStrategy():
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def guess(self, game):
        print(game)
        human = input("Guess word: ").lower()
        while human not in allowed_words:
            print(f"*{allowed_words[200]}*, #{human}#")
            print("Word is not in allowed word list")
            print(game)
            human = input("Guess word: ").lower()
        return human

def play_antiwordle(game, strategy):
    game.reset()
    result = game.CONTINUE
    while result < 0: # either CONTINUE or INVALID_WORD
        word_guess = strategy.guess(game)
        #print("Attempting guess of", word_guess)
        #print(game)
        result = game.play_word(word_guess)
        break
    return result

def evaluate_strategy(strategy, iterate=True, profile=False):
    total = 0
    game = AntiWordle()
    g_profiler.profile = profile
    if not iterate:
        looper = answer_words[:1]
    else:
        looper = answer_words
    for answer in tqdm(looper):
        strategy.reset()
        game.reset()
        game.answer = answer
        total += play_antiwordle(game, strategy)
        break
    print("Average number of rounds lasted:", total/len(answer_words))
    if profile:
        print(g_profiler)

# pre-processing step
word_size, np_type = 8, np.uint8

# for masked/packed mode of addressing (each word has its own bit within some element of the array) => 2 indices needed
batched_word_to_idx = {word:(i//word_size,1<<(i%word_size)) for i,word in enumerate(allowed_words)}
batched_idx_to_word = {i:word for word,i in batched_word_to_idx.items()}
batched_arr_size = int(np.ceil(len(allowed_words)/word_size))

# for index-based addressing (each word has its own element in the array) => 1 index needed
word_to_idx = {word:i for i,word in enumerate(allowed_words)}
idx_to_word = {i:word for word,i in word_to_idx.items()}
arr_size = len(allowed_words)

# mask, where we pack many booleans into 1 element of the array
batched_pos_let_to_words = [{} for _ in range(len(allowed_words[0]))]  # map pos,let -> set of words with let in position pos
batched_letter_to_words = {}  # maps letter -> set of words that contain that letter
batched_letter_to_not_words = {} # maps letter -> set of words that don't contain that letter

# indices
pos_let_to_words = [{} for _ in range(len(allowed_words[0]))]  # map pos,let -> set of words WITHOUT let in position pos
letter_to_words = {}  # maps letter -> set of words that contain that letter
letter_to_not_words = {} # maps letter -> set of words that don't contain that letter

for let in tqdm(string.ascii_lowercase):
    for pos in range(len(allowed_words[0])):
        arr = np.zeros(arr_size).astype(np_type)
        for word, idx in word_to_idx.items():
            words_list = []
            if word[pos] == let:
                arr[idx[0]] |= idx[1]
            else:
                words_list.append(word)
        batched_pos_let_to_words[pos][let] = arr
        pos_let_to_words[pos][let] = np.array(words_list)

        #arr[list((word_to_idx[w] for w in (filter(lambda w: w[pos] == let, allowed_words))))] = 1
        #pos_let_to_words[pos][let] = arr
        #pos_let_to_words[pos][let] = np.array(list((word_to_idx[w] for w in (filter(lambda w: w[pos] == let, allowed_words)))))
    #arr2 = np.zeros((len(allowed_words))).astype(np.uint8)
    #arr2[list((word_to_idx[w] for w in (filter(lambda w: let in w, allowed_words))))] = 1
    #letter_to_words[let] = arr2
    #letter_to_words[let] = np.array(list((word_to_idx[w] for w in (filter(lambda w: let in w, allowed_words)))))
    #letter_to_not_words[let] = np.array(list((word_to_idx[w] for w in (filter(lambda w: let not in w, allowed_words)))))
    arr2 = np.zeros(arr_size).astype(np_type)
    arr3 = np.zeros(arr_size).astype(np_type)
    words_list = []
    not_words_list = []
    for word,idx in batched_word_to_idx.items():
        if let in word:
            arr2[idx[0]] |= idx[1]
            words_list.append(word)
        else:
            arr3[idx[0]] |= idx[1]
            not_words_list.append(word)
    batched_letter_to_words[let] = arr2
    batched_letter_to_not_words[let] = arr3
    letter_to_words[let] = np.array(words_list)
    letter_to_not_words[let] = np.array(not_words_list)
#evaluate_strategy(RandStrategy(), profile=False)
#evaluate_strategy(HumanStrategy(), runs=10, profile=False)
g_profiler = Profiler()
evaluate_strategy(GreedyStrategy(), iterate=False, profile=False)

