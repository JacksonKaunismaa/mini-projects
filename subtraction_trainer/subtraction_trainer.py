#!/usr/bin/env python3
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cmaps
import pickle
import argparse
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--num_q", type=int, default=10)
parser.add_argument("--mode", choices=["pos", "mul", "any"], default="any")
parser.add_argument("--max_size", type=int, default=10)
opt = parser.parse_args()
num_questions, max_range = opt.num_q, opt.max_size
MIN_MUL = 2  # inclusive

pos_mode = opt.mode == "pos"
mul_mode = opt.mode == "mul"

curr_dir = osp.dirname(osp.realpath(__file__))
if pos_mode:
    save_file = osp.join(curr_dir, "pos.pkl")
elif mul_mode:
    save_file = osp.join(curr_dir, "mul.pkl")
else:
    save_file = osp.join(curr_dir, "any.pkl")
correct = 0

try:
    with open(save_file, "rb") as p:
        times, correctness = pickle.load(p)
except FileNotFoundError:
    correctness = {}
    times = {}

def get_int_resp(q):
    while True:
        resp = input(question)
        try:
            return int(resp)
        except ValueError:
            print("\tInvalid response, try again")

def all_possible():
    if mul_mode:
        mul_problems = []
        for n1 in range(MIN_MUL, max_range):
            for n2 in range(n1, 100//n1):
                if max(n1, n2) <= 10:
                    continue
                if n1 == 2 and n2 <= 36:
                    continue
                mul_problems.append((n1,n2))
        return mul_problems
    elif pos_mode:
        return [(n1,n2) for n1 in range(0,max_range) for n2 in range(0, max_range)]
    else:
        raise NotImplementedError

def generate_question(curr_q):
    if pos_mode:
        q = np.random.choice(len(possible_questions))
        n1,n2 = possible_questions[q]
        question = f"{n1} - {n2} = ? "
        idx = n1,n2
        if n2 > n1:
            answer = n1+10-n2
        else:
            answer = n1-n2
    elif mul_mode:
        order = np.random.random() < 0.5
        q = np.random.choice(len(possible_questions))
        n1,n2 = possible_questions[q]
        answer = n1*n2
        if order:
            n1,n2 = n2,n1
        question = f"{n1} * {n2} = ? "
        idx = min(n1,n2), max(n1,n2)
    else:
        raise NotImplementedError
    actual_question = f"({curr_q}/{num_questions}): {question}"
    return idx, actual_question, answer


possible_questions = all_possible()
try:
    for i in range(num_questions):
        start_time = time.perf_counter()
        idx,question,answer = generate_question(i+1)
        resp = get_int_resp(question)

        if idx not in times:
            times[idx] = []
        if idx not in correctness:
            correctness[idx] = [0,0]

        if resp == answer:
            correct += 1
            print("\tCorrect!")
            correctness[idx][0] += 1
        else:
            print("\tWrong, correct was", answer)
            correctness[idx][1] += 1
        times[idx].append(time.perf_counter() - start_time)
finally:
    with open(save_file, "wb") as p:
        pickle.dump((times,correctness),p)

print(f"Session Number Correct {correct}/{num_questions}")
if not mul_mode:
    pct_grid = np.full((max_range, max_range), np.nan)
    time_grid = np.full((max_range, max_range), np.nan)
else:
    num_factors = max(times, key=lambda t: t[0])[0]
    pct_grid = np.full((num_factors, 100//MIN_MUL), np.nan)
    time_grid = np.full((num_factors, 100//MIN_MUL), np.nan)
for i in range(pct_grid.shape[0]):
    for j in range(pct_grid.shape[1]):
        if (i,j) in correctness and sum(correctness[i,j]) != 0:
            pct_grid[i,j] = correctness[i,j][0]/sum(correctness[i,j])
        if (i,j) in times and len(times[i,j]) > 0:
            time_grid[i,j] = sum(times[i,j]) / len(times[i,j])

cmap = cmaps.viridis.copy()
cmap.set_bad("white")

ratio = pct_grid.shape[0] / pct_grid.shape[1]
plt.subplot(2,1,1)
if mul_mode:
    plt.ylim(MIN_MUL, time_grid.shape[0])
    plt.yticks(np.arange(MIN_MUL, pct_grid.shape[0], step=2))
im = plt.imshow(pct_grid, cmap=cmap)
plt.title("Percent correct")
plt.colorbar(im, fraction=0.046*ratio, pad=0.04)

plt.subplot(2,1,2)
time_max = np.nanpercentile(time_grid, 95)
print("95th percentile time:", time_max)
im = plt.imshow(time_grid, cmap=cmap, vmax=time_max, vmin=0)
if mul_mode:
    plt.ylim(MIN_MUL, time_grid.shape[0])
    plt.yticks(np.arange(MIN_MUL, time_grid.shape[0], step=2))
plt.title("Average time")
plt.colorbar(im, fraction=0.046*ratio, pad=0.04)
plt.show()
