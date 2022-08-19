import numpy as np

class RNG():
    def __init__(self, func, shape, *params,  N=8192, min_freq=450, max_freq=700, weight=0.9, **kparams):
        self.params = params
        self.kparams = kparams
        self.shape = (N, shape)
        self.func = func
        self.N = 0
        self.i = 0
        # tracking buffer stats
        self.hit_freq = 0
        self.hits = 0
        self.weight = weight
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.dropped = 0
        self.raised = 0

    def generate(self):
        self.hit_freq = (1-self.weight)*self.hit_freq + self.weight*self.hits
        if not self.N:
            self.N = self.shape[0]
        if self.hit_freq < self.min_freq:  # buffer size is too small
            self.N = int(self.N*1.2)
            self.raised += 1
        elif self.hit_freq > self.max_freq: # buffer size is too large
            self.N = int(self.N/1.2)
            self.dropped += 1
        #self.data = np.random.randint(self.lo, self.hi, (self.N, *self.shape[1]))
        self.data = self.func(*self.params, size=(self.N, *self.shape[1]), **self.kparams)
        self.i = 0
        self.hits = 0

    def __repr__(self):
        return f"{self.data}, {self.i}, {self.hits}, {self.hit_freq}"

    def next(self):
        if self.i < self.N:
            result = self.data[self.i]
            self.i += 1
            self.hits += 1
        else:
            self.generate()
            return self.next()
        return result

    def next_n(self, num):
        if self.i + num - 1 < self.N:
            result = self.data[self.i:self.i+num]
            self.i += num
            self.hits += 1
        else:
            self.generate()
            return self.next_n(num)
        return result

def shifting_workload(rng, amount_min, amount_max, switches, iters):
    for _ in range(switches):
        amount = np.random.randint(amount_min, amount_max)
        for _ in range(iters):
            rng.next_n(amount)


def test(settings):
    import time
    repetitive_tests = [(1, 50000), (5, 50000), (10, 50000), (100, 50000), (5000, 10000), (100000, 1000)]
    shifting_tests = [(1,10, 5000, 1000), (10, 100, 500, 1000), (100, 10_000, 100, 1000)]
    times = []
    changes = []

    for amount,iters in repetitive_tests:
        rng = RNG((2,),0,10,**settings)
        start = time.perf_counter()
        shifting_workload(rng, amount,amount+1,1,iters)
        times.append(time.perf_counter() - start)
        changes.append((rng.dropped, rng.raised))

    for params in shifting_tests:
        rng = RNG((2,),0,10,**settings)
        start = time.perf_counter()
        shifting_workload(rng, *params)
        times.append(time.perf_counter() - start)
        changes.append((rng.dropped, rng.raised))

    return times,changes

def plot_results(results, params):
    import matplotlib.pyplot as plt
    times,changes = results
    plt.gcf().clear()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(221)
    ax.plot(times[:6])
    ax.set_title("Repetitive test times")

    ax = fig.add_subplot(223)
    ax.plot([c[0] for c in changes[:6]], label="drops")
    ax.plot([c[1] for c in changes[:6]], label="raises")
    ax.legend()
    ax.set_title("Repetitive test changes")

    ax = fig.add_subplot(222)
    ax.plot(times[6:])
    ax.set_title("Shifting test times")

    ax = fig.add_subplot(224)
    ax.plot([c[0] for c in changes[6:]], label="drops")
    ax.plot([c[1] for c in changes[6:]], label="raises")
    ax.legend()
    ax.set_title("Shifting test changes")
    total_time = sum(times)
    plt.savefig(f"rng_images3/{total_time}_{'-'.join([str(x) for x in params.values()])}.png")

def pool_func(p):
    results = test(p)
    plot_results(results, p)

def optimize(runs):
    import multiprocessing as mp
    tests = []
    for _ in range(runs):
        min_freq = np.random.randint(10,500)
        max_freq = np.random.randint(500,1000)
        weight = np.random.uniform(0.5, 0.99)
        N = np.random.randint(1000, 500_000)
        test_params = {"min_freq": min_freq, "max_freq": max_freq, "weight": weight, "N":N}
        tests.append(test_params)


    with mp.Pool(7) as pool:
        pool.map(pool_func, tests)

def dkl(d1, d2):
    total = 0
    for k in d1:
        total += d1[k]*np.log(d1[k]/d2[k])
    return total

def jsd(d1, d2):
    d1_total = sum(d1.values())
    d2_total = sum(d2.values())
    d1_norm = {k:v/d1_total for k,v in d1.items()}
    d2_norm = {k:v/d2_total for k,v in d2.items()}
    return (dkl(d1_norm, d2_norm) + dkl(d2_norm, d1_norm))/2

def naive_comparison(runs, iters=100_000):
    import time
    from collections import Counter
    rng = RNG(np.random.choice, (5,2), [49, 50, 8512, 20], p=[0.2, 0.3, 0.1, 0.4])
    counts_naive = Counter()
    start = time.perf_counter()
    for _ in range(runs):
        for _ in range(iters):
            counts_naive += Counter(np.random.choice([49, 50, 8512, 20], size=(5,2), p=[0.2, 0.3, 0.1, 0.4]).flatten())
    print("Naive avg:", (time.perf_counter() - start)/runs)
    counts_rng = Counter()
    start = time.perf_counter()
    for _ in range(runs):
        for _ in range(iters):
            counts_rng += Counter(rng.next().flatten())
    print("RNG avg:  ", (time.perf_counter() - start)/runs)
    print("diff:", jsd(counts_rng, counts_naive))  # disable this if distrubitions are continuous

if __name__ == "__main__":
    #optimize(1000)
    naive_comparison(10)
