import numpy as np
import matimg
import matplotlib.pyplot as pyplt
import os
import time

"""Simple kmeans classifier that also includes some functions to generate random data for testing"""

def update_means(x, y, K):
    new_means = np.zeros((K, x.shape[1]))
    for k in range(K):
        categorized = x[np.where(y == k)]
        new_means[k] = categorized.sum(axis=0) / categorized.shape[0]
    return new_means

def forgy(x, k):
    return x[np.random.choice(x.shape[0], size=k)]

def rand_part(x, k):
    partition = np.random.choice(k, size=x.shape[0])
    return update_means(x, partition, k)


class KMeans(object):
    def __init__(self, k, data, logdir=f"./kmviz/{time.time()}",init_method=rand_part):
        self.k = k
        self.init_method = init_method
        self.data = data
        self.p = data.shape[0]
        self.N = data.shape[1]
        self.means = init_method(self.data, self.k)
        self.plt = matimg.Plot(res=2000, block=15)
        self.logdir = logdir
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def almost_equal(self, old, new, abs_eps=1e-8, rel_eps=1e-10):
        diff = abs(new-old)
        return np.max(diff/(new+old)) <= rel_eps or np.max(diff) <= abs_eps

    def categorize(self):
        diffs = [((self.data - mean)**2).sum(axis=1) for mean in self.means]
        return np.argmin(np.vstack(diffs), axis=0)

    def slow_vis(self):
        cats = self.categorize()
        for an_k in range(2):
            points = self.data[np.where(cats==an_k)]
            pyplt.scatter(points[:,0], points[:,1])
            pyplt.scatter(self.means[an_k, 0], self.means[an_k, 1], c='k')
        pyplt.show()

    def qck_vis(self, cats, it):
        self.plt.clear()
        self.plt.set_frame(self.data)
        self.plt.colorize(self.means)
        self.plt.plot_points(self.data, cats)
        self.plt.plot_means(self.means)
        self.plt.save(os.path.join(self.logdir, "%04d.png" % it))

    def train(self, rec=False):
        old_means = self.means.copy()
        cats = self.categorize()
        if rec:
            self.qck_vis(cats, 0)
        self.means = update_means(self.data, cats, self.k)
        cats = self.categorize()
        if rec:
            self.qck_vis(cats, 1)
        cnt = 2
        while not self.almost_equal(old_means, self.means):
            old_means = self.means.copy()
            cats = self.categorize()
            self.means = update_means(self.data, cats, self.k)
            if rec:
                self.qck_vis(cats, cnt)
            cnt += 1

def gen_rand_means(dimensions, starting_mean, amount):
    return np.random.multivariate_normal(starting_mean, np.identity(dimensions)/5.0, size=amount)

def gen_rand_point(dimensions, the_mean):
    return np.random.multivariate_normal(the_mean, np.identity(dimensions)/10.0)

def gen_rand_data(dimensions, classes, ppc):
    num_means = 10
    full_data = []
    cnt = 0
    for i in range(-5, 6, 2):
        for j in range(-1, 2, 2):
            batch_means = gen_rand_means(dimensions, np.array([i, j]), num_means)
            points = []
            for _ in range(ppc):
                an_mean = batch_means[np.random.choice(num_means)]
                an_point = gen_rand_point(dimensions, an_mean)
                points.append(an_point)
            full_data += points
            cnt += 1
            if cnt == classes:
                break
        if cnt == classes:
            break
    return np.array(full_data)


def main():
    test_data = gen_rand_data(2, 4, 50)
    km = KMeans(4, test_data)
    km.train(rec=True)

if __name__ == "__main__":
    main()
