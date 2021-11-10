import numpy as np
import operator


class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        if not self.k & 1:
            raise ValueError("K must be odd!")
        self.points = np.array([])
        self.classes = np.array([])

    def input_training(self, points, classes):
        # points should be Nxp, classes Nx1
        self.points = points
        self.classes = classes

    def classify(self, point):
        dists = ((self.points - point)**2).sum(axis=1)
        large_to_small = np.argsort(dists)
        sorted_classes = self.classes[large_to_small]
        chosen_classes = sorted_classes[:self.k]
        closest_classes, class_counts = np.unique(chosen_classes, return_counts=True)
        return closest_classes[np.argmax(class_counts)]




def main():
    import matplotlib.pyplot as plt
    l = np.array([[-2.3, 5.4],
         [10.2, 9.1],
         [-10.2, -9.1],
         [0.2, -0.1],
         [-2.2, 1.1],
         [4, -9.1]])
    c = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([1.0, 1.0])
    k = KNNClassifier(3)
    k.input_training(l, c)
    print(k.classify(p))
    class_0 = l[np.where(c == 0)[0]]
    class_1 = l[np.where(c == 1)[0]]
    plt.scatter(class_0[:,0], class_0[:,1], c="k")
    plt.scatter(class_1[:,0], class_1[:,1], c="r")
    plt.scatter(p[0], p[1], c='b')
    plt.legend(["class 0", "class 1", "classify"])
    plt.show()


if __name__ == "__main__":
    main()
