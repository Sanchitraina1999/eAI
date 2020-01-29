import collections
import io
import itertools as it

import numpy as np
import pandas as pd


class NearestNeighbor(object):
    def __init__(self, X, y, sim_func):
        self.X = X
        self.y = y
        self.sim_func = sim_func

    def classify(self, x):
        scores = self._eval(x)
        idx = np.argmax(scores)
        return self.y[idx], idx

    def _eval(self, x):
        return list(map(self.sim_func, zip(self.X, it.repeat(x))))


def parse_eoa_data():

    eoa_data = """Sanni;boxing gloves;Moby Dick (novel);headphones;sunglasses;coffee beans
Jouni;t-shirt;coffee beans;coffee maker;coffee beans;coffee beans
Janina;sunglasses;sneakers;t-shirt;sneakers;ragg wool socks
Henrik;2001: A Space Odyssey (dvd);headphones;t-shirt;boxing gloves;flip flops
Ville;t-shirt;flip flops;sunglasses;Moby Dick (novel);sunscreen
Teemu;Moby Dick (novel);coffee beans;2001: A Space Odyssey (dvd);headphones;coffee beans"""

    data = pd.read_csv(io.StringIO(eoa_data), sep=";", header=None)
    Xlab = data[0].tolist()
    X = data.loc[:, 1:4].apply(lambda x: collections.Counter(x), axis=1).tolist()
    y = data[5].tolist()

    return (X, y, Xlab)


def answer_exercise():
    X, y, Xlab = parse_eoa_data()
    nn_clf = NearestNeighbor(X, y, sim_func=lambda x: sum((x[0] & x[1]).values()))
    y_pred, idx = nn_clf.classify(
        collections.Counter(["green tea", "t-shirt", "sunglasses", "flip flops"])
    )

    print(f"similar user: {Xlab[idx]}")
    print(f"predicted purchase: {y_pred}")


if __name__ == "__main__":
    answer_exercise()
