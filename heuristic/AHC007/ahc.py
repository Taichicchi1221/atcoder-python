# author:  Taichicchi
# created: 11.06.2022 10:39:29

import sys
import math
import copy
import time
import random
from itertools import combinations
from collections import namedtuple


Edge = namedtuple("Edge", "idx start end weight")


class DSU:
    def __init__(self, n):
        self._n = n
        self.parent_or_size = [-1] * n

    def merge(self, a, b):
        assert 0 <= a < self._n
        assert 0 <= b < self._n
        x, y = self.leader(a), self.leader(b)
        if x == y:
            return x
        if -self.parent_or_size[x] < -self.parent_or_size[y]:
            x, y = y, x
        self.parent_or_size[x] += self.parent_or_size[y]
        self.parent_or_size[y] = x
        return x

    def same(self, a, b):
        assert 0 <= a < self._n
        assert 0 <= b < self._n
        return self.leader(a) == self.leader(b)

    def leader(self, a):
        assert 0 <= a < self._n
        if self.parent_or_size[a] < 0:
            return a
        self.parent_or_size[a] = self.leader(self.parent_or_size[a])
        return self.parent_or_size[a]

    def size(self, a):
        assert 0 <= a < self._n
        return -self.parent_or_size[self.leader(a)]

    def groups(self):
        leader_buf = [self.leader(i) for i in range(self._n)]
        result = [[] for _ in range(self._n)]
        for i in range(self._n):
            result[leader_buf[i]].append(i)
        return [r for r in result if r != []]


def dist(p1, p2):
    d = round(math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)))
    return d


def random_dist(p1, p2):
    d = dist(p1, p2)
    return random.randrange(round(d * 1.5), round(d * 2.5))


def sim(N, M, p, queries, use_random_dist=True):
    # edges
    edges = []
    for m, (u, v) in enumerate(queries):
        if use_random_dist:
            d = random_dist(p[u], p[v])
        else:
            d = dist(p[u], p[v])
        edges.append(Edge(m, u, v, d))

    edges.sort(key=lambda x: x.weight)

    dsu = DSU(N)

    # kruskal method
    kruskal = []
    ans = [0] * M
    for edge in edges:
        if not dsu.same(edge.start, edge.end):
            dsu.merge(edge.start, edge.end)
            kruskal.append(edge)
            ans[edge.idx] = 1

    # calc score
    score = calc_score(N, M, p, edges, ans)

    return ans, score


def calc_score(N, M, p, edges, ans):
    score = 0
    for edge in edges:
        if ans[edge.idx]:
            score += edge.weight

    return score


def main(START_TIME):

    N = 400
    M = 1995

    p = [tuple(map(int, input().split())) for _ in range(N)]
    queries = [tuple(map(int, input().split())) for _ in range(M)]

    ans_ls = []
    while time.time() - START_TIME < 1.0:
        ans, score = sim(N, M, p, queries, use_random_dist=True)
        ans_ls.append((ans, score))

    ans_ls.sort(key=lambda x: x[1])
    ans = ans_ls[0][0]

    # output
    for m in range(M):
        l = int(input())
        print(ans[m], flush=True)


if __name__ == "__main__":
    START_TIME = time.time()
    main(START_TIME)
