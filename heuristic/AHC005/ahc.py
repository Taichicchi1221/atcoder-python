# author:  Taichicchi
# created: 10.06.2022 21:25:18

from copy import copy
import sys
import random
from time import perf_counter

DIJ = ((0, -1), (-1, 0), (0, 1), (1, 0))
DIR = ("L", "U", "D", "R")
INF = 1 << 32
INVERSE_DIR = {
    "L": "R",
    "R": "L",
    "U": "D",
    "D": "U",
}


def parse(x):
    if x == "#":
        return -1

    return int(x)


def get_visited(inputs, outputs):
    visited = [[False] * inputs.N for _ in range(inputs.N)]
    pi, pj = inputs.si, inputs.sj
    length = 0
    ps = [(pi, pj)]
    err = ""

    for c in outputs:
        idx = DIR.index(c)
        pi += DIJ[idx][0]
        pj += DIJ[idx][0]
        if not (0 <= pi < inputs.N and 0 <= pj < inputs.N) or inputs.c[pi][pj] == "#":
            err = "Visiting an obstacle."
            break

        length += inputs.c[pi][pj]
        ps.append((pi, pj))

    for pi, pj in ps:
        for d in range(4):
            for k in range(inputs.N + 100):
                i = pi + DIJ[d][0] * k
                j = pj + DIJ[d][1] * k
                if 0 <= i < inputs.N and 0 <= j < inputs.N and inputs.c[i][j] != "#":
                    visited[i][j] = True
                else:
                    break

    return visited, length, ps, err


def compute_score(inputs, outputs):
    visited, length, ps, err = get_visited(inputs, outputs)

    if err:
        return 0, err

    num = 0
    den = 0

    for i in range(inputs.N):
        for j in range(inputs.N):
            if inputs.c[i][j] != "#":
                den += 1
                if visited[i][j]:
                    num += 1

    if ps[-1][0] != inputs.si or ps[-1][1] != inputs.sj:
        return 0, "You have to go back to the starting point"

    score = 1e4 * num / den
    if num == den:
        score += 1e7 * inputs.N / length

    return round(score), ""


def check_crossroad(inputs, i, j):
    c = 4
    for di, dj in DIJ:
        i2 = i + di
        j2 = j + dj
        if not (0 <= i2 < inputs.N and 0 <= j2 < inputs.N):
            c -= 1
            continue
        if inputs.c[i2][j2] == -1:
            c -= 1

    return int(c >= 3)


class Input(object):
    def __init__(self, N, si, sj, c):
        self.N = N
        self.si = si
        self.sj = sj
        self.c = c


def main():
    N, si, sj = map(int, input().split())
    c = [list(map(parse, list(input()))) for _ in range(N)]
    print(c)

    inputs = Input(N, si, sj, c)

    arr = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if c[i][j] == -1:
                arr[i][j] = 1

    print(*arr, sep="\n")


if __name__ == "__main__":
    main()
