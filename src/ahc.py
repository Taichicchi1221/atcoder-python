# author:  Taichicchi
# created: 28.05.2022 12:00:00

from collections import Counter, deque
import os
import sys
import copy
import random
import re
import math
from itertools import product, combinations
from time import perf_counter

sys.setrecursionlimit(10**6)

try:
    import pypyjit

    pypyjit.set_param("max_unroll_recursion=-1")
except ModuleNotFoundError:
    pass

INF = 1 << 64

DIJ = [(0, -1), (-1, 0), (0, 1), (1, 0)]
DIR = ["L", "U", "R", "D"]

INV_DIR_MAP = {
    "L": "R",
    "R": "L",
    "D": "U",
    "U": "D",
}

DIR_DICT = dict(zip(DIR, DIJ))
DIR_DICT_INV = dict(zip(DIJ, DIR))


class DSU:
    def __init__(self, n):
        self._n = n
        self.parent_or_size = [-1] * n

    def merge(self, a, b):
        assert 0 <= a < self._n, f"a = {a} is not valid."
        assert 0 <= b < self._n, f"b = {b} is not valid."
        x, y = self.leader(a), self.leader(b)
        if x == y:
            return x
        if -self.parent_or_size[x] < -self.parent_or_size[y]:
            x, y = y, x
        self.parent_or_size[x] += self.parent_or_size[y]
        self.parent_or_size[y] = x
        return x

    def same(self, a, b):
        assert 0 <= a < self._n, f"a = {a} is not valid."
        assert 0 <= b < self._n, f"b = {b} is not valid."
        return self.leader(a) == self.leader(b)

    def leader(self, a):
        assert 0 <= a < self._n, f"a = {a} is not valid."
        if self.parent_or_size[a] < 0:
            return a
        self.parent_or_size[a] = self.leader(self.parent_or_size[a])
        return self.parent_or_size[a]

    def size(self, a):
        assert 0 <= a < self._n, f"a = {a} is not valid."
        return -self.parent_or_size[self.leader(a)]

    def groups(self):
        leader_buf = [self.leader(i) for i in range(self._n)]
        result = [[] for _ in range(self._n)]
        for i in range(self._n):
            result[leader_buf[i]].append(i)
        return [r for r in result if r != []]


class Node:
    def __init__(self, i, j, cost, parent, num):
        self.i = i
        self.j = j
        self.state = 1  # 0:none 1:open 2:closed
        self.score = 0
        self.cost = cost
        self.parent = parent
        self.expect_cost = 0
        self.num = num
        self.calculated = 0

    def close(self):
        self.state = 2


class Astar:
    def __init__(self, gi, gj, si, sj, obstacle):
        self.N = len(obstacle)
        self.gi = gi
        self.gj = gj
        self.si = si
        self.sj = sj
        self.i = si
        self.j = sj
        self.obstacle_list = copy.deepcopy(obstacle)
        self.maked_list = []
        self.num = 0

        start = Node(si, sj, 0, -1, self.num)
        self.Node_list = [start]
        self.num = self.num + 1
        self.now = start  # 現在のノード
        self.route = []
        self.goal = -1  # gaal の　ノード
        self.finished = 0  # goal したかどうか

        if gi == si and gj == sj:
            self.finished == 1
            self.goal = start
            self.route = [[si, sj]]

    def open(self):
        self.now.close()
        # 周りをopen
        """
        壁・障害　が有るときはopen できない　－＞obstacle_list
        既に作っていないか？－＞maked_list 
        """
        cost = self.now.cost
        parent = self.now.num

        for di, dj in DIJ:
            i2, j2 = self.i + di, self.j + dj
            if not (0 <= i2 < self.N and 0 <= j2 < self.N):
                continue

            if self.maked_list.count((i2, j2)) == 0 and self.obstacle_list[i2][j2] == 0:
                self.Node_list.append(Node(i2, j2, cost + 1, parent, self.num))
                self.num = self.num + 1
                self.maked_list.append((i2, j2))

        # open しているものを計算
        for node in self.Node_list:
            if node.state == 1 and node.calculated == 0:
                node.calculated = 1
                node.expect_cost = abs(node.i - self.gi) + abs(node.j - self.gj)
                node.score = node.cost + node.expect_cost

        # open しているもののうち、スコアの小さいものをリストにまとめる
        min_cost = INF
        min_cost_list = []
        for node in self.Node_list:
            if node.state == 1:
                if node.cost < min_cost:
                    min_cost = node.cost
                    min_cost_list = [node]
                elif node.cost == min_cost:
                    min_cost_list.append(node)

        if min_cost_list:
            self.now = random.choice(min_cost_list)
            self.i = self.now.i
            self.j = self.now.j
        else:
            return -1

        if self.now.i == self.gi and self.now.j == self.gj:
            return 1
        else:
            return 0

    def explore(self):
        """
        0 :goal
        -1:goal できない
        """
        if self.finished == 1:
            return 0
        else:
            while True:
                hoge = self.open()
                if hoge == 1:
                    self.goal = self.now
                    self.finished = 1
                    self.Route()
                    return 0
                elif hoge == -1:
                    return -1

    def Route(self):
        if self.finished == 1:
            while True:
                self.route.append((self.now.i, self.now.j))
                if self.now.parent == -1:
                    break
                else:
                    self.now = self.Node_list[self.now.parent]

            self.route.reverse()

            self.move_str = []
            for idx in range(len(self.route) - 1):
                di = self.route[idx + 1][0] - self.route[idx][0]
                dj = self.route[idx + 1][1] - self.route[idx][1]
                self.move_str.append(DIR_DICT_INV[(di, dj)])
            self.move_str = "".join(self.move_str)

    def Express(self):
        if self.finished == 1:
            if self.route:
                graph = self.obstacle_list
                for r in self.route:
                    graph[r[0]][r[1]] = 2

                print(*graph, sep="\n")
            else:
                print("not goaled")


class Sim(object):
    def __init__(self, inputs):
        self.N = inputs.N
        self.T = inputs.T
        self.i = INF
        self.j = INF
        self.tiles = copy.deepcopy(inputs.tiles)
        self.fixed = [[0] * self.N for _ in range(self.N)]
        self.fr = [[0] * self.N for _ in range(self.N)]
        self.turn = 0
        self.move_str = []

        for x in range(self.N):
            for y in range(self.N):
                if inputs.tiles[x][y] == 0:
                    self.i = x
                    self.j = y
                self.fr[x][y] = (x, y)

    def get_move_str(self):
        return "".join(self.move_str)

    def is_able_to_apply(self, c):
        if c not in DIR:
            return False

        i2 = self.i + DIR_DICT[c][0]
        j2 = self.j + DIR_DICT[c][1]

        if not (0 <= i2 < self.N and 0 <= j2 < self.N):
            return False

        if self.fixed[i2][j2]:
            return False

        return True

    def apply(self, c):
        if c not in DIR:
            raise ValueError(f"{c} not in DIR: {DIR}")

        i2 = self.i + DIR_DICT[c][0]
        j2 = self.j + DIR_DICT[c][1]

        if not (0 <= i2 < self.N and 0 <= j2 < self.N):
            raise ValueError(f"illegal move: {c} (turn {self.turn})")

        self.tiles[self.i][self.j], self.tiles[i2][j2] = (
            self.tiles[i2][j2],
            self.tiles[self.i][self.j],
        )

        f1 = self.fr[self.i][self.j]
        f2 = self.fr[i2][j2]
        self.fr[i2][j2] = f1
        self.fr[self.i][self.j] = f2
        self.i = i2
        self.j = j2
        self.turn += 1
        self.move_str.append(c)
        return c

    def compute_score(self, inputs):
        tiles = [[0] * self.N for _ in range(self.N)]

        for i in range(self.N):
            for j in range(self.N):
                tiles[i][j] = inputs.tiles[self.fr[i][j][0]][self.fr[i][j][1]]

        dsu, max_tree, bs = compute_dsu_maxtree_bs(inputs, tiles)

        if self.turn > self.T:
            return 0, f"too many moves", bs

        size = 0 if max_tree == -1 else dsu.size(max_tree)

        score = (
            round(500000.0 * (1.0 + (self.T - self.turn) / self.T))
            if size == self.N**2 - 1
            else round(500000.0 * size / (self.N**2 - 1))
        )

        return score, "", bs

    def move_0(self, i, j):
        """
        0マスをi, jに動かす
        """
        if i == self.i and j == self.j:
            return -1

        astar = Astar(i, j, self.i, self.j, self.fixed)
        result = astar.explore()
        move_str = ""
        if result == 0:
            move_str = astar.move_str
        else:
            return -1

        for c in move_str:
            self.apply(c)

        if self.i != i or self.j != j:
            return -1
        else:
            return move_str

    def move_x(self, i, j, c):
        """
        i, jにあるtileをc方向に移動する
        """
        if c == "D":
            if i == self.N - 1:
                return -1
            self.fixed[i][j] = 1
            result = self.move_0(i + 1, j)
            self.fixed[i][j] = 0
            if result == -1:
                return -1
            self.apply("U")
        if c == "R":
            if j == self.N - 1:
                return -1
            self.fixed[i][j] = 1
            result = self.move_0(i, j + 1)
            self.fixed[i][j] = 0
            if result == -1:
                return -1
            self.apply("L")
        if c == "L":
            if j == 0:
                return -1
            self.fixed[i][j] = 1
            result = self.move_0(i, j - 1)
            self.fixed[i][j] = 0
            if result == -1:
                return -1
            self.apply("R")
        if c == "U":
            if i == 0:
                return -1
            self.fixed[i][j] = 1
            result = self.move_0(i - 1, j)
            self.fixed[i][j] = 0
            if result == -1:
                return -1
            self.apply("D")

        return 0

    def operate(self, i, j, i2, j2):
        """
        i, jにあるtileをi2, j2に移動してfixする
        """

        if i == i2 and j == j2:
            return 0

        astar = Astar(i2, j2, i, j, self.fixed)
        result = astar.explore()
        if result == 0:
            move_str = astar.move_str
        else:
            return -1

        for c in move_str:
            self.move_x(i, j, c)
            i += DIR_DICT[c][0]
            j += DIR_DICT[c][1]

        self.fixed[i2][j2] = 1

        return 0

    def operate2(self, gi, gj, t):
        """
        gi, gjに一番近いかつunfixedなtをセットする
        """
        deq = deque([(gi, gj)])
        arr = [[0] * self.N for _ in range(self.N)]

        while deq:
            i, j = deq.popleft()
            arr[i][j] = 1
            for di, dj in DIJ:
                i2 = i + di
                j2 = j + dj
                if not (0 <= i2 < self.N and 0 <= j2 < self.N):
                    continue
                if arr[i2][j2]:
                    continue

                if self.tiles[i2][j2] == t and self.fixed[i2][j2] == 0:
                    return self.operate(i2, j2, gi, gj)

                deq.append((i2, j2))

        return -1

    # def operate3(gi1, gj1, t1, gi2, gj2, t2):
    #     """
    #     端っこ部分を2つ一気にそろえる
    #     """
    #     assert gi1 <= gi2 and gj1 <= gj2
    #     assert gi1 == gi2 or gj1 == gj2

    #     if abs(gi1 - gi2) == 1:

    #     elif abs(gj1 - gj2) == 1:

    #     else:
    #         raise NotImplementedError("(^ω^#)")


def compute_dsu_maxtree_bs(inputs, tiles):
    dsu = DSU(inputs.N**2)
    tree = [True] * (inputs.N**2)
    for i in range(inputs.N):
        for j in range(inputs.N):
            if i + 1 < inputs.N and tiles[i][j] & 8 != 0 and tiles[i + 1][j] & 2 != 0:
                a = dsu.leader(i * inputs.N + j)
                b = dsu.leader((i + 1) * inputs.N + j)
                if a == b:
                    tree[a] = False
                else:
                    t = tree[a] and tree[b]
                    dsu.merge(a, b)
                    tree[dsu.leader(a)] = t

            if j + 1 < inputs.N and tiles[i][j] & 4 != 0 and tiles[i][j + 1] & 1 != 0:
                a = dsu.leader(i * inputs.N + j)
                b = dsu.leader(i * inputs.N + j + 1)
                if a == b:
                    tree[a] = False
                else:
                    t = tree[a] and tree[b]
                    dsu.merge(a, b)
                    tree[dsu.leader(a)] = t

    max_tree = -1
    for i in range(inputs.N):
        for j in range(inputs.N):
            if tiles[i][j] != 0 and tree[dsu.leader(i * inputs.N + j)]:
                if max_tree == -1 or dsu.size(max_tree) < dsu.size(i * inputs.N + j):
                    max_tree = i * inputs.N + j

    bs = [[False] * inputs.N for _ in range(inputs.N)]
    if max_tree != -1:
        for i in range(inputs.N):
            for j in range(inputs.N):
                bs[i][j] = dsu.same(max_tree, i * inputs.N + j)

    return dsu, max_tree, bs


def check_connect(t1, t2, i, j, i2, j2):
    # 条件1
    b11 = t1 & 8 != 0 and t2 & 2 != 0
    b12 = i - i2 == -1 and j - j2 == 0
    if b11 and b12:
        return True

    # 条件2
    b21 = t1 & 2 != 0 and t2 & 8 != 0
    b22 = i - i2 == 1 and j - j2 == 0
    if b21 and b22:
        return True

    # 条件3
    b31 = t1 & 4 != 0 and t2 & 1 != 0
    b32 = i - i2 == 0 and j - j2 == -1
    if b31 and b32:
        return True

    # 条件4
    b41 = t1 & 1 != 0 and t2 & 4 != 0
    b42 = i - i2 == 0 and j - j2 == 1
    if b41 and b42:
        return True

    return False


def is_able_to_use_tile(tiles, i, j, t):
    N = len(tiles)
    if i == j == 0 and t not in (4, 8, 12):
        return False
    if i == j == N - 1 and t not in (1, 2, 3):
        return False

    if i == 0 and j == N - 1 and t not in (1, 8, 9):
        return False
    if i == N - 1 and j == 0 and t not in (2, 4, 6):
        return False

    if i == 0 and (t >> 1) & 1 == 1:
        return False
    if j == 0 and (t >> 0) & 1 == 1:
        return False

    if i == N - 1 and (t >> 3) & 1 == 1:
        return False
    if j == N - 1 and (t >> 2) & 1 == 1:
        return False

    for di, dj in DIJ:
        i2 = i + di
        j2 = j + dj
        if not (0 <= i2 < N and 0 <= j2 < N):
            continue
        if tiles[i2][j2] == 0:
            continue
        if not check_connect(t, tiles[i2][j2], i, j, i2, j2):
            return False

    return True


def compute_score(inputs, outputs):
    sim = Sim(inputs)
    for c in outputs:
        sim.apply(c)
    score, err, tree = sim.compute_score(inputs)
    return score, err, (sim.fr, tree)


class Annealing(object):
    def __init__(self, inputs, state):
        self.inputs = inputs
        self.state = state

    @staticmethod
    def calc_temp(temp, start_temp, end_temp, start_time, now_time):
        return start_temp + (end_temp - start_temp) * (now_time - start_time)

    def modify_state(self):
        state = list(self.state)
        return "".join(state)

    def annealing(self, TIME_LIMIT=2.0):
        ### settings
        start_temp = 1000000
        end_temp = 1
        temp = start_temp
        max_state = self.state
        max_score = 0

        step = 0
        start_time = perf_counter()
        pre_score, _, _ = compute_score(inputs=self.inputs, outputs=self.state)
        while True:
            now_time = perf_counter()
            step += 1
            if now_time - start_time > TIME_LIMIT:
                break

            modified_state = self.modify_state()
            try:
                score, _, _ = compute_score(inputs=self.inputs, outputs=self.state)
            except:
                continue

            diff = score - pre_score

            # temp
            temp = self.calc_temp(temp, start_temp, end_temp, start_time, now_time)

            # prob
            try:
                prob = math.exp(diff / temp)
            except:
                prob = 100000.0

            if diff >= 0:
                self.state = modified_state
            elif prob >= random.uniform(0, 1):
                self.state = modified_state
                score = pre_score

            print(
                f"score={score}, step={step}, time={now_time - start_time}, state={self.state}",
                file=sys.stderr,
            )

            if score > max_score:
                max_score = score
                max_state = modified_state
                print(f"max_score={max_score}", file=sys.stderr)

        self.state = max_state
        return step


class Inputs:
    def __init__(self, N: int, T: int, tiles: list) -> None:
        self.N = N
        self.T = T
        self.tiles = copy.deepcopy(tiles)


def postprocess(inputs, outputs):
    while True:
        outputs = re.sub("UD|DU|LR|RL", "", outputs)

        if not re.search("UD|DU|LR|RL", outputs):
            break

    max_score = 0
    max_return = None
    for i in range(1, len(outputs) + 1):
        score, _, _ = compute_score(inputs, outputs[:i])
        if score > max_score or max_return is None:
            max_score = score
            max_return = outputs[:i]

    return max_return


def step_simulate_tiles(inputs):
    N = inputs.N
    tiles = [[0] * N for _ in range(N)]
    counter = Counter(sum(inputs.tiles, []))
    counter[0] = 0

    t = 12

    ls = [(0, 0, t)]
    arr = [[0] * N for _ in range(N)]
    arr[N - 1][N - 1] = 1

    while ls:
        i, j, k = ls.pop()
        arr[i][j] = 1
        tiles[i][j] = k
        counter[k] -= 1

        for di, dj in DIJ:
            i2 = i + di
            j2 = j + dj
            if not (0 <= i2 < N and 0 <= j2 < N):
                continue
            if arr[i2][j2]:
                continue

            keys = [
                t
                for t in counter.keys()
                if counter[t]
                and check_connect(k, t, i, j, i2, j2)
                and is_able_to_use_tile(tiles, i2, j2, t)
            ]
            if not keys:
                continue

            t = random.choice(keys)
            ls.append((i2, j2, t))

    dsu, max_tree, bs = compute_dsu_maxtree_bs(inputs, tiles)
    size = 0 if max_tree == -1 else dsu.size(max_tree)
    return tiles, size


def get_selection_priority(inputs):
    N = inputs.N
    ret = []
    for i in range(N - 1):
        for j in range(N - 1):
            ret.append((i, j))
    ret.sort(key=lambda x: (x[0] + x[1], x[0] ** 2 + x[1] ** 2))

    return ret


def main():
    START = perf_counter()
    random.seed(1221)

    N, T = map(int, input().split())
    tiles = [list(map(lambda x: int(x, base=16), list(input()))) for _ in range(N)]
    counter = Counter(sum(tiles, []))

    inputs = Inputs(N, T, tiles)

    sim = Sim(inputs)

    max_sim_tiles = None
    max_size = 0
    for _ in range(5000):
        sim_tiles, size = step_simulate_tiles(inputs)
        if size >= max_size:
            max_size = size
            max_sim_tiles = sim_tiles

    selection_priority = get_selection_priority(inputs)

    print("#" * 30, file=sys.stderr)
    print(
        *["".join(map(lambda x: hex(x)[2], t)) for t in max_sim_tiles],
        sep="\n",
        file=sys.stderr,
    )
    for i, j in selection_priority:
        if max_sim_tiles[i][j] == 0:
            continue
        sim.operate2(i, j, max_sim_tiles[i][j])
        sim.fixed[i][j] = 1

    outputs = sim.get_move_str()
    outputs = postprocess(inputs, outputs)
    print(outputs)
    score, _, _ = compute_score(inputs, outputs)
    print(f"score: {score}", file=sys.stderr)


if __name__ == "__main__":
    main()

    # try:
    #     main()
    # except:
    #     print("")
    #     exit()
