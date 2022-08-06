# author:  Taichicchi
# created: 03.07.2022 15:00:00

import sys
import copy
import math
import random
from time import perf_counter


class Input():
    def __init__(self, N, K, a, x, y, START_TIME):
        self.N = N
        self.K = K
        self.a = a
        self.x = x
        self.y = y
        self.START_TIME = START_TIME


class Output():
    px: list
    py: list
    qx: list
    qy: list

    def __init__(self, px, py, qx, qy):
        self.px = px
        self.py = py
        self.qx = qx
        self.qy = qy

    def __len__(self):
        return len(self.px)


def compute(input, output):
    pieces = [list(range(input.N))]
    for px, py, qx, qy in zip(output.px, output.py, output.qx, output.qy):
        new_pieces = []
        for piece in pieces:
            left = []
            right = []
            for j in piece:
                x, y = input.x[j], input.y[j]
                side = (qx - px) * (y - py) - (qy - py) * (x - px)
                if side > 0:
                    left.append(j)
                elif side < 0:
                    right.append(j)

            if len(left) > 0:
                new_pieces.append(left)
            if len(right) > 0:
                new_pieces.append(right)

        pieces = new_pieces

    b = [0] * 10
    for piece in pieces:
        if len(piece) <= 10:
            b[len(piece) - 1] += 1

    num = 0
    den = 0
    for d in range(10):
        num += min(input.a[d], b[d])
        den += input.a[d]

    score = round((1e+06 * num) / den)

    return score, "", (b, pieces)


def make_vertical(x):
    return x, 0, x, 1


def make_horizon(y):
    return 0, y, 1, y


def make_random_line():
    r = random.randint(0, 1)
    if r == 0:
        return make_vertical(random.randint(-10**4, 10**4))
    elif r == 1:
        return make_horizon(random.randint(-10**4, 10**4))


class Annealing(object):
    def __init__(self, inputs):
        self.input = inputs

    @staticmethod
    def calc_temp(temp, start_temp, end_temp, start_time, now_time):
        return start_temp + (end_temp - start_temp) * (now_time - start_time)

    def modify_state(self, state):
        ret_state = Output([], [], [], [])

        for idx, output in enumerate(zip(state.px, state.py, state.qx, state.qy)):
            px, py, qx, qy = output
            r = random.randint(-250, 250)
            if idx < 50:
                px = min(10**4, max(-10**4, px + r))
                qx = min(10**4, max(-10**4, qx + r))
            else:
                py = min(10**4, max(-10**4, py + r))
                qy = min(10**4, max(-10**4, qy + r))

            ret_state.px.append(px)
            ret_state.py.append(py)
            ret_state.qx.append(qx)
            ret_state.qy.append(qy)

        ret_state.px = ret_state.px[:self.input.K]
        ret_state.py = ret_state.py[:self.input.K]
        ret_state.qx = ret_state.qx[:self.input.K]
        ret_state.qy = ret_state.qy[:self.input.K]

        return ret_state

    def annealing(self, init_state, TIME_LIMIT=2.0):
        # settings
        start_temp = 100
        end_temp = 1
        temp = start_temp
        state = copy.deepcopy(init_state)

        pre_score = compute(self.input, state)[0]
        init_score = pre_score

        max_state = copy.deepcopy(init_state)
        max_score = pre_score

        step = 0
        start_time = perf_counter()

        while True:
            now_time = perf_counter()
            step += 1
            if now_time - start_time > TIME_LIMIT:
                break

            modified_state = self.modify_state(state)
            score = compute(self.input, modified_state)[0]
            diff = score - pre_score

            # temp
            temp = self.calc_temp(
                temp, start_temp, end_temp, start_time, now_time,
            )

            # prob
            try:
                prob = math.exp(diff / temp)
            except:
                prob = 100000.0

            if diff >= 0:
                state = copy.deepcopy(modified_state)
            elif prob >= random.uniform(0, 1):
                state = copy.deepcopy(modified_state)
                score = pre_score

            if score > max_score:
                max_score = score
                max_state = copy.deepcopy(modified_state)

        print(
            f"annealing result: step={step}, init_score={init_score}, max_score={max_score}",
            file=sys.stderr,
        )

        return state


def make_output(input, length, TIME_LIMIT=None):
    step = 0
    max_score = 0
    max_output = Output([], [], [], [])

    time_limit = TIME_LIMIT if TIME_LIMIT is not None else 1
    start_time = perf_counter()

    while perf_counter() - start_time < time_limit:
        step += 1
        output = Output([], [], [], [])

        for k in range(length):
            px, py, qx, qy = make_random_line()
            output.px.append(px)
            output.py.append(py)
            output.qx.append(qx)
            output.qy.append(qy)

        score = compute(input, output)[0]
        if score > max_score:
            max_output = output
            max_score = score

        # iterate only once
        if TIME_LIMIT is None:
            break

    print(
        f"make output result: step={step}, max_score={max_score}", file=sys.stderr,
    )

    return max_output


def make_output_uniform(input):
    output = Output([], [], [], [])
    rng = list(map(lambda x: x + 200, range(-10000, 10000, 400)))

    for x in rng:
        px, py, qx, qy = make_vertical(x)
        output.px.append(px)
        output.py.append(py)
        output.qx.append(qx)
        output.qy.append(qy)

    for y in rng:
        px, py, qx, qy = make_horizon(y)
        output.px.append(px)
        output.py.append(py)
        output.qx.append(qx)
        output.qy.append(qy)

    return output


def greedy(input, init_output, TIME_LIMIT=None):
    step = 0
    max_score = compute(input, init_output)[0]
    max_output = copy.deepcopy(init_output)

    time_limit = TIME_LIMIT if TIME_LIMIT is not None else 1
    start_time = perf_counter()

    while perf_counter() - start_time < time_limit:
        if len(max_output) >= input.K:
            break

        step += 1
        output = copy.deepcopy(max_output)
        px, py, qx, qy = make_random_line()
        output.px.append(px)
        output.py.append(py)
        output.qx.append(qx)
        output.qy.append(qy)

        score = compute(input, output)[0]

        if score > max_score:
            max_score = score
            max_output = copy.deepcopy(output)

    print(f"greedy: step={step}, max_score={max_score}", file=sys.stderr)

    return max_output


def main():
    random.seed(42)
    START_TIME = perf_counter()

    N, K = map(int, input().split())
    a = list(map(int, input().split()))
    x, y = zip(*[map(int, input().split()) for _ in range(N)])

    inputs = Input(N, K, a, x, y, START_TIME)

    # randomly make initial output
    # make_output(input, length=100, TIME_LIMIT=0.5)

    # greedy make initial output
    # outputs = make_output(inputs, length=1, TIME_LIMIT=None)
    # outputs = greedy(inputs, outputs, TIME_LIMIT=2.5)

    # uniformly made output
    outputs = make_output_uniform(inputs)

    # annealing
    annearling = Annealing(inputs)
    outputs = annearling.annealing(init_state=outputs, TIME_LIMIT=2.5)

    # post greedy
    # outputs = greedy(inputs, outputs, TIME_LIMIT=1.5)

    assert \
        len(outputs.px) == \
        len(outputs.py) == \
        len(outputs.qx) == \
        len(outputs.qy)

    print(len(outputs.px))
    for output in zip(outputs.px, outputs.py, outputs.qx, outputs.qy):
        print(*output)

    score = compute(inputs, outputs)[0]
    print(score, file=sys.stderr)


if __name__ == "__main__":
    main()
