# author:  Taichicchi
# created: 19.02.2022 15:44:03

import sys
import random


class Human(object):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

        self.Actions = ".udlrUDLR"

    def u(self, state):
        if not (0 <= self.y - 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        state[self.x][self.y] = 1
        return True

    def d(self, state):
        if not (0 <= self.y + 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        state[self.x][self.y] = 1
        return True

    def l(self, state):
        if not (0 <= self.x - 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        state[self.x][self.y] = 1
        return True

    def r(self, state):
        if not (0 <= self.x - 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        state[self.x][self.y] = 1
        return True

    def U(self, state):
        if not (0 <= self.y - 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        self.y -= 1
        return True

    def D(self, state):
        if not (0 <= self.y + 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        self.y += 1
        return True

    def L(self, state):
        if not (0 <= self.x - 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        self.x -= 1
        return True

    def R(self, state):
        if not (0 <= self.x + 1 < 30):
            return False
        if state[self.x][self.y] is not None:
            return False

        self.x += 1
        return True

    def act(self, state):
        f = False
        while not f:
            a = self.Actions[random.randint(0, len(self.Actions) - 1)]
            if a == ".":
                f = True
                continue
            f = getattr(self, a)(state)

        return a


class Pet(object):
    def __init__(self, t, x, y) -> None:
        self.t = t
        self.x = x
        self.y = y


def main():
    pets = []
    humans = []
    N = int(input())
    for _ in range(N):
        px, py, pt = map(int, input().split())
        px -= 1
        py -= 1
        p = Pet(pt, px, py)
        pets.append(p)

    M = int(input())
    for _ in range(M):
        hx, hy = map(int, input().split())
        hx -= 1
        hy -= 1
        h = Human(hx, hy)
        humans.append(h)

    try:
        input()
    except:
        pass

    state = [[None] * 30 for _ in range(30)]
    for i in range(300):
        res = []
        for human in humans:
            a = human.act(state)
            res.append(a)

        print("".join(res))


if __name__ == "__main__":
    main()
