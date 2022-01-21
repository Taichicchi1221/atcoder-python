# author:  Taichicchi
# created: 21.01.2022 17:11:42

import sys
from math import comb

A, B, K = map(int, input().split())

S = comb(A + B, A)

ans = ""
for i in range(A + B):
    if A == 0:
        r = "b"
    elif B == 0:
        r = "a"

    elif K <= comb(A + B - 1, A - 1):
        r = "a"
        A -= 1
    else:
        r = "b"
        K -= comb(A + B - 1, A - 1)
        B -= 1

    ans += r

print(ans)
