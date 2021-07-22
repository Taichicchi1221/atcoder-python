# author:  Taichicchi
# created: 19.06.2021 21:07:26

import sys

N = int(input())
A = list(map(int, input().split()))

A_l = A[:N // 2]
A_r = A[N // 2 + int(N % 2):][::-1]

l = [A_l[i] for i in range(len(A_l)) if not A_l[i] == A_r[i]]
r = [A_r[i] for i in range(len(A_r)) if not A_l[i] == A_r[i]]

print(len(l) - 1)
