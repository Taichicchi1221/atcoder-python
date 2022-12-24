# author:  Taichicchi
# created: 24.12.2022 11:53:33

import sys

s = input()
t = input()

N = len(s)
M = len(t)

ans1 = []
pos_j = 0
for i in range(N):
    for j in range(pos_j, M):
        if s[i] == t[j]:
            print(f"{i=}, {j=}")
            ans1.append(s[i])
            pos_j = j + 1
            break

print(ans1)

ans2 = []
pos_i = 0
for j in range(M):
    for i in range(pos_i, N):
        if s[i] == t[j]:
            print(f"{i=}, {j=}")
            ans2.append(s[i])
            pos_i = i + 1
            break

print(ans2)
