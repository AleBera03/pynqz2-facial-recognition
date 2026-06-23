import os

N = 1000
FILENAME = "calibr.txt"

with open(FILENAME, "w") as f:
    for i in range(N):
        if i == N - 1:
            f.write(f"{i}.jpg 0")
        else:
            f.write(f"{i}.jpg 0\n")