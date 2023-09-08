def solution(n):
    table = [1] + [0]*(n)
    for brick in range(1, n+1):
        for height in range(n, brick-1, -1):
            table[height] += table[height - brick]
    return table[-1]-1

print(solution(3))