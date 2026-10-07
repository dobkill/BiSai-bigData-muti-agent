"""Exact maximum-weight rectangular assignment with explicit unmatched columns."""


def maximum_assignment(weights: list[list[float]]) -> list[int]:
    if not weights:
        return []
    n, m = len(weights), len(weights[0])
    if m < n or any(len(row) != m for row in weights):
        raise ValueError("assignment requires rectangular matrix with columns >= rows")
    u, v = [0.0]*(n+1), [0.0]*(m+1)
    p, way = [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")]*(m+1)
        used = [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m+1):
                if not used[j]:
                    cost = -weights[i0-1][j-1]-u[i0]-v[j]
                    if cost < minimum[j]:
                        minimum[j], way[j] = cost, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [-1]*n
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result
