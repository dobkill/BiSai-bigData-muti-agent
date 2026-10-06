"""Maximum-weight one-to-one assignment, including unassigned dummy columns."""
from math import inf


def maximum_assignment(weights: list[list[float]], forbidden=None) -> tuple[list[int | None], float]:
    """Hungarian assignment; a zero-weight dummy lets every row remain unmatched.

    None/nonpositive real edges are unusable. Input order must be stable when
    callers want deterministic provisional choices among equally good solutions.
    """
    n = len(weights)
    if not n:
        return [], 0.0
    width = len(weights[0])
    if any(len(row) != width for row in weights):
        raise ValueError("assignment matrix must be rectangular")
    blocked = set(forbidden or [])
    costs = [[-value if value > 0 and (i, j) not in blocked else 1e9
              for j, value in enumerate(row)] + [0.0] * n for i, row in enumerate(weights)]
    m = width + n
    u, v, p, way = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv, used = [inf] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], inf, 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    selected = [None] * n
    for j in range(1, width + 1):
        if p[j] and costs[p[j] - 1][j - 1] < 0:
            selected[p[j] - 1] = j - 1
    total = sum(weights[i][j] for i, j in enumerate(selected) if j is not None)
    return selected, round(total, 6)
