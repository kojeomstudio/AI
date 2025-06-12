from collections import deque

def solution(A):
    N = len(A)

    fib = [0, 1]
    while fib[-1] + fib[-2] <= N + 1:
        fib.append(fib[-1] + fib[-2])
    fib = fib[2:] 

    queue = deque()
    visited = [False] * (N + 1)

    for jump in fib:
        next_pos = jump - 1 

        if next_pos == N:
            return 1
        if 0 <= next_pos < N and A[next_pos] == 1:
            queue.append((next_pos, 1)) 
            visited[next_pos] = True

    while len(queue) > 0:
        curr, jumps = queue.popleft()
        for jump in fib:
            next_pos = curr + jump
            if next_pos == N:
                return jumps + 1
            if 0 <= next_pos < N and A[next_pos] == 1 and not visited[next_pos]:
                visited[next_pos] = True
                queue.append((next_pos, jumps + 1))

    return -1
