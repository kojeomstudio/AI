def solution(A):
    N = len(A)
    if N < 3:
        return 0

    left_max_arr = [0] * N
    right_max_arr = [0] * N

    # 1st pass: 왼쪽 최대값 누적
    left_max_arr[0] = A[0]
    for i in range(1, N):
        left_max_arr[i] = max(left_max_arr[i-1], A[i])

    # 2nd pass: 오른쪽 최대값 누적
    right_max_arr[N-1] = A[N-1]
    for i in range(N-2, -1, -1):
        right_max_arr[i] = max(right_max_arr[i+1], A[i])

    # 각 위치에서 물 깊이 계산
    max_depth = 0
    for i in range(N):
        water_depth = min(left_max_arr[i], right_max_arr[i]) - A[i]
        if water_depth > max_depth:
            max_depth = water_depth

    return max_depth