def solution(K, A):
    count = 0
    length = 0

    memorize = [0] * (len(A) + 1)
    memorize[0] = 1

    for rope in A:
        length += rope

        if count >= memorize[length]:
            count = memorize[length]
        elif length >= K:
            count += 1
            memorize[length] = count
            length = 0

    return count
