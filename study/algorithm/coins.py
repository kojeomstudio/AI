# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

def solution(K, A):

    memorize = [0] * (K + 1)

    memorize[0] = 1

    for coin in A:
        for step in range(coin, K + 1):
            memorize[step] += memorize[step - coin]
    
    return memorize[K]