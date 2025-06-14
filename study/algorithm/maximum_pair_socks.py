from collections import Counter

def solution(K, C, D):
    clean_count = Counter(C)
    dirty_count = Counter(D)
    pairs = 0

    # 1. 깨끗한 양말로 쌍 먼저 계산
    for color in list(clean_count):
        count = clean_count[color]
        pairs += count // 2
        clean_count[color] = count % 2  # 홀수개 남기기 (짝은 제거)

    # 2. 깨끗한 홀수 남은 양말과 dirty에서 매칭
    for color in clean_count:
        if clean_count[color] == 1 and dirty_count[color] > 0 and K > 0:
            pairs += 1
            dirty_count[color] -= 1
            K -= 1

    # 3. 남은 세탁 가능 수로 dirty만으로 쌍 만들기
    for color in dirty_count:
        while dirty_count[color] >= 2 and K >= 2:
            pairs += 1
            dirty_count[color] -= 2
            K -= 2

    return pairs
