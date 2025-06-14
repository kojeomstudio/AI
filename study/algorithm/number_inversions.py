
'''
An array A consisting of N integers is given. An inversion is a pair of indexes (P, Q) such that P < Q and A[Q] < A[P].

Write a function:

def solution(A)

that computes the number of inversions in A, or returns −1 if it exceeds 1,000,000,000.

For example, in the following array:

 A[0] = -1 A[1] = 6 A[2] = 3
 A[3] =  4 A[4] = 7 A[5] = 4
there are four inversions:

   (1,2)  (1,3)  (1,5)  (4,5)
so the function should return 4.

Write an efficient algorithm for the following assumptions:

N is an integer within the range [0..100,000];
each element of array A is an integer within the range [−2,147,483,648..2,147,483,647].
'''

def solution(input_array):
    def merge_sort_and_count(array):
        # 기저 조건: 배열 길이가 1 이하이면 그대로 반환
        if len(array) <= 1:
            return array, 0
        
        # 배열을 반으로 나누기
        middle_index = len(array) // 2
        left_array, left_inversion_count = merge_sort_and_count(array[:middle_index])
        right_array, right_inversion_count = merge_sort_and_count(array[middle_index:])
        
        # 병합 및 inversion count
        merged_array = []
        inversion_count = 0
        left_index = 0
        right_index = 0
        
        # 병합 과정에서 역전 쌍을 찾음
        while left_index < len(left_array) and right_index < len(right_array):
            if left_array[left_index] <= right_array[right_index]:
                merged_array.append(left_array[left_index])
                left_index += 1
            else:
                merged_array.append(right_array[right_index])
                right_index += 1

                # 핵심: 왼쪽 배열에 남아 있는 모든 원소는 이 오른쪽 원소보다 큼
                inversion_count += len(left_array) - left_index

        # 남은 원소들 추가
        merged_array += left_array[left_index:]
        merged_array += right_array[right_index:]

        total_count = left_inversion_count + right_inversion_count + inversion_count
        return merged_array, total_count

    # 정렬은 무시하고 inversion 수만 사용
    _, total_inversion_count = merge_sort_and_count(input_array)
    return total_inversion_count

        
        