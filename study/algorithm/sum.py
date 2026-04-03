'''

For a given array A of N integers and a sequence S of N integers from the set {−1, 1}, we define val(A, S) as follows:

val(A, S) = |sum{ A[i]*S[i] for i = 0..N−1 }|

(Assume that the sum of zero elements equals zero.)

For a given array A, we are looking for such a sequence S that minimizes val(A,S).

Write a function:

def solution(A)

that, given an array A of N integers, computes the minimum value of val(A,S) from all possible values of val(A,S) for all possible sequences S of N integers from the set {−1, 1}.

For example, given array:

  A[0] =  1
  A[1] =  5
  A[2] =  2
  A[3] = -2
your function should return 0, since for S = [−1, 1, −1, 1], val(A, S) = 0, which is the minimum possible value.

Write an efficient algorithm for the following assumptions:

N is an integer within the range [0..20,000];
each element of array A is an integer within the range [−100..100].


'''

def solution(A):
    if not A:
        return 0

    total_sum = sum(abs(a) for a in A)

    reachable = {0}
    for a in A:
        new_reachable = set()
        for val in reachable:
            new_reachable.add(val + a)
            new_reachable.add(val - a)
        reachable = new_reachable

    min_abs = total_sum
    for val in reachable:
        abs_val = abs(val)
        if abs_val < min_abs:
            min_abs = abs_val

    return min_abs


if __name__ == "__main__":
    print(solution([1, 5, 2, -2]))
    print(solution([3, 1, 2, 4, 3]))
    print(solution([1, 2, 3]))
