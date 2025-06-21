# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

'''

🧩 Problem Description (English)
Title: Check if All Points Lie on the Border of a Square

Task:
You are given two integer arrays X and Y, both of length N (1 ≤ N ≤ 20), representing the coordinates of N distinct points on a 2D plane. Your task is to determine whether all given points lie exactly on the border of some axis-aligned square (i.e., a square whose sides are parallel to the X and Y axes).

You may assume that:

Points may lie on any of the four sides of the square (top, bottom, left, or right), including corners.

The square's position and size are not given — you must check whether such a square can be constructed based on the point set.

🧮 Input
X: a list of integers, where X[i] is the x-coordinate of the i-th point.

Y: a list of integers, where Y[i] is the y-coordinate of the i-th point.

Constraints:

1 ≤ N ≤ 20

-10 ≤ X[i], Y[i] ≤ 10

All points are distinct

🎯 Output
Return True if all points lie on the border of some axis-aligned square.

Return False otherwise.

'''

def solution(X, Y):
    
    # givened positions are matched on square..
    # how to..?
    # point on plane.
    # input is X -> x array(points)
    # input is Y -> y arrary(points)
    # is make a sqaure form input x,y...?
   
    point_pairs = list(zip(X, Y))
    points_length = len(X)

    for index_outer in range(points_length):
        for index_inner in range(index_outer + 1, points_length):
            x_coord_1, y_coord_1 = point_pairs[index_outer]
            x_coord_2, y_coord_2 = point_pairs[index_inner]

            # make sqaure...
            min_x = min(x_coord_1, x_coord_2)
            min_y = min(y_coord_1, y_coord_2)

            side_of_sqaure = max(abs(x_coord_1 - x_coord_2), abs(y_coord_1 - y_coord_2))
            if side_of_sqaure <= 0:
                continue
            
            left = min_x
            right = left + side_of_sqaure

            bottom = min_y
            top = bottom + side_of_sqaure

            # check all point.
            is_all = True
            for point in point_pairs:
                target_x = point[0]
                target_y = point[1]

                is_all &=(((target_x == left or target_x == right) and (bottom <= target_y <= top)) or ((target_y == bottom or target_y == top) and (left <= target_x <= right)))

    return False
        