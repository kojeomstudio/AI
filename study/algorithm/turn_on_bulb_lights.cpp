
/*

💡 Problem: Moments When All Bulbs Are On and Glowing
You are given a sequence of bulbs arranged in a straight line, numbered from 1 to N. All bulbs are initially off.

You are also given an array A of length N, where A[i] indicates that the bulb with number A[i] is turned on at the i-th moment (0-indexed).

However, a bulb will only start glowing if it is turned on and all bulbs with smaller numbers are already on.
That is, bulb k glows only when bulbs 1 through k - 1 are all on and glowing.

🎯 Goal
Determine how many moments exist in the sequence where all bulbs that have been turned on so far are glowing.

🛠 Input
A list A of N distinct integers from 1 to N (a permutation of 1..N).

📤 Output
Return an integer representing the number of moments when all turned-on bulbs are glowing (i.e., when the turned-on bulbs form a contiguous prefix from 1 to k with no missing bulbs).

💡 Constraints
1 ≤ N ≤ 10⁵

A is a permutation of the integers from 1 to N.

*/

int solution(vector<int> &A) {
    // N^2?
    // find lights time...(sum)

    int lights_count = 0;
    int total_turn_on = 0;
    int max_switch_index = 0;

    for(size_t idx = 0; idx < A.size(); ++idx)
    {
        max_switch_index = max(max_switch_index, A[idx]);
        total_turn_on++;

        if(total_turn_on == max_switch_index)
        {
            lights_count++;
        }
    }

    return lights_count;
}
