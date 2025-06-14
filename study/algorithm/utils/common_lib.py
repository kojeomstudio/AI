def is_valid_bracket(string):
    stack = []
    bracket_pair = {')': '(', ']': '[', '}': '{'}
    for character in string:
        if character in '([{':
            stack.append(character)
        else:
            if not stack or stack.pop() != bracket_pair[character]:
                return False
    return not stack

# back-tracking
def get_combinations(elements, target_count):
    result_combinations = []

    def backtrack(start_index, current_combination):
        if len(current_combination) == target_count:
            result_combinations.append(current_combination[:])
            return
        for index in range(start_index, len(elements)):
            current_combination.append(elements[index])
            backtrack(index + 1, current_combination)
            current_combination.pop()

    backtrack(0, [])
    return result_combinations

from collections import deque

# bfs
def bfs_area_search(grid, start_position):
    row_count, column_count = len(grid), len(grid[0])
    visited_matrix = [[False] * column_count for _ in range(row_count)]
    search_queue = deque([start_position])
    visited_matrix[start_position[0]][start_position[1]] = True

    while search_queue:
        current_row, current_column = search_queue.popleft()
        for row_offset, column_offset in [(-1,0), (1,0), (0,-1), (0,1)]:
            next_row = current_row + row_offset
            next_column = current_column + column_offset
            if (0 <= next_row < row_count and 0 <= next_column < column_count and 
                not visited_matrix[next_row][next_column] and grid[next_row][next_column] == 1):
                visited_matrix[next_row][next_column] = True
                search_queue.append((next_row, next_column))


# 다익스트라 (우선순위 큐 사용)
import heapq
def dijkstra_shortest_path(start_node, graph_structure, node_count):
    shortest_distances = [float('inf')] * (node_count + 1)
    shortest_distances[start_node] = 0
    priority_queue = [(0, start_node)]

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)
        if shortest_distances[current_node] < current_cost:
            continue

        for adjacent_node, edge_cost in graph_structure[current_node]:
            new_cost = current_cost + edge_cost
            if shortest_distances[adjacent_node] > new_cost:
                shortest_distances[adjacent_node] = new_cost
                heapq.heappush(priority_queue, (new_cost, adjacent_node))

    return shortest_distances

# 최대 부분합.
def get_maximum_subarray_sum(number_list):
    max_subarray = [0] * len(number_list)
    max_subarray[0] = number_list[0]

    for index in range(1, len(number_list)):
        max_subarray[index] = max(number_list[index], max_subarray[index - 1] + number_list[index])

    return max(max_subarray)

# 애너그램.
from collections import Counter
def is_anagram_string(string_one, string_two):
    character_counter_one = Counter(string_one)
    character_counter_two = Counter(string_two)
    return character_counter_one == character_counter_two

# dfs 기반 모든 경로 탐색.
def find_all_paths_dfs(graph_structure, start_node, end_node):
    all_paths = []
    visited_path = []

    def dfs(current_node):
        visited_path.append(current_node)
        if current_node == end_node:
            all_paths.append(visited_path[:])
        else:
            for adjacent_node in graph_structure.get(current_node, []):
                if adjacent_node not in visited_path:
                    dfs(adjacent_node)
        visited_path.pop()

    dfs(start_node)
    return all_paths

# bfs 기반 단일 경로 추적. (간선 비용 동일)
from collections import deque
def find_shortest_path_bfs(graph_structure, start_node, end_node):
    visited_set = set()
    parent_tracker = {}
    traversal_queue = deque([start_node])
    visited_set.add(start_node)

    while traversal_queue:
        current_node = traversal_queue.popleft()
        if current_node == end_node:
            break
        for adjacent_node in graph_structure.get(current_node, []):
            if adjacent_node not in visited_set:
                visited_set.add(adjacent_node)
                parent_tracker[adjacent_node] = current_node
                traversal_queue.append(adjacent_node)

    # 역추적
    if end_node not in parent_tracker and start_node != end_node:
        return []  # 경로 없음
    path = [end_node]
    while path[-1] != start_node:
        path.append(parent_tracker[path[-1]])
    path.reverse()
    return path

######순열, 조합등 itertools 기반 사용.
from itertools import combinations, permutations, product, combinations_with_replacement

###### 조합
items = ['A', 'B', 'C', 'D']
result = list(combinations(items, 2))

print(result)
# [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'C'), ('B', 'D'), ('C', 'D')]

###### 순열
items = ['A', 'B', 'C']
result = list(permutations(items, 2))

print(result)
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]


###### 중복-조합
items = [1, 2, 3]
result = list(combinations_with_replacement(items, 2))

print(result)
# [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

###### 중복-순열

items = ['A', 'B']
result = list(product(items, repeat=2))

print(result)
# [('A', 'A'), ('A', 'B'), ('B', 'A'), ('B', 'B')]

# 유니온-파인드 구조.
class DisjointSetUnion:
    def __init__(self, total_count):
        self.parent_node = [i for i in range(total_count + 1)]
        self.set_size = [1] * (total_count + 1)

    def find_representative(self, node_index):
        if self.parent_node[node_index] != node_index:
            self.parent_node[node_index] = self.find_representative(self.parent_node[node_index])
        return self.parent_node[node_index]

    def union_sets(self, node_a, node_b):
        root_a = self.find_representative(node_a)
        root_b = self.find_representative(node_b)

        if root_a == root_b:
            return False  # 이미 같은 집합

        if self.set_size[root_a] < self.set_size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent_node[root_b] = root_a
        self.set_size[root_a] += self.set_size[root_b]
        return True
        
# 유니온파인드 구조 사용 예시...
dsu = DisjointSetUnion(10)
dsu.union_sets(1, 2)
dsu.union_sets(2, 3)
print(dsu.find_representative(3))  # 1 또는 2 또는 3 중 루트

    
# 세그먼트 트리.
class SegmentTree:
    def __init__(self, original_array):
        self.original_array = original_array
        self.total_count = len(original_array)
        self.segment_tree = [0] * (self.total_count * 4)
        self._initialize(1, 0, self.total_count - 1)

    def _initialize(self, current_index, start_index, end_index):
        if start_index == end_index:
            self.segment_tree[current_index] = self.original_array[start_index]
            return self.segment_tree[current_index]

        mid_index = (start_index + end_index) // 2
        left_sum = self._initialize(current_index * 2, start_index, mid_index)
        right_sum = self._initialize(current_index * 2 + 1, mid_index + 1, end_index)
        self.segment_tree[current_index] = left_sum + right_sum
        return self.segment_tree[current_index]

    def query_range_sum(self, query_start, query_end):
        return self._query_sum(1, 0, self.total_count - 1, query_start, query_end)

    def _query_sum(self, current_index, start_index, end_index, query_start, query_end):
        if query_end < start_index or end_index < query_start:
            return 0
        if query_start <= start_index and end_index <= query_end:
            return self.segment_tree[current_index]

        mid_index = (start_index + end_index) // 2
        left_result = self._query_sum(current_index * 2, start_index, mid_index, query_start, query_end)
        right_result = self._query_sum(current_index * 2 + 1, mid_index + 1, end_index, query_start, query_end)
        return left_result + right_result

    def update_single_value(self, target_index, new_value):
        difference = new_value - self.original_array[target_index]
        self.original_array[target_index] = new_value
        self._update_value(1, 0, self.total_count - 1, target_index, difference)

    def _update_value(self, current_index, start_index, end_index, target_index, difference):
        if target_index < start_index or target_index > end_index:
            return
        self.segment_tree[current_index] += difference
        if start_index != end_index:
            mid_index = (start_index + end_index) // 2
            self._update_value(current_index * 2, start_index, mid_index, target_index, difference)
            self._update_value(current_index * 2 + 1, mid_index + 1, end_index, target_index, difference)

# 세그먼트 트리 사용 예시...
array = [5, 8, 6, 3, 2, 7, 4, 6]
segment_tree = SegmentTree(array)

print(segment_tree.query_range_sum(2, 5))  # 6 + 3 + 2 + 7 = 18

segment_tree.update_single_value(3, 10)  # array[3] = 3 → 10으로 변경
print(segment_tree.query_range_sum(2, 5))  # 6 + 10 + 2 + 7 = 25
