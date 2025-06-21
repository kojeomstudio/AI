'''

🧩 Problem: Robot Walking Along a Rectangle Border
You are given a string moves consisting of characters that describe the movement of a robot on a 2D grid. The robot starts at coordinate (0, 0) and follows the instructions in moves one by one.

Each character in moves represents a movement in one of the four cardinal directions:

'^': move up (i.e., y += 1)

'v': move down (i.e., y -= 1)

'>': move right (i.e., x += 1)

'<': move left (i.e., x -= 1)

✅ Constraints
1 ≤ len(moves) ≤ 100

The robot never visits the same point twice, except for the starting point (0, 0), which may be visited again at the end.

The path must form a closed loop — i.e., the robot must return to the starting point after executing all moves.

'''

def solution(moves):
    # Implement your solution here

    # if rectangle paths and return true otherwise return false.
    # closed rect area conditions...
    # 'up' 'right'
    #'right'  'down'
    # 'down' 'left'
    # 'left' 'up'
    # check start to end position(same)
    # and check parallel line

    #pos
    x, y = 0, 0
    move_paths = [(x,y)] #pair(tuple)

    for move_char in moves:
        if move_char == '^':
            #up
            y += 1
        elif move_char == '<':
            #left
            x -= 1
        elif move_char == '>':
            #right
            x += 1
        elif move_char == 'v':
            #down
            y -= 1
        
        move_paths.append((x, y))
    
    if (x, y) != (0, 0):
        return False

    border_x_values = set(element[0] for element in move_paths)
    border_y_values = set(element[1] for element in move_paths)

    if (len(border_x_values) != 2) or (len(border_y_values) != 2):
        return False

    path_x_min = min(border_x_values)
    path_x_max = max(border_x_values)
    path_y_min = min(border_y_values)
    path_y_max = max(border_y_values)

    '''
    for path_x, path_y in move_paths:
        if (path_x < path_x_min) or (path_x > path_x_max):
            return False
        elif (path_y < path_y_min) or (path_y > path_y_max):
            return False
    '''

    #check in border on line.
    def is_on_border_line(in_path_x, in_path_y):
        return((path_x_min <= in_path_x <= path_x_max) and (in_path_y == path_y_min or in_path_y == path_y_max) or 
               (path_y_min <= in_path_y <= path_y_max) and (in_path_x == path_x_min or in_path_x == path_x_max))

    for path_x, path_y in move_paths:
        if not is_on_border_line(path_x, path_y):
            return False

    return True

    