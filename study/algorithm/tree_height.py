# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

from extratypes import Tree  # library with types used in the task

Height = 0
def traverse(T, depth):
    global Height

    if T.l != None:
        traverse(T.l, depth + 1)
    elif T.r != None:
        traverse(T.r, depth + 1)
    else:
        if depth >= Height:
            Height = depth
        return depth

def solution(T):
    traverse(T, 0)
    
    return Height
