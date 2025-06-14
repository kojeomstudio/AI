# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")

from collections import deque

def solution(S):
    
    string_stack = deque()
    
    for char in S:
        if len(string_stack):
            top = string_stack[-1]
            if top != None and top == char:
                string_stack.pop()
            else:
                string_stack.append(char)
        else:
            string_stack.append(char)
    
    return ''.join(string_stack)