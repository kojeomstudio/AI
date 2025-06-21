
'''
🧩 Problem: Token Compression on Stacks
You are given a list of stacks indexed from 0 to N - 1, where each stack may contain a number of identical tokens. You are allowed to perform a special exchange operation any number of times:

Exchange Operation:
You can remove two tokens from stack i and add one token to stack i + 1.

This operation can be performed as many times as possible, on any stack i, as long as there are at least two tokens on that stack.

🎯 Goal
Your task is to determine the minimum number of tokens that may remain across all stacks after any number of exchange operations.

🛠 Input
A list A of N non-negative integers.
Each A[i] represents the number of tokens initially on stack i.

📤 Output
Return an integer representing the minimum total number of tokens remaining on the stacks after all possible exchange operations.

'''

def solution(A):
    
    # token
    # element is stack -> stakc count ( [0] = 2, [1] = 3)
    # -> like a compression.
    # added new stack..
    # 2->1 (2:1??)

    stack_tokens = list(A)

    idx = 0
    origin_stack_len = len(A)

    while idx < len(stack_tokens):
        cur_token_stack = stack_tokens[idx]

        tokenized = cur_token_stack % 2
        stack_tokens[idx] = tokenized

        pass_tokens = cur_token_stack // 2

        if pass_tokens > 0:
            next_idx = idx + 1
            if next_idx < origin_stack_len:
                stack_tokens[next_idx] += pass_tokens
            else:
                stack_tokens.append(pass_tokens)
        
        idx += 1

    return sum(stack_tokens)

    
