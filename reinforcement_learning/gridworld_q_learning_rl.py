import numpy as np
import random
from game.grid_world_template import *  # 그리드월드를 별도 패키지로 참조

# Q-Learning 파라미터
alpha = 0.1  # 학습률
gamma = 0.99  # 할인율 (=감쇄율)
epsilon = 0.1  # 탐험-탐사 균형을 위한 epsilon 값
num_episodes = 50  # 에피소드 수

# GridWorld 환경 설정
grid_size = (5, 5)
start = (0, 0)
goal = (4, 4)
obstacles = [(1, 1), (2, 2), (3, 3)]

env = GridWorld(grid_size, start, goal, obstacles)

# Q-테이블 초기화 (상태 공간: 그리드 셀, 행동 공간: 4방향)
Q_table = np.zeros((*grid_size, 4))

def choose_action(state):
    """Epsilon-greedy 방식으로 행동 선택"""
    result_value = None
    if random.uniform(0, 1) < epsilon:
        result_value = AgentHelper.get_rand_agent_action_type()  # 랜덤 행동 (탐험)
    else:
        result_value = np.argmax(Q_table[state])  # Q값이 가장 높은 행동 선택 (탐사)
    
    return result_value

def q_learning():
    """Q-learning 알고리즘"""
    for episode in range(num_episodes):
        state = env.reset()
        is_done = False

        while not is_done:
            action = choose_action(state)
            next_state, reward, is_done = env.step(AgentActionType(action))
            
            # Q-값 업데이트: Q(s, a) <- Q(s, a) + α * [R + γ * max(Q(s', a')) - Q(s, a)]
            Q_table[state][action] = Q_table[state][action] + alpha * (
                reward + gamma * np.max(Q_table[next_state]) - Q_table[state][action]
            )

            state = next_state  # 상태 전환
            
            # 그리드를 렌더링하여 학습 과정을 확인
            env.render()
            print(f"Episode {episode+1}, Action: {action}, Reward: {reward}")

        # 에피소드 완료 후 보상 출력
        print(f"Episode {episode+1} finished with reward: {reward}\n")
        
    print("Q-Learning 완료!")
    
# Q-Learning 알고리즘 실행
q_learning()

# 학습된 Q-테이블 출력
print("최종 Q-테이블:")
print(Q_table)
