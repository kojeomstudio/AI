#!/usr/bin/env python3
"""
입력 시스템 테스트 스크립트
다양한 입력 메서드의 동작을 테스트합니다.
"""

import sys
import os
import time
import json

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from input_manager import InputManager
from logger_helper import get_logger

logger = get_logger()

def test_basic_input():
    """기본 입력 테스트"""
    print("=== 기본 입력 테스트 ===")
    
    # 입력 매니저 초기화
    input_manager = InputManager("Mabinogi Mobile")
    
    if not input_manager.target_hwnd:
        print("❌ 타겟 윈도우를 찾을 수 없습니다.")
        return False
    
    print(f"✅ 타겟 윈도우 찾음: HWND {input_manager.target_hwnd}")
    
    # 윈도우 정보 출력
    window_info = input_manager._get_window_info()
    if window_info:
        print(f"윈도우 크기: {window_info['width']}x{window_info['height']}")
        print(f"클라이언트 크기: {window_info['client_width']}x{window_info['client_height']}")
    
    return True

def test_click_methods():
    """클릭 메서드 테스트"""
    print("\n=== 클릭 메서드 테스트 ===")
    
    input_manager = InputManager("Mabinogi Mobile")
    
    if not input_manager.target_hwnd:
        print("❌ 타겟 윈도우를 찾을 수 없습니다.")
        return
    
    # 화면 중앙 좌표 계산
    window_info = input_manager._get_window_info()
    if not window_info:
        print("❌ 윈도우 정보를 가져올 수 없습니다.")
        return
    
    center_x = window_info['left'] + window_info['width'] // 2
    center_y = window_info['top'] + window_info['height'] // 2
    
    print(f"테스트 좌표: ({center_x}, {center_y})")
    print("5초 후 클릭 테스트를 시작합니다...")
    time.sleep(5)
    
    # 각 메서드별 테스트
    methods = [
        ('pyautogui', input_manager.method1_pyautogui_with_window_focus),
        ('win32api', input_manager.method2_win32api_sendinput),
        ('postmessage', input_manager.method3_postmessage),
        ('sendmessage', input_manager.method4_sendmessage)
    ]
    
    for name, method in methods:
        print(f"\n--- {name} 메서드 테스트 ---")
        try:
            result = method(center_x, center_y, 'left')
            print(f"{'✅ 성공' if result else '❌ 실패'}: {name}")
            time.sleep(2)  # 테스트 간격
        except Exception as e:
            print(f"❌ 오류: {name} - {e}")

def test_key_methods():
    """키보드 입력 메서드 테스트"""
    print("\n=== 키보드 입력 메서드 테스트 ===")
    
    input_manager = InputManager("Mabinogi Mobile")
    
    if not input_manager.target_hwnd:
        print("❌ 타겟 윈도우를 찾을 수 없습니다.")
        return
    
    print("5초 후 키 입력 테스트를 시작합니다...")
    time.sleep(5)
    
    # 테스트할 키들
    test_keys = ['space', 'enter', 'tab']
    
    for key in test_keys:
        print(f"\n--- 키 '{key}' 테스트 ---")
        
        # pyautogui 방법
        try:
            result = input_manager.send_key(key, 'pyautogui')
            print(f"{'✅ 성공' if result else '❌ 실패'}: pyautogui - {key}")
        except Exception as e:
            print(f"❌ 오류: pyautogui - {key} - {e}")
        
        time.sleep(1)
        
        # postmessage 방법
        try:
            result = input_manager.send_key(key, 'postmessage')
            print(f"{'✅ 성공' if result else '❌ 실패'}: postmessage - {key}")
        except Exception as e:
            print(f"❌ 오류: postmessage - {key} - {e}")
        
        time.sleep(1)

def test_action_config():
    """액션 설정 테스트"""
    print("\n=== 액션 설정 테스트 ===")
    
    config_path = "./config/action_config.json"
    
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print("✅ 액션 설정 파일 로드 성공")
        print(f"기본 입력 메서드: {config.get('input_method', 'N/A')}")
        print(f"기본 지연 시간: {config.get('default_delay', 'N/A')}초")
        
        actions = config.get('actions', {})
        print(f"정의된 액션 수: {len(actions)}")
        
        for action_name, action_config in actions.items():
            action_type = action_config.get('type', 'unknown')
            description = action_config.get('description', 'N/A')
            print(f"  - {action_name}: {action_type} ({description})")
        
        priority_rules = config.get('priority_rules', [])
        print(f"우선순위 규칙 수: {len(priority_rules)}")
        
        for rule in priority_rules:
            name = rule.get('name', 'N/A')
            conditions = rule.get('conditions', [])
            action = rule.get('action', 'N/A')
            print(f"  - {name}: {conditions} -> {action}")
            
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")

def main():
    """메인 테스트 함수"""
    print("마비노기 모바일 매크로 - 입력 시스템 테스트")
    print("=" * 50)
    
    # 기본 입력 테스트
    if not test_basic_input():
        print("기본 테스트 실패, 프로그램을 종료합니다.")
        return
    
    # 액션 설정 테스트
    test_action_config()
    
    # 사용자 선택
    print("\n" + "=" * 50)
    print("테스트할 항목을 선택하세요:")
    print("1. 클릭 메서드 테스트")
    print("2. 키보드 입력 메서드 테스트")
    print("3. 모든 테스트 실행")
    print("4. 종료")
    
    while True:
        try:
            choice = input("\n선택 (1-4): ").strip()
            
            if choice == '1':
                test_click_methods()
                break
            elif choice == '2':
                test_key_methods()
                break
            elif choice == '3':
                test_click_methods()
                test_key_methods()
                break
            elif choice == '4':
                print("테스트를 종료합니다.")
                break
            else:
                print("잘못된 선택입니다. 1-4 중에서 선택하세요.")
                
        except KeyboardInterrupt:
            print("\n테스트를 중단합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            break
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    main() 