"""
Simple test script to verify agent CLI commands work via subprocess.
Tests: claude, gemini, codex
"""
import subprocess
import shutil
import sys
import os


def test_agent_path(agent_name: str, command: str) -> tuple[bool, str]:
    """Check if agent command exists in PATH"""
    path = shutil.which(command)
    if path:
        print(f"[OK] {agent_name}: Found at {path}")
        return True, path
    else:
        print(f"[FAIL] {agent_name}: '{command}' not found in PATH")
        return False, ""


def test_agent_version(agent_name: str, command: str, version_args: list, cmd_path: str) -> bool:
    """Try to get agent version"""
    try:
        # For .CMD/.BAT files on Windows, use shell=True
        is_cmd_file = cmd_path.lower().endswith(('.cmd', '.bat'))

        if is_cmd_file:
            # Use the full command string for shell execution
            cmd_str = f'{command} {" ".join(version_args)}'
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=10,
                shell=True
            )
        else:
            result = subprocess.run(
                [command] + version_args,
                capture_output=True,
                text=True,
                timeout=10
            )

        output = result.stdout.strip() or result.stderr.strip()
        if output:
            # Truncate long output
            display = output[:100].replace('\n', ' ')
            print(f"[OK] {agent_name} version: {display}")
            return True
        else:
            print(f"[WARN] {agent_name}: No version output (return code: {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] {agent_name}: Version check timed out")
        return False
    except Exception as e:
        print(f"[FAIL] {agent_name}: {e}")
        return False


def test_agent_simple_prompt(agent_name: str, command: str, prompt_args: list, cmd_path: str) -> bool:
    """Try a simple prompt execution"""
    try:
        print(f"[INFO] {agent_name}: Testing simple prompt...")

        is_cmd_file = cmd_path.lower().endswith(('.cmd', '.bat'))

        if is_cmd_file:
            cmd_str = f'{command} {" ".join(prompt_args)}'
            result = subprocess.run(
                cmd_str,
                capture_output=True,
                text=True,
                timeout=60,
                shell=True
            )
        else:
            result = subprocess.run(
                [command] + prompt_args,
                capture_output=True,
                text=True,
                timeout=60
            )

        if result.returncode == 0:
            output = result.stdout.strip()[:200] if result.stdout else "(no output)"
            print(f"[OK] {agent_name}: Prompt executed successfully")
            print(f"     Output preview: {output}")
            return True
        else:
            error = result.stderr.strip()[:200] if result.stderr else "(no error message)"
            stdout = result.stdout.strip()[:200] if result.stdout else ""
            print(f"[FAIL] {agent_name}: Return code {result.returncode}")
            if error:
                print(f"     Stderr: {error}")
            if stdout:
                print(f"     Stdout: {stdout}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] {agent_name}: Prompt execution timed out (60s)")
        return False
    except Exception as e:
        print(f"[FAIL] {agent_name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Agent CLI Test Script")
    print("=" * 60)
    print()

    # Agent configurations
    agents = {
        "claude-code": {
            "command": "claude",
            "version_args": ["--version"],
            "prompt_args": ["-p", '"Say hello in one word"']
        },
        "gemini": {
            "command": "gemini",
            "version_args": ["--version"],
            "prompt_args": ["-p", '"Say hello in one word"']
        },
        "codex": {
            "command": "codex",
            "version_args": ["--version"],
            "prompt_args": ["exec", '"Say hello in one word"']
        },
    }

    results = {}

    # Test 1: PATH verification
    print("[Step 1] PATH Verification")
    print("-" * 60)
    for name, config in agents.items():
        found, path = test_agent_path(name, config["command"])
        results[name] = {"path": found, "cmd_path": path}
    print()

    # Test 2: Version check
    print("[Step 2] Version Check")
    print("-" * 60)
    for name, config in agents.items():
        if results[name]["path"]:
            results[name]["version"] = test_agent_version(
                name, config["command"], config["version_args"], results[name]["cmd_path"]
            )
        else:
            print(f"[SKIP] {name}: Skipped (not in PATH)")
            results[name]["version"] = False
    print()

    # Test 3: Simple prompt (optional, may take time)
    print("[Step 3] Simple Prompt Test (60s timeout each)")
    print("-" * 60)

    try:
        run_prompt_test = input("Run prompt tests? (y/N): ").strip().lower() == 'y'
    except EOFError:
        run_prompt_test = False

    if run_prompt_test:
        for name, config in agents.items():
            if results[name]["path"]:
                results[name]["prompt"] = test_agent_simple_prompt(
                    name, config["command"], config["prompt_args"], results[name]["cmd_path"]
                )
            else:
                print(f"[SKIP] {name}: Skipped (not in PATH)")
                results[name]["prompt"] = False
    else:
        print("[INFO] Prompt tests skipped")
        for name in agents:
            results[name]["prompt"] = None
    print()

    # Summary (ASCII-safe for Windows console)
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, result in results.items():
        path_status = "O" if result["path"] else "X"
        version_status = "O" if result["version"] else "X"
        prompt_status = "O" if result.get("prompt") else ("-" if result.get("prompt") is None else "X")
        print(f"  {name:12} | PATH: {path_status} | Version: {version_status} | Prompt: {prompt_status}")
    print()
    print("Legend: O=Pass, X=Fail, -=Skipped")
    print()


if __name__ == "__main__":
    main()
