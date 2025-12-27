"""
Agent executor module for running coding agents as subprocess

This module provides functionality to execute various coding agents (Claude Code, Codex, Gemini, etc.)
as subprocesses with comprehensive logging and error handling.
"""
import subprocess
import time
import shlex
import shutil
import sys
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging

# Handle imports for both module mode and PyInstaller frozen mode
try:
    from .config import settings, format_prompt_template
except ImportError:
    from config import settings, format_prompt_template

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes coding agents as subprocess and captures output.

    Supports multiple agent types with configurable commands and detailed logging
    for tracking execution flow and debugging issues.
    """

    def __init__(self):
        """Initialize the agent executor with configuration from settings"""
        self._agent_commands = settings.agent_commands.copy()
        logger.info("[EXECUTOR] AgentExecutor initialized")
        logger.info(f"[EXECUTOR] Supported agents: {list(self._agent_commands.keys())}")

    def execute(
        self,
        agent_type: str,
        command_args: str,
        timeout: int = 300,
        working_directory: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a coding agent with the given arguments.

        Args:
            agent_type: Type of agent to execute (e.g., 'claude-code')
            command_args: Command line arguments as a single string
            timeout: Maximum execution time in seconds
            working_directory: Working directory for execution
            request_id: Optional request ID for tracking (auto-generated if not provided)

        Returns:
            Dictionary containing execution results with detailed information
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        logger.info(f"[EXECUTOR:{request_id}] ========== Execution Started ==========")
        logger.info(f"[EXECUTOR:{request_id}] Timestamp: {timestamp}")
        logger.info(f"[EXECUTOR:{request_id}] Agent type: {agent_type}")
        logger.info(f"[EXECUTOR:{request_id}] Timeout: {timeout}s")
        logger.info(f"[EXECUTOR:{request_id}] Working directory: {working_directory or 'default'}")
        logger.debug(f"[EXECUTOR:{request_id}] Command args: {command_args}")

        # Get the base command for the agent type
        try:
            base_command = self._get_agent_command(agent_type)
            logger.info(f"[EXECUTOR:{request_id}] Base command: {base_command}")
        except ValueError as e:
            logger.error(f"[EXECUTOR:{request_id}] Invalid agent type: {agent_type}")
            return self._create_error_result(
                request_id=request_id,
                error_code="INVALID_AGENT_TYPE",
                error_message=str(e),
                start_time=start_time
            )

        # Parse command_args safely
        try:
            args_list = shlex.split(command_args) if command_args else []
            logger.debug(f"[EXECUTOR:{request_id}] Parsed args: {args_list}")
        except ValueError as e:
            logger.error(f"[EXECUTOR:{request_id}] Failed to parse command arguments: {e}")
            return self._create_error_result(
                request_id=request_id,
                error_code="INVALID_COMMAND_ARGS",
                error_message=f"Invalid command arguments: {str(e)}",
                start_time=start_time
            )

        # Resolve command for platform (handles Windows .CMD/.BAT files)
        full_command, use_shell = self._resolve_command_for_platform(base_command, args_list)
        if use_shell:
            logger.info(f"[EXECUTOR:{request_id}] Full command (shell): {full_command[:200]}...")
        else:
            logger.info(f"[EXECUTOR:{request_id}] Full command: {' '.join(full_command)}")

        # Validate working directory
        cwd = Path(working_directory) if working_directory else None
        if cwd:
            if not cwd.exists():
                logger.error(f"[EXECUTOR:{request_id}] Working directory does not exist: {cwd}")
                return self._create_error_result(
                    request_id=request_id,
                    error_code="INVALID_WORKING_DIRECTORY",
                    error_message=f"Working directory does not exist: {cwd}",
                    start_time=start_time
                )
            logger.info(f"[EXECUTOR:{request_id}] Validated working directory: {cwd}")

        # Execute the command
        logger.info(f"[EXECUTOR:{request_id}] Starting subprocess execution...")

        try:
            result = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                shell=use_shell,  # True for Windows .CMD/.BAT, False otherwise
                encoding="utf-8",
                errors="replace"  # Handle encoding errors gracefully on Windows
            )

            execution_time = time.time() - start_time
            success = result.returncode == 0

            # Handle None stdout/stderr safely
            stdout = result.stdout or ""
            stderr = result.stderr or ""

            logger.info(f"[EXECUTOR:{request_id}] ========== Execution Completed ==========")
            logger.info(f"[EXECUTOR:{request_id}] Exit code: {result.returncode}")
            logger.info(f"[EXECUTOR:{request_id}] Success: {success}")
            logger.info(f"[EXECUTOR:{request_id}] Execution time: {execution_time:.2f}s")
            logger.info(f"[EXECUTOR:{request_id}] Stdout length: {len(stdout)} chars")
            logger.info(f"[EXECUTOR:{request_id}] Stderr length: {len(stderr)} chars")

            if stdout:
                logger.debug(f"[EXECUTOR:{request_id}] Stdout preview: {stdout[:500]}...")
            if stderr:
                logger.warning(f"[EXECUTOR:{request_id}] Stderr: {stderr}")

            return {
                "success": success,
                "output": stdout,
                "error": stderr if stderr else None,
                "exit_code": result.returncode,
                "execution_time": execution_time,
                "request_id": request_id,
                "timestamp": timestamp
            }

        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            logger.error(f"[EXECUTOR:{request_id}] ========== Execution Timeout ==========")
            logger.error(f"[EXECUTOR:{request_id}] Command timed out after {timeout}s")
            logger.error(f"[EXECUTOR:{request_id}] Partial stdout: {e.stdout[:500] if e.stdout else 'None'}...")
            logger.error(f"[EXECUTOR:{request_id}] Partial stderr: {e.stderr[:500] if e.stderr else 'None'}...")

            return self._create_error_result(
                request_id=request_id,
                error_code="EXECUTION_TIMEOUT",
                error_message=f"Execution timed out after {timeout} seconds",
                start_time=start_time,
                partial_output=e.stdout if e.stdout else ""
            )

        except FileNotFoundError:
            execution_time = time.time() - start_time
            logger.error(f"[EXECUTOR:{request_id}] ========== Command Not Found ==========")
            logger.error(f"[EXECUTOR:{request_id}] Agent executable not found: {base_command}")
            logger.error(f"[EXECUTOR:{request_id}] Ensure {agent_type} is installed and in PATH")

            return self._create_error_result(
                request_id=request_id,
                error_code="COMMAND_NOT_FOUND",
                error_message=f"Agent executable not found: {base_command}. Make sure {agent_type} is installed and in PATH.",
                start_time=start_time
            )

        except PermissionError as e:
            execution_time = time.time() - start_time
            logger.error(f"[EXECUTOR:{request_id}] ========== Permission Error ==========")
            logger.error(f"[EXECUTOR:{request_id}] Permission denied: {e}")

            return self._create_error_result(
                request_id=request_id,
                error_code="PERMISSION_DENIED",
                error_message=f"Permission denied when executing {base_command}: {str(e)}",
                start_time=start_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"[EXECUTOR:{request_id}] ========== Unexpected Error ==========")
            logger.error(f"[EXECUTOR:{request_id}] Error type: {type(e).__name__}")
            logger.error(f"[EXECUTOR:{request_id}] Error message: {str(e)}")
            logger.exception(f"[EXECUTOR:{request_id}] Full traceback:")

            return self._create_error_result(
                request_id=request_id,
                error_code="UNEXPECTED_ERROR",
                error_message=f"Unexpected error: {type(e).__name__}: {str(e)}",
                start_time=start_time
            )


    def _execute_with_args_list(
        self,
        agent_type: str,
        args_list: list,
        timeout: int = 300,
        working_directory: Optional[str] = None,
        request_id: Optional[str] = None,
        stdin_input: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a coding agent with arguments as a list. Supports stdin input for long prompts."""
        if not request_id:
            request_id = str(uuid.uuid4())[:8]

        start_time = time.time()
        timestamp = datetime.now().isoformat()

        logger.info(f'[EXECUTOR:{request_id}] ========== Execution Started (list args) ==========')
        logger.info(f'[EXECUTOR:{request_id}] Agent type: {agent_type}, Timeout: {timeout}s')
        if stdin_input:
            logger.info(f'[EXECUTOR:{request_id}] Using stdin: {len(stdin_input)} chars')

        try:
            base_command = self._get_agent_command(agent_type)
        except ValueError as e:
            return self._create_error_result(request_id=request_id, error_code='INVALID_AGENT_TYPE',
                error_message=str(e), start_time=start_time)

        full_command, use_shell = self._resolve_command_for_platform(base_command, args_list)
        if use_shell:
            logger.info(f'[EXECUTOR:{request_id}] Shell command: {full_command[:200]}...')
        else:
            logger.info(f'[EXECUTOR:{request_id}] Command: {base_command} + {len(args_list)} args')

        cwd = Path(working_directory) if working_directory else None
        if cwd and not cwd.exists():
            return self._create_error_result(request_id=request_id, error_code='INVALID_WORKING_DIRECTORY',
                error_message=f'Working directory does not exist: {cwd}', start_time=start_time)

        try:
            result = subprocess.run(full_command, capture_output=True, text=True, timeout=timeout, input=stdin_input,
                cwd=cwd, shell=use_shell, encoding='utf-8', errors='replace')
            execution_time = time.time() - start_time
            success = result.returncode == 0
            stdout = result.stdout or ''
            stderr = result.stderr or ''
            logger.info(f'[EXECUTOR:{request_id}] Completed: exit_code={result.returncode}, time={execution_time:.2f}s')
            if stderr:
                logger.warning(f'[EXECUTOR:{request_id}] Stderr: {stderr}')
            return {'success': success, 'output': stdout, 'error': stderr if stderr else None,
                'exit_code': result.returncode, 'execution_time': execution_time,
                'request_id': request_id, 'timestamp': timestamp}
        except subprocess.TimeoutExpired as e:
            return self._create_error_result(request_id=request_id, error_code='EXECUTION_TIMEOUT',
                error_message=f'Timed out after {timeout}s', start_time=start_time,
                partial_output=e.stdout if e.stdout else '')
        except FileNotFoundError:
            return self._create_error_result(request_id=request_id, error_code='COMMAND_NOT_FOUND',
                error_message=f'Agent not found: {base_command}', start_time=start_time)
        except Exception as e:
            return self._create_error_result(request_id=request_id, error_code='UNEXPECTED_ERROR',
                error_message=f'{type(e).__name__}: {str(e)}', start_time=start_time)

    def execute_code_review(
        self,
        agent_type: str,
        context: str,
        build_id: str,
        user_id: str,
        timeout: int = 600,
        working_directory: Optional[str] = None,
        additional_args: Optional[str] = None,
        custom_template: Optional[str] = None,
        extra_template_args: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a code review using the specified agent.

        This method formats the code review context using a prompt template
        and executes the agent with appropriate arguments.

        Prompt template placeholders:
            {context}    - The code or content to review
            {build_id}   - Build identifier
            {user_id}    - User ID to notify
            {agent_type} - The agent type being used
            {timestamp}  - Current timestamp (ISO format)
            {date}       - Current date (YYYY-MM-DD)
            {time}       - Current time (HH:MM:SS)

        Args:
            agent_type: Type of agent to execute
            context: Code review context (diff, code, instructions, etc.)
            build_id: Build identifier for tracking
            user_id: User ID to notify after completion
            timeout: Maximum execution time in seconds
            working_directory: Working directory for execution
            additional_args: Additional command line arguments
            custom_template: Optional custom template string (overrides file-based template)
            extra_template_args: Additional key-value pairs for custom placeholders

        Returns:
            Dictionary containing code review results
        """
        request_id = f"CR-{str(uuid.uuid4())[:8]}"

        logger.info(f"[CODE_REVIEW:{request_id}] ========== Code Review Started ==========")
        logger.info(f"[CODE_REVIEW:{request_id}] --- Input Arguments ---")
        logger.info(f"[CODE_REVIEW:{request_id}]   Build ID: {build_id}")
        logger.info(f"[CODE_REVIEW:{request_id}]   User ID: {user_id}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Agent type: {agent_type}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Timeout: {timeout}s")
        logger.info(f"[CODE_REVIEW:{request_id}]   Working directory: {working_directory or 'default'}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Additional args: {additional_args or 'None'}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Custom template: {'Yes' if custom_template else 'No'}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Extra template args: {extra_template_args or 'None'}")
        logger.info(f"[CODE_REVIEW:{request_id}]   Context length: {len(context)} chars")
        logger.info(f"[CODE_REVIEW:{request_id}]   Context preview: {context[:200]}...")

        # Get the prompt template
        if custom_template:
            template = custom_template
            logger.info(f"[CODE_REVIEW:{request_id}] Using custom template provided in request")
        else:
            template = settings.get_prompt_template_for_agent(agent_type)
            logger.info(f"[CODE_REVIEW:{request_id}] Using template from file/config")

        # Format the prompt using the template
        extra_args = extra_template_args or {}
        prompt = format_prompt_template(
            template=template,
            context=context,
            build_id=build_id,
            user_id=user_id,
            agent_type=agent_type,
            **extra_args
        )

        logger.info(f"[CODE_REVIEW:{request_id}] --- Formatted Prompt ---")
        logger.info(f"[CODE_REVIEW:{request_id}]   Prompt length: {len(prompt)} chars")
        # Log prompt preview (first 1000 chars, max 20 lines)
        prompt_preview = prompt[:1000] + ('...' if len(prompt) > 1000 else '')
        prompt_lines = prompt_preview.split(chr(10))[:20]
        for line in prompt_lines:
            logger.info(f"[CODE_REVIEW:{request_id}]   | {line}")
        if len(prompt) > 1000 or len(prompt.split(chr(10))) > 20:
            logger.info(f"[CODE_REVIEW:{request_id}]   | ... (truncated)")

        # Build command arguments as a list (bypasses shlex.split issues with quotes)
        args_list = self._build_code_review_args_list(agent_type, prompt, additional_args)

        # Log the final CLI command
        base_command = self._get_agent_command(agent_type)
        logger.info(f"[CODE_REVIEW:{request_id}] --- Final CLI Command ---")
        logger.info(f"[CODE_REVIEW:{request_id}]   Executable: {base_command}")
        args_preview = ' '.join(str(a)[:50] for a in args_list[:3])
        logger.info(f"[CODE_REVIEW:{request_id}]   Args: {args_preview}...")

        # Execute using list-based args (bypasses shlex.split)
        # For gemini, pass prompt via stdin for better handling of long prompts
        stdin_input = prompt if agent_type.lower() == 'gemini' else None
        result = self._execute_with_args_list(
            agent_type=agent_type,
            args_list=args_list,
            timeout=timeout,
            working_directory=working_directory,
            request_id=request_id,
            stdin_input=stdin_input
        )

        # Add code review specific metadata
        result["build_id"] = build_id
        result["user_id"] = user_id
        result["agent_type"] = agent_type

        logger.info(f"[CODE_REVIEW:{request_id}] ========== Code Review Completed ==========")
        logger.info(f"[CODE_REVIEW:{request_id}] Success: {result.get('success', False)}")

        return result

    def _build_code_review_args_list(
        self,
        agent_type: str,
        prompt: str,
        additional_args: Optional[str] = None
    ) -> list:
        """
        Build command arguments for code review as a list (bypasses shlex.split issues).

        Different agents may require different argument formats.

        Args:
            agent_type: Type of agent
            prompt: The formatted prompt (no escaping needed - subprocess handles it)
            additional_args: Additional arguments to append (will be shlex.split)

        Returns:
            Command arguments as a list
        """
        agent_lower = agent_type.lower()
        args_list = []

        # Build args based on agent type
        # Reference documentation:
        #   Claude Code: https://github.com/anthropics/claude-code
        #   OpenAI Codex: https://github.com/openai/codex
        #   Google Gemini: https://github.com/google-gemini/gemini-cli
        #   Aider: https://github.com/Aider-AI/aider
        if agent_lower == "claude-code":
            # Claude Code: claude -p "prompt" (non-interactive print mode)
            args_list = ['-p', prompt]
        elif agent_lower == "codex":
            # OpenAI Codex CLI: codex exec "prompt" (non-interactive mode)
            args_list = ['exec', prompt]
        elif agent_lower == "gemini":
            # Google Gemini CLI: use stdin for long prompts (more reliable than cmd line)
            # Returns empty args - prompt will be passed via stdin in execute_code_review
            args_list = []
        elif agent_lower == "aider":
            # Aider: aider --message "prompt" (non-interactive mode)
            args_list = ['--message', prompt]
        elif agent_lower == "cursor":
            # Cursor: cursor "prompt" (basic prompt mode)
            args_list = [prompt]
        else:
            # Default: pass prompt as positional argument
            args_list = [prompt]

        # Append additional arguments if provided (these use shlex.split)
        if additional_args:
            try:
                extra_args = shlex.split(additional_args)
                args_list.extend(extra_args)
            except ValueError:
                # If shlex fails, just append as-is
                args_list.append(additional_args)

        return args_list

    def _create_error_result(
        self,
        request_id: str,
        error_code: str,
        error_message: str,
        start_time: float,
        partial_output: str = ""
    ) -> Dict[str, Any]:
        """
        Create a standardized error result dictionary.

        Args:
            request_id: Request identifier
            error_code: Error code for programmatic handling
            error_message: Human-readable error message
            start_time: Execution start time
            partial_output: Any partial output captured before error

        Returns:
            Standardized error result dictionary
        """
        execution_time = time.time() - start_time
        return {
            "success": False,
            "output": partial_output,
            "error": error_message,
            "error_code": error_code,
            "exit_code": -1,
            "execution_time": execution_time,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }

    def _resolve_command_for_platform(self, base_command: str, args_list: list) -> tuple:
        """
        Resolve command for platform-specific execution.

        On Windows, .CMD and .BAT files need shell=True with subprocess.list2cmdline()
        for proper argument quoting.

        Args:
            base_command: The base command name (e.g., 'gemini', 'codex')
            args_list: List of arguments to pass to the command

        Returns:
            Tuple of (command, use_shell):
            - command: str (for shell=True) or list (for shell=False)
            - use_shell: bool indicating whether to use shell=True
        """
        # Find the actual executable path
        resolved_path = shutil.which(base_command)

        if resolved_path and sys.platform == 'win32':
            # On Windows, .CMD/.BAT files need shell=True
            if resolved_path.lower().endswith(('.cmd', '.bat')):
                logger.debug(f"[EXECUTOR] Detected Windows batch file: {resolved_path}")
                # Use list2cmdline for proper Windows argument escaping
                full_args = [base_command] + args_list
                cmd_line = subprocess.list2cmdline(full_args)
                logger.debug(f"[EXECUTOR] Shell command: {cmd_line[:200]}...")
                return (cmd_line, True)

        # For regular executables, use list with shell=False
        return ([base_command] + args_list, False)

    def _get_agent_command(self, agent_type: str) -> str:
        """
        Get the executable command for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Base command string

        Raises:
            ValueError: If agent type is not supported
        """
        command = self._agent_commands.get(agent_type.lower())
        if not command:
            supported = ', '.join(self._agent_commands.keys())
            raise ValueError(
                f"Unsupported agent type: {agent_type}. Supported types: {supported}"
            )
        return command

    def list_supported_agents(self) -> list:
        """Get list of supported agent types"""
        return list(self._agent_commands.keys())

    def add_agent_command(self, agent_type: str, command: str) -> None:
        """
        Add or update an agent command mapping.

        Args:
            agent_type: Agent type identifier
            command: Executable command for the agent
        """
        self._agent_commands[agent_type.lower()] = command
        logger.info(f"[EXECUTOR] Added agent command: {agent_type} -> {command}")
