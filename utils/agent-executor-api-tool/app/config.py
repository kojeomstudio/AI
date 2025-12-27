"""
Configuration settings for agent executor API

Configuration is loaded from multiple sources in the following priority (highest first):
1. Environment variables (system level)
2. config.json file

This allows for flexible configuration management across different deployment scenarios.
"""
import json
import sys
import os
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Union, List, Dict, Any
from datetime import datetime


def get_config_directory() -> Path:
    """
    Get the configuration directory.
    Handles both development and PyInstaller frozen executable scenarios.

    Returns:
        Path to the configuration directory
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return Path(sys.executable).parent
    else:
        # Running in development
        return Path(__file__).parent.parent


def load_json_config() -> Dict[str, Any]:
    """
    Load configuration from config.json file.

    Returns:
        Dictionary containing configuration values, or empty dict if file not found
    """
    config_dir = get_config_directory()
    config_file = config_dir / "config.json"

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                print(f"[CONFIG] Loaded configuration from: {config_file}")
                return config
        except json.JSONDecodeError as e:
            print(f"[CONFIG] Warning: Failed to parse config.json: {e}")
            return {}
        except Exception as e:
            print(f"[CONFIG] Warning: Failed to read config.json: {e}")
            return {}
    else:
        print(f"[CONFIG] No config.json found at: {config_file}, using defaults")
        return {}


def get_prompts_directory() -> Path:
    """
    Get the prompts directory path.

    Returns:
        Path to the prompts directory
    """
    return get_config_directory() / "prompts"


def load_prompt_template(template_name: str) -> Optional[str]:
    """
    Load a prompt template from file.

    Args:
        template_name: Name of the template file (with or without .md extension)

    Returns:
        Template content as string, or None if file not found
    """
    prompts_dir = get_prompts_directory()

    # Add .md extension if not present
    if not template_name.endswith('.md'):
        template_name = f"{template_name}.md"

    template_path = prompts_dir / template_name

    if template_path.exists():
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"[CONFIG] Loaded prompt template from: {template_path}")
                return content
        except Exception as e:
            print(f"[CONFIG] Warning: Failed to read template {template_name}: {e}")
            return None
    else:
        print(f"[CONFIG] Template not found: {template_path}")
        return None


def format_prompt_template(
    template: str,
    context: str,
    build_id: str,
    user_id: str,
    agent_type: str = "claude-code",
    **extra_args
) -> str:
    """
    Format a prompt template with the provided arguments.

    Supported placeholders:
        {context}    - The code or content to review
        {build_id}   - Build identifier
        {user_id}    - User ID to notify
        {agent_type} - The agent type being used
        {timestamp}  - Current timestamp (ISO format)
        {date}       - Current date (YYYY-MM-DD)
        {time}       - Current time (HH:MM:SS)

    Additional placeholders can be passed via **extra_args.

    Args:
        template: The template string with placeholders
        context: Code review context
        build_id: Build identifier
        user_id: User ID
        agent_type: Agent type
        **extra_args: Additional key-value pairs for custom placeholders

    Returns:
        Formatted prompt string
    """
    now = datetime.now()

    # Build the format arguments
    format_args = {
        "context": context,
        "build_id": build_id,
        "user_id": user_id,
        "agent_type": agent_type,
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        **extra_args
    }

    # Use safe formatting that ignores missing keys
    try:
        # First try standard format
        return template.format(**format_args)
    except KeyError as e:
        # If there's a missing key, use a safer approach
        print(f"[CONFIG] Warning: Missing placeholder in template: {e}")
        result = template
        for key, value in format_args.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


# Load JSON config once at module import
_json_config = load_json_config()


class Settings(BaseSettings):
    """
    Application settings with multi-source configuration support.

    Priority (highest to lowest):
    1. Environment variables (system level)
    2. config.json file
    3. Default values
    """

    # API Settings
    app_name: str = _json_config.get("app_name", "Agent Executor API")
    app_version: str = _json_config.get("app_version", "1.2.0")
    host: str = _json_config.get("host", "0.0.0.0")
    port: int = _json_config.get("port", 8000)
    reload: bool = _json_config.get("reload", False)

    # Execution Settings
    default_timeout: int = _json_config.get("default_timeout", 900)  # 15 minutes
    max_timeout: int = _json_config.get("max_timeout", 3600)  # 1 hour
    default_working_directory: Optional[str] = _json_config.get("default_working_directory", None)

    # Agent Settings
    agent_commands: Dict[str, str] = _json_config.get("agent_commands", {
        "claude-code": "claude",
        "codex": "codex",
        "gemini": "gemini",
        "cursor": "cursor",
        "aider": "aider"
    })

    # Prompt Template Settings
    # Supports both .md (preferred) and .txt extensions
    prompt_template_file: str = _json_config.get("prompt_template_file", "code_review.md")
    prompt_templates_dir: str = _json_config.get("prompt_templates_dir", "prompts")

    # Agent-specific prompt templates (optional, falls back to default)
    agent_prompt_templates: Dict[str, str] = _json_config.get("agent_prompt_templates", {
        "claude-code": "code_review_claude.md",
        "codex": "code_review.md",
        "gemini": "code_review.md",
        "cursor": "code_review.md",
        "aider": "code_review.md"
    })

    # Logging Settings
    log_level: str = _json_config.get("log_level", "INFO")
    log_format: str = _json_config.get(
        "log_format",
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    )
    log_to_file: bool = _json_config.get("log_to_file", False)
    log_file_path: Optional[str] = _json_config.get("log_file_path", "agent_executor.log")
    log_max_bytes: int = _json_config.get("log_max_bytes", 10485760)  # 10MB
    log_backup_count: int = _json_config.get("log_backup_count", 5)

    # CORS Settings
    # These can be provided as comma-separated strings via environment variables
    # Example: ALLOW_ORIGINS=http://localhost:3000,http://localhost:8080
    allow_origins: Union[str, List[str]] = _json_config.get("allow_origins", "*")
    allow_credentials: bool = _json_config.get("allow_credentials", True)
    allow_methods: Union[str, List[str]] = _json_config.get("allow_methods", "*")
    allow_headers: Union[str, List[str]] = _json_config.get("allow_headers", "*")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("allow_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("allow_methods", mode="before")
    @classmethod
    def parse_cors_methods(cls, v):
        """Parse CORS methods from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [method.strip().upper() for method in v.split(",") if method.strip()]
        return v

    @field_validator("allow_headers", mode="before")
    @classmethod
    def parse_cors_headers(cls, v):
        """Parse CORS headers from string or list"""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [header.strip() for header in v.split(",") if header.strip()]
        return v

    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for logging/debugging"""
        return {
            "config_directory": str(get_config_directory()),
            "prompts_directory": str(get_prompts_directory()),
            "config_json_loaded": bool(_json_config),
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
            "default_timeout": self.default_timeout,
            "max_timeout": self.max_timeout,
            "agent_commands": self.agent_commands,
            "prompt_template_file": self.prompt_template_file,
            "agent_prompt_templates": self.agent_prompt_templates
        }

    def get_prompt_template_for_agent(self, agent_type: str) -> str:
        """
        Get the prompt template for a specific agent type.

        First tries to load agent-specific template from file,
        then falls back to default template file.
        Raises FileNotFoundError if no template is found.

        Args:
            agent_type: The agent type (e.g., 'claude-code', 'gemini')

        Returns:
            Template string with placeholders
        """
        # Try agent-specific template
        agent_template_file = self.agent_prompt_templates.get(agent_type.lower())
        if agent_template_file:
            template = load_prompt_template(agent_template_file)
            if template:
                return template

        # Try default template file
        template = load_prompt_template(self.prompt_template_file)
        if template:
            return template

        # No template found - raise error
        raise FileNotFoundError(
            f"No prompt template found for agent '{agent_type}'. "
            f"Please ensure template files exist in the prompts directory."
        )


# Global settings instance
settings = Settings()
