"""
Configuration validation script
Displays current configuration values loaded from config.json
"""
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings


def display_settings():
    """Display all current settings"""
    print("=" * 70)
    print("Agent Executor API - Configuration Check")
    print("=" * 70)
    print()

    print("API Settings:")
    print("-" * 70)
    print(f"  App Name:            {settings.app_name}")
    print(f"  App Version:         {settings.app_version}")
    print(f"  Host:                {settings.host}")
    print(f"  Port:                {settings.port}")
    print(f"  Reload:              {settings.reload}")
    print()

    print("Execution Settings:")
    print("-" * 70)
    print(f"  Default Timeout:     {settings.default_timeout} seconds")
    print(f"  Max Timeout:         {settings.max_timeout} seconds")
    print(f"  Default Work Dir:    {settings.default_working_directory or '(not set)'}")
    print()

    print("Logging Settings:")
    print("-" * 70)
    print(f"  Log Level:           {settings.log_level}")
    print()

    print("CORS Settings:")
    print("-" * 70)
    print(f"  Allow Origins:       {', '.join(settings.allow_origins)}")
    print(f"  Allow Credentials:   {settings.allow_credentials}")
    print(f"  Allow Methods:       {', '.join(settings.allow_methods)}")
    print(f"  Allow Headers:       {', '.join(settings.allow_headers)}")
    print()

    print("=" * 70)
    print("Configuration loaded successfully!")
    print("=" * 70)
    print()

    # Check for config.json file
    config_file = Path(__file__).parent / "config.json"
    if config_file.exists():
        print(f"[OK] config.json file found at: {config_file}")
    else:
        print(f"[WARNING] config.json file not found. Using default values.")
        print(f"  Copy config.json.example to config.json to customize settings.")
    print()


def validate_settings():
    """Validate settings for common issues"""
    print("Validation Checks:")
    print("-" * 70)

    issues = []
    warnings = []

    # Check timeout values
    if settings.default_timeout > settings.max_timeout:
        issues.append("DEFAULT_TIMEOUT is greater than MAX_TIMEOUT")

    if settings.default_timeout <= 0:
        issues.append("DEFAULT_TIMEOUT must be positive")

    if settings.max_timeout <= 0:
        issues.append("MAX_TIMEOUT must be positive")

    # Check port range
    if not (1 <= settings.port <= 65535):
        issues.append(f"PORT must be between 1 and 65535 (current: {settings.port})")

    # Check log level
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if settings.log_level.upper() not in valid_log_levels:
        issues.append(f"LOG_LEVEL must be one of {', '.join(valid_log_levels)}")

    # Check CORS settings for production
    if settings.host == "0.0.0.0" and "*" in settings.allow_origins:
        warnings.append("ALLOW_ORIGINS is set to '*' with public host. Consider restricting for production.")

    # Display results
    if issues:
        print("[ERROR] Configuration Issues Found:")
        for issue in issues:
            print(f"  * {issue}")
        print()
        return False
    else:
        print("[OK] No configuration issues found")
        print()

    if warnings:
        print("[WARNING] Configuration Warnings:")
        for warning in warnings:
            print(f"  * {warning}")
        print()

    return True


if __name__ == "__main__":
    try:
        display_settings()
        is_valid = validate_settings()

        if is_valid:
            print("Configuration is valid and ready to use!")
            sys.exit(0)
        else:
            print("Please fix the configuration issues before starting the server.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
