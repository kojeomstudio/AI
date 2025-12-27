"""
FastAPI application for executing coding agents via HTTP API

This application provides HTTP endpoints for executing coding agents (Claude Code, Codex, Gemini, etc.)
as subprocess processes. It supports multiple input formats (JSON, form-data, raw) and provides
detailed logging and error responses.

Features:
- Multiple API endpoints for different input formats
- Code review execution with context, build_id, and user_id
- Detailed console logging with request tracking
- Comprehensive error responses with error codes
- Configuration via config.json and environment variables
"""
import sys
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional
import logging
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI, HTTPException, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Handle imports for both module mode and PyInstaller frozen mode
try:
    from .models import (
        AgentExecuteRequest,
        AgentExecuteResponse,
        CodeReviewRequest,
        CodeReviewResponse,
        ErrorResponse,
        HealthResponse
    )
    from .executor import AgentExecutor
    from .config import settings
except ImportError:
    # Fallback for frozen executable (PyInstaller)
    from models import (
        AgentExecuteRequest,
        AgentExecuteResponse,
        CodeReviewRequest,
        CodeReviewResponse,
        ErrorResponse,
        HealthResponse
    )
    from executor import AgentExecutor
    from config import settings


def setup_logging():
    """Configure logging with detailed format and optional file output"""
    handlers = [logging.StreamHandler(sys.stdout)]

    # Add file handler if configured
    if settings.log_to_file and settings.log_file_path:
        file_handler = RotatingFileHandler(
            settings.log_file_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding='utf-8'
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        handlers=handlers
    )


# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("=" * 60)
    logger.info(f"Server running on {settings.host}:{settings.port}")
    logger.info(f"Log level: {settings.log_level}")
    logger.info(f"Default timeout: {settings.default_timeout}s")
    logger.info(f"Max timeout: {settings.max_timeout}s")
    logger.info(f"Configuration info: {settings.get_config_info()}")
    logger.info("=" * 60)
    yield
    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Agent Executor API")
    logger.info("=" * 60)


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    FastAPI-based proxy server for executing coding agents (Claude Code, Codex, Gemini, etc.)

    ## Features
    - Execute coding agents via HTTP API
    - Multiple input formats: JSON, form-data, raw text
    - Code review execution with build tracking
    - Detailed logging and error responses

    ## Endpoints
    - `POST /execute` - Execute agent with JSON body
    - `POST /review/json` - Code review with JSON body
    - `POST /review/form` - Code review with form-data
    - `POST /review/raw` - Code review with raw text body
    """,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allow_methods,
    allow_headers=settings.allow_headers,
)

# Initialize executor
executor = AgentExecutor()


def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]


def create_error_response(
    error_code: str,
    error_message: str,
    detail: Optional[str] = None,
    request_id: Optional[str] = None,
    build_id: Optional[str] = None,
    user_id: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """Create a standardized error response"""
    error_data = ErrorResponse(
        success=False,
        error_code=error_code,
        error_message=error_message,
        detail=detail,
        timestamp=datetime.now().isoformat(),
        request_id=request_id,
        build_id=build_id,
        user_id=user_id
    )

    logger.error(f"[API:{request_id}] Error response: {error_code} - {error_message}")
    if detail:
        logger.error(f"[API:{request_id}] Detail: {detail}")

    return JSONResponse(
        status_code=status_code,
        content=error_data.model_dump()
    )


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    logger.debug("[API] Health check request at root")
    return HealthResponse(
        status="healthy",
        version=settings.app_version
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    logger.debug("[API] Health check request")
    return HealthResponse(
        status="healthy",
        version=settings.app_version
    )


@app.get("/agents")
async def list_agents():
    """List supported agent types"""
    logger.info("[API] Listing supported agents")
    return {
        "supported_agents": executor.list_supported_agents(),
        "description": "List of supported coding agent types"
    }


@app.get("/config")
async def get_config():
    """Get current configuration (non-sensitive)"""
    logger.info("[API] Configuration info requested")
    return settings.get_config_info()


# =============================================================================
# Generic Agent Execution Endpoint (Original)
# =============================================================================

@app.post("/execute", response_model=AgentExecuteResponse)
async def execute_agent(request: AgentExecuteRequest):
    """
    Execute a coding agent with the provided configuration (JSON body)

    Args:
        request: Agent execution request containing agent type, command args, and options

    Returns:
        AgentExecuteResponse with execution results

    Raises:
        HTTPException: If execution fails or validation errors occur
    """
    request_id = generate_request_id()

    logger.info(f"[API:{request_id}] ========== Execute Request (JSON) ==========")
    logger.info(f"[API:{request_id}] Agent type: {request.agent_type}")
    logger.info(f"[API:{request_id}] Timeout: {request.timeout}s")
    logger.info(f"[API:{request_id}] Working directory: {request.working_directory or 'default'}")
    logger.debug(f"[API:{request_id}] Command args: {request.command_args}")

    # Validate timeout
    if request.timeout > settings.max_timeout:
        return create_error_response(
            error_code="TIMEOUT_EXCEEDED",
            error_message=f"Timeout exceeds maximum allowed value of {settings.max_timeout} seconds",
            detail=f"Requested: {request.timeout}s, Maximum: {settings.max_timeout}s",
            request_id=request_id
        )

    # Validate agent type
    if request.agent_type.lower() not in executor.list_supported_agents():
        return create_error_response(
            error_code="UNSUPPORTED_AGENT",
            error_message=f"Unsupported agent type: {request.agent_type}",
            detail=f"Supported types: {', '.join(executor.list_supported_agents())}",
            request_id=request_id
        )

    # Execute the agent
    try:
        result = executor.execute(
            agent_type=request.agent_type,
            command_args=request.command_args,
            timeout=request.timeout,
            working_directory=request.working_directory or settings.default_working_directory,
            request_id=request_id
        )

        logger.info(f"[API:{request_id}] Execution completed - Success: {result['success']}")
        return AgentExecuteResponse(**result)

    except ValueError as e:
        logger.error(f"[API:{request_id}] Validation error: {e}")
        return create_error_response(
            error_code="VALIDATION_ERROR",
            error_message=str(e),
            request_id=request_id
        )

    except Exception as e:
        logger.error(f"[API:{request_id}] Unexpected error during agent execution: {e}", exc_info=True)
        return create_error_response(
            error_code="INTERNAL_ERROR",
            error_message="Internal server error during execution",
            detail=str(e),
            request_id=request_id,
            status_code=500
        )


# =============================================================================
# Code Review Endpoints (Multiple Input Formats)
# =============================================================================

@app.post("/review/json", response_model=CodeReviewResponse)
async def code_review_json(request: CodeReviewRequest):
    """
    Execute a code review with JSON body

    Request body (JSON):
    - agent_type: Type of coding agent (default: claude-code)
    - context: Code review context (required)
    - build_id: Build identifier (required)
    - user_id: User ID to notify (required)
    - timeout: Timeout in seconds (default: 600)
    - working_directory: Working directory (optional)
    - additional_args: Additional agent arguments (optional)
    - custom_template: Custom prompt template string (optional)
    - extra_template_args: Additional template placeholders (optional)

    Prompt template placeholders:
    - {context}: Code review context
    - {build_id}: Build identifier
    - {user_id}: User ID
    - {agent_type}: Agent type
    - {timestamp}: Current timestamp
    - {date}: Current date
    - {time}: Current time
    """
    request_id = generate_request_id()

    logger.info(f"[API:{request_id}] ========== Code Review Request (JSON) ==========")
    logger.info(f"[API:{request_id}] Build ID: {request.build_id}")
    logger.info(f"[API:{request_id}] User ID: {request.user_id}")
    logger.info(f"[API:{request_id}] Agent type: {request.agent_type}")
    logger.info(f"[API:{request_id}] Timeout: {request.timeout}s")
    logger.info(f"[API:{request_id}] Context length: {len(request.context)} chars")
    logger.info(f"[API:{request_id}] Custom template: {'Yes' if request.custom_template else 'No'}")

    return await _execute_code_review(
        request_id=request_id,
        agent_type=request.agent_type,
        context=request.context,
        build_id=request.build_id,
        user_id=request.user_id,
        timeout=request.timeout,
        working_directory=request.working_directory,
        additional_args=request.additional_args,
        custom_template=request.custom_template,
        extra_template_args=request.extra_template_args
    )


@app.post("/review/form", response_model=CodeReviewResponse)
async def code_review_form(
    context: str = Form(..., description="Code review context"),
    build_id: str = Form(..., description="Build identifier"),
    user_id: str = Form(..., description="User ID to notify"),
    agent_type: str = Form(default="claude-code", description="Agent type to use"),
    timeout: int = Form(default=600, description="Timeout in seconds"),
    working_directory: Optional[str] = Form(default=None, description="Working directory"),
    additional_args: Optional[str] = Form(default=None, description="Additional arguments"),
    custom_template: Optional[str] = Form(default=None, description="Custom prompt template (optional)")
):
    """
    Execute a code review with form-data

    Form fields:
    - context: Code review context (required)
    - build_id: Build identifier (required)
    - user_id: User ID to notify (required)
    - agent_type: Type of coding agent (default: claude-code)
    - timeout: Timeout in seconds (default: 600)
    - working_directory: Working directory (optional)
    - additional_args: Additional agent arguments (optional)
    - custom_template: Custom prompt template string (optional)

    Prompt template placeholders: {context}, {build_id}, {user_id}, {agent_type}, {timestamp}, {date}, {time}
    """
    request_id = generate_request_id()

    logger.info(f"[API:{request_id}] ========== Code Review Request (Form-Data) ==========")
    logger.info(f"[API:{request_id}] Build ID: {build_id}")
    logger.info(f"[API:{request_id}] User ID: {user_id}")
    logger.info(f"[API:{request_id}] Agent type: {agent_type}")
    logger.info(f"[API:{request_id}] Timeout: {timeout}s")
    logger.info(f"[API:{request_id}] Context length: {len(context)} chars")
    logger.info(f"[API:{request_id}] Custom template: {'Yes' if custom_template else 'No'}")

    return await _execute_code_review(
        request_id=request_id,
        agent_type=agent_type,
        context=context,
        build_id=build_id,
        user_id=user_id,
        timeout=timeout,
        working_directory=working_directory,
        additional_args=additional_args,
        custom_template=custom_template,
        extra_template_args=None
    )


@app.post("/review/raw", response_model=CodeReviewResponse)
async def code_review_raw(
    request: Request,
    build_id: str,
    user_id: str,
    agent_type: str = "claude-code",
    timeout: int = 600,
    working_directory: Optional[str] = None,
    additional_args: Optional[str] = None,
    custom_template: Optional[str] = None
):
    """
    Execute a code review with raw text body (context in body, params in query)

    Query parameters:
    - build_id: Build identifier (required)
    - user_id: User ID to notify (required)
    - agent_type: Type of coding agent (default: claude-code)
    - timeout: Timeout in seconds (default: 600)
    - working_directory: Working directory (optional)
    - additional_args: Additional agent arguments (optional)
    - custom_template: Custom prompt template string (optional)

    Request body (raw text):
    - The code review context as plain text

    Prompt template placeholders: {context}, {build_id}, {user_id}, {agent_type}, {timestamp}, {date}, {time}
    """
    request_id = generate_request_id()

    # Read raw body
    body_bytes = await request.body()
    try:
        context = body_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return create_error_response(
            error_code="INVALID_ENCODING",
            error_message="Request body must be valid UTF-8 text",
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    if not context.strip():
        return create_error_response(
            error_code="EMPTY_CONTEXT",
            error_message="Code review context cannot be empty",
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    logger.info(f"[API:{request_id}] ========== Code Review Request (Raw) ==========")
    logger.info(f"[API:{request_id}] Build ID: {build_id}")
    logger.info(f"[API:{request_id}] User ID: {user_id}")
    logger.info(f"[API:{request_id}] Agent type: {agent_type}")
    logger.info(f"[API:{request_id}] Timeout: {timeout}s")
    logger.info(f"[API:{request_id}] Context length: {len(context)} chars")
    logger.info(f"[API:{request_id}] Custom template: {'Yes' if custom_template else 'No'}")

    return await _execute_code_review(
        request_id=request_id,
        agent_type=agent_type,
        context=context,
        build_id=build_id,
        user_id=user_id,
        timeout=timeout,
        working_directory=working_directory,
        additional_args=additional_args,
        custom_template=custom_template,
        extra_template_args=None
    )


async def _execute_code_review(
    request_id: str,
    agent_type: str,
    context: str,
    build_id: str,
    user_id: str,
    timeout: int,
    working_directory: Optional[str],
    additional_args: Optional[str],
    custom_template: Optional[str] = None,
    extra_template_args: Optional[dict] = None
) -> CodeReviewResponse:
    """
    Internal function to execute code review with common validation and error handling

    Args:
        request_id: Request tracking ID
        agent_type: Type of coding agent
        context: Code review context
        build_id: Build identifier
        user_id: User ID to notify
        timeout: Timeout in seconds
        working_directory: Working directory for execution
        additional_args: Additional agent arguments
        custom_template: Custom prompt template string (optional)
        extra_template_args: Additional template placeholders (optional)

    Returns:
        CodeReviewResponse with execution results
    """
    # Validate timeout
    if timeout > settings.max_timeout:
        logger.warning(f"[API:{request_id}] Timeout {timeout}s exceeds max {settings.max_timeout}s")
        return create_error_response(
            error_code="TIMEOUT_EXCEEDED",
            error_message=f"Timeout exceeds maximum allowed value of {settings.max_timeout} seconds",
            detail=f"Requested: {timeout}s, Maximum: {settings.max_timeout}s",
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    # Validate agent type
    if agent_type.lower() not in executor.list_supported_agents():
        logger.warning(f"[API:{request_id}] Unsupported agent type: {agent_type}")
        return create_error_response(
            error_code="UNSUPPORTED_AGENT",
            error_message=f"Unsupported agent type: {agent_type}",
            detail=f"Supported types: {', '.join(executor.list_supported_agents())}",
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    # Validate context
    if not context or not context.strip():
        logger.warning(f"[API:{request_id}] Empty context provided")
        return create_error_response(
            error_code="EMPTY_CONTEXT",
            error_message="Code review context cannot be empty",
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    # Execute the code review
    try:
        result = executor.execute_code_review(
            agent_type=agent_type,
            context=context,
            build_id=build_id,
            user_id=user_id,
            timeout=timeout,
            working_directory=working_directory or settings.default_working_directory,
            additional_args=additional_args,
            custom_template=custom_template,
            extra_template_args=extra_template_args
        )

        timestamp = result.get('timestamp', datetime.now().isoformat())

        logger.info(f"[API:{request_id}] Code review completed - Success: {result['success']}")

        return CodeReviewResponse(
            success=result['success'],
            build_id=build_id,
            user_id=user_id,
            agent_type=agent_type,
            output=result['output'],
            error=result.get('error'),
            exit_code=result['exit_code'],
            execution_time=result['execution_time'],
            timestamp=timestamp
        )

    except ValueError as e:
        logger.error(f"[API:{request_id}] Validation error: {e}")
        return create_error_response(
            error_code="VALIDATION_ERROR",
            error_message=str(e),
            request_id=request_id,
            build_id=build_id,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"[API:{request_id}] Unexpected error during code review: {e}", exc_info=True)
        return create_error_response(
            error_code="INTERNAL_ERROR",
            error_message="Internal server error during code review",
            detail=str(e),
            request_id=request_id,
            build_id=build_id,
            user_id=user_id,
            status_code=500
        )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with detailed responses"""
    request_id = generate_request_id()
    logger.error(f"[API:{request_id}] HTTP Exception: {exc.status_code} - {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "error_message": str(exc.detail),
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    request_id = generate_request_id()
    logger.error(f"[API:{request_id}] Unhandled exception: {type(exc).__name__}: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "error_message": "An unexpected error occurred",
            "detail": f"{type(exc).__name__}: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server from __main__")
    
    # Check if running as frozen executable (PyInstaller)
    if getattr(sys, 'frozen', False):
        # Running as frozen executable - use app object directly
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level.lower()
        )
    else:
        # Running as module - use string reference for reload support
        uvicorn.run(
            "app.main:app",
            host=settings.host,
            port=settings.port,
            reload=settings.reload,
            log_level=settings.log_level.lower()
        )
