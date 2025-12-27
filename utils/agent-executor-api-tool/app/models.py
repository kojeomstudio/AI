"""
Pydantic models for agent executor API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class AgentExecuteRequest(BaseModel):
    """Request model for executing a coding agent"""
    agent_type: str = Field(
        ...,
        description="Type of coding agent to execute (e.g., 'claude-code', 'codex', 'gemini')",
        example="claude-code"
    )
    command_args: str = Field(
        ...,
        description="Complete command line arguments including prompt (e.g., '-exec \"something works\"')",
        example="-exec \"Create a hello world function\""
    )
    timeout: Optional[int] = Field(
        default=900,
        description="Timeout in seconds for agent execution",
        example=300
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory for the agent execution",
        example="D:\\projects\\myproject"
    )


class CodeReviewRequest(BaseModel):
    """
    Request model for code review execution via JSON.

    The prompt template supports the following placeholders:
        {context}    - The code or content to review
        {build_id}   - Build identifier
        {user_id}    - User ID to notify
        {agent_type} - The agent type being used
        {timestamp}  - Current timestamp (ISO format)
        {date}       - Current date (YYYY-MM-DD)
        {time}       - Current time (HH:MM:SS)

    Additional custom placeholders can be passed via extra_template_args.
    """
    agent_type: str = Field(
        default="claude-code",
        description="Type of coding agent to execute (e.g., 'claude-code', 'codex', 'gemini')",
        example="claude-code"
    )
    context: str = Field(
        ...,
        description="Code review context - the content to be reviewed (diff, code, instructions, etc.)",
        example="Review the following code changes:\\n+ def hello():\\n+     print('hello')"
    )
    build_id: str = Field(
        ...,
        description="Build identifier for tracking the code review request",
        example="build-12345"
    )
    user_id: str = Field(
        ...,
        description="User ID to send notification after review completion",
        example="user-abc123"
    )
    timeout: Optional[int] = Field(
        default=900,
        description="Timeout in seconds for agent execution (default: 10 minutes)",
        example=600
    )
    working_directory: Optional[str] = Field(
        default=None,
        description="Working directory for the agent execution",
        example="D:\\projects\\myproject"
    )
    additional_args: Optional[str] = Field(
        default=None,
        description="Additional command line arguments for the agent",
        example="--verbose"
    )
    custom_template: Optional[str] = Field(
        default=None,
        description="Custom prompt template string (overrides file-based template). Use placeholders like {context}, {build_id}, {user_id}",
        example="Review this code for {user_id}:\\n{context}"
    )
    extra_template_args: Optional[Dict[str, str]] = Field(
        default=None,
        description="Additional key-value pairs for custom placeholders in the template",
        example={"project_name": "MyProject", "reviewer": "Senior Dev"}
    )


class AgentExecuteResponse(BaseModel):
    """Response model for agent execution result"""
    success: bool = Field(..., description="Whether the execution was successful")
    output: str = Field(..., description="Standard output from the agent")
    error: Optional[str] = Field(None, description="Error output if any")
    exit_code: int = Field(..., description="Exit code from the agent process")
    execution_time: float = Field(..., description="Execution time in seconds")


class CodeReviewResponse(BaseModel):
    """Response model for code review execution result"""
    success: bool = Field(..., description="Whether the execution was successful")
    build_id: str = Field(..., description="Build ID that was processed")
    user_id: str = Field(..., description="User ID to be notified")
    agent_type: str = Field(..., description="Agent type that was used")
    output: str = Field(..., description="Standard output from the agent (review result)")
    error: Optional[str] = Field(None, description="Error output if any")
    exit_code: int = Field(..., description="Exit code from the agent process")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(..., description="Timestamp when the review was completed")


class ErrorResponse(BaseModel):
    """Detailed error response model"""
    success: bool = Field(default=False, description="Always false for error responses")
    error_code: str = Field(..., description="Error code for programmatic handling")
    error_message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Timestamp when the error occurred")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    build_id: Optional[str] = Field(None, description="Build ID if available")
    user_id: Optional[str] = Field(None, description="User ID if available")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
