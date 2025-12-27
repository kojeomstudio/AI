# Custom Prompt Template Example
# ================================
# Copy this file to 'custom_template.txt' and modify as needed.
#
# Available placeholders:
#   {context}    - The code or content to review (required)
#   {build_id}   - Build identifier for tracking
#   {user_id}    - User ID to notify
#   {agent_type} - The agent type being used
#   {timestamp}  - Current timestamp (ISO format)
#
# You can add any custom text around these placeholders.
# The placeholders will be replaced with actual values at runtime.

=== Custom Code Review Request ===

Project Build: {build_id}
Reviewer Assignment: {user_id}
Agent: {agent_type}
Time: {timestamp}

--- BEGIN CODE ---
{context}
--- END CODE ---

Please analyze the above code and provide:
1. Summary of changes
2. Issues found (if any)
3. Recommendations

Thank you!
