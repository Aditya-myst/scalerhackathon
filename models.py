from pydantic import Field
from typing import Dict, Any, Optional
from openenv.core.env_server import Action, Observation, State

class SQLAction(Action):
    """What the AI agent can do."""
    command: str = ""
    parameters: dict = Field(default_factory=dict)

class SQLObservation(Observation):
    """What the AI agent sees after taking an action."""
    result: str = ""
    success: bool = True
    error: Optional[str] = None

class SQLState(State):
    """What we track secretly behind the scenes."""
    current_level: str = "easy"
    tasks_completed: int = 0