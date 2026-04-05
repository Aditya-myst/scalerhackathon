from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SQLAction, SQLObservation, SQLState

class DataEngineerClient(EnvClient[SQLAction, SQLObservation, SQLState]):
    """The Walkie Talkie that the AI uses to talk to the environment."""
    
    def _step_payload(self, action: SQLAction) -> dict:
        return {"command": action.command, "parameters": action.parameters}

    def _parse_result(self, payload: dict) -> StepResult[SQLObservation]:
        obs = SQLObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SQLState:
        return SQLState(**payload)