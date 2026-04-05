# AI Data Engineer Escape Room (OpenEnv)
**Meta × Hugging Face OpenEnv Hackathon Submission**

[![Hosted on Hugging Face Spaces](https://img.shields.io/badge/Hosted%20on-Hugging%20Face-blue)](https://huggingface.co/spaces/aadiiityaa007/data_engineer_env)
[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green)](#)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](#)

*(Drag and drop your demo video `.mp4` here when editing on GitHub!)*

## 📖 Overview
The **Data Engineer Escape Room** is a Reinforcement Learning (RL) environment designed to evaluate and train autonomous AI agents in real-world data engineering tasks. 

Instead of simple text games, this environment provides a secure, in-memory **SQLite sandbox** and virtual file system. The agent must autonomously read raw data (JSON), construct relational databases, and clean corrupted datasets (e.g., negative ages, missing fields) using raw SQL commands.

## 🎯 Hackathon Criteria Met
1. **Built on OpenEnv:** Uses `openenv.core.env_server` for standardized WebSocket communication.
2. **Hugging Face Integration:** Fully dockerized and deployed live on Hugging Face Spaces.
3. **Strict Graders:** Tasks aren't graded by LLM vibes; they are graded by hidden SQL queries executing against the agent's database to verify exact row counts and data integrity.
4. **Resource-Optimized RL (Energy Penalties):** The agent is penalized `-0.01` reward points for every action (compute cycle) it takes, forcing it to write efficient SQL pipelines rather than blindly guessing or spamming the server.

---

## Hackathon requirement mapping

This implementation aligns with OpenEnv hackathon expectations:

1. **Built on OpenEnv**
   - Uses OpenEnv server/client abstractions for standardized environment interaction.

2. **Deployed on Hugging Face Spaces**
   - Containerized and hosted as a live environment endpoint.

3. **Strict grading**
   - Task completion is validated via hidden SQL checks against actual DB state.

4. **Reward shaping**
   - Includes per-step penalty and positive rewards for meaningful progress/task completion.

5. **Autonomous inference path**
   - Includes `inference.py` in repository root for standardized automated evaluation.

---

## Architecture

High-level flow:

1. Agent calls `reset()`.
2. Environment returns initial observation with system task instructions.
3. Agent iteratively issues actions:
   - `read_file`
   - `execute_sql`
   - `submit_task`
4. Environment executes action in sandbox and returns:
   - observation
   - reward
   - done
   - success/failure details
5. Agent stops on terminal state or max steps.
6. Script emits strict logs for evaluator parsing.

---

## Environment mechanics

### Action space

The agent acts through JSON commands:

- `read_file`
  - Example:
    ```json
    {"command":"read_file","parameters":{"filename":"users.json"}}
    ```

- `execute_sql`
  - Example:
    ```json
    {"command":"execute_sql","parameters":{"query":"CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)"}}
    ```

- `submit_task`
  - Example:
    ```json
    {"command":"submit_task","parameters":{"task":"easy"}}
    ```

---

### Reward model

Typical shaping (environment-config dependent):

- `-0.01` per step (efficiency penalty)
- negative reward for invalid actions / SQL errors
- positive reward for valid progress
- larger positive reward on validated task completion

This reward structure discourages random action spam and encourages robust planning.

---

### Levels

- **Easy**: basic table creation + data insertion
- **Medium**: additional schema/data constraints and correctness checks
- **Hard**: corrupted data cleanup and strict validation constraints

---

## Endpoints and connectivity

- **Environment base URL (Space):**  
  `https://aadiiityaa007-data-engineer-env.hf.space`

OpenEnv client handles the underlying protocol details.  
In client code, connect via:

```python
from openenv.core.env_client import EnvironmentClient

async with EnvironmentClient(base_url="https://aadiiityaa007-data-engineer-env.hf.space") as env:
    obs = await env.reset()
```

---

## Quickstart (remote usage)

### Prerequisites
- Python 3.11+
- `pip`

### Install
```bash
pip install openenv openai
```

### Minimal connectivity check
```python
import asyncio
from openenv.core.env_client import EnvironmentClient

REMOTE_URL = "https://aadiiityaa007-data-engineer-env.hf.space"

async def main():
    async with EnvironmentClient(base_url=REMOTE_URL) as env:
        obs = await env.reset()
        print(obs.observation.result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Submission inference script (`inference.py`)

This repository includes a submission-ready `inference.py` at project root.

### Required environment variables

- `API_BASE_URL`  
  Default: `https://router.huggingface.co/v1`
- `MODEL_NAME`  
  Default: `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN` (recommended) or `API_KEY`
- `ENV_BASE_URL`  
  Default: `https://aadiiityaa007-data-engineer-env.hf.space`

### Why `HF_TOKEN` is needed
If using Hugging Face Router as your LLM endpoint, authentication is required.  
Without token/key, model inference calls typically fail with authorization errors.

### Run
```bash
python inference.py
```

### Mandatory stdout contract

The script emits exactly these line types in this order:

1. `[START] ...` (once)
2. `[STEP] ...` (one per action)
3. `[END] ...` (always emitted, even on failure)

Example:
```text
[START] task=data-engineer env=sqlite model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=read_file({"filename":"users.json"}) reward=-0.01 done=false error=null
[END] success=true steps=30 score=1.00 rewards=-0.01,0.04,...
```

---

### 📂 Using Your Own Custom Datasets
By default, the cloud-hosted environment runs the standard "Escape Room" puzzle (`users.json`, `corrupted.json`). 

However, if you clone this repository to run locally, the environment is highly modular. You can inject your own virtual datasets (CSV, JSON, XML) directly into the environment to test how your AI handles different data engineering challenges!

Simply modify the initialization in `server/environment.py`:
```python
# You can pass ANY dictionary of virtual files into the sandbox!
my_custom_data = {
    "sales_data.csv": "id,amount,status\n1,500,SUCCESS\n2,1000,FAILED",
    "messy_logs.json": '[{"error": "null pointer"}, {"error": "timeout"}]'
}

# The environment will now serve your custom files to the AI Agent
env = DataEngineerEnv(custom_files=my_custom_data)

## Running locally (custom datasets)

The environment is intentionally modular. You can inject custom virtual files for your own benchmark scenarios.

Example (`server/environment.py` usage pattern):

```python
my_custom_data = {
    "sales_data.csv": "id,amount,status\n1,500,SUCCESS\n2,1000,FAILED",
    "events.json": '[{"event":"checkout","value":1200},{"event":"refund","value":-100}]'
}

env = DataEngineerEnv(custom_files=my_custom_data)
```

Use this to build domain-specific tasks (fintech, ecommerce, telemetry, logs, etc.).

---

## Integration testing

For deterministic verification (without relying on LLM output variability), include and run a “golden path” test script that executes known-correct action sequences.

Example:
```bash
python test_remote_full_game.py
```
This script acts as the "perfect player", passing all 3 levels, validating the SQLite sandbox, and proving the cloud server correctly tracks state and total rewards.

---
*Created by [@Aditya-myst](https://github.com/Aditya-myst) for the Meta x Hugging Face Hackathon.*
```