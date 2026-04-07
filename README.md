# AI Data Engineer Escape Room (OpenEnv)
**Meta × Hugging Face OpenEnv Hackathon Submission**

[![Hosted on Hugging Face Spaces](https://img.shields.io/badge/Hosted%20on-Hugging%20Face-blue)](https://huggingface.co/spaces/aadiiityaa007/data_engineer_env)
[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green)](#)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](#)

---

## 📖 Overview

**AI Data Engineer Escape Room** is a deterministic RL-style evaluation environment for autonomous data-engineering agents.

Agents operate inside:
- a secure in-memory **SQLite** sandbox,
- a virtual file system (`users.json`, `orders.json`, `corrupted.json`),
- and a staged task flow (`easy`, `medium`, `hard`).

Agents can only use:
1. `read_file`
2. `execute_sql`
3. `submit_task`

Scoring is deterministic and based on hidden SQL validators against actual DB state (not subjective LLM grading).

---

## 🔗 Live Links

- **GitHub Repo:** https://github.com/Aditya-myst/scalerhackathon  
- **HF Space:** https://huggingface.co/spaces/aadiiityaa007/data_engineer_env  
- **Environment Base URL:** `https://aadiiityaa007-data-engineer-env.hf.space`

---

## ✅ Hackathon Requirement Mapping

1. **Built on OpenEnv**  
   Uses OpenEnv-compatible server/client abstractions.

2. **Deployed on Hugging Face Spaces**  
   Containerized and hosted with a public endpoint.

3. **Strict deterministic grading**  
   Hidden SQL checks validate real DB state and constraints.

4. **Reward shaping**  
   Includes per-step penalty and completion rewards.

5. **Autonomous inference path**  
   Includes `inference.py` in repository root.

---

## 🧠 Environment Mechanics

### Action Space

- `read_file`
  ```json
  {"command":"read_file","parameters":{"filename":"users.json"}}
  ```

- `execute_sql`
  ```json
  {"command":"execute_sql","parameters":{"query":"CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)"}}
  ```

- `submit_task`
  ```json
  {"command":"submit_task","parameters":{"task":"easy"}}
  ```

### Reward Model

Typical shaping:
- `-0.01` per step,
- negative reward for invalid actions/SQL errors,
- positive reward for correct progress,
- larger reward on validated task completion.

### Levels

- **Easy:** create/load `users`
- **Medium:** create/load `orders` + consistency
- **Hard:** clean corrupted data with strict constraints

---

## 🌐 Endpoint and Connectivity

Environment base URL:
`https://aadiiityaa007-data-engineer-env.hf.space`

Example client connection:
```python
from openenv.core.env_client import EnvironmentClient

async with EnvironmentClient(base_url="https://aadiiityaa007-data-engineer-env.hf.space") as env:
    obs = await env.reset()
```

---

## 🚀 Quickstart (Remote Usage)

### Prerequisites
- Python 3.11+
- pip

### Install
```bash
pip install openenv openai
```

### Run submission script
```bash
python inference.py
```

---

## 🧾 Submission-Critical `inference.py` Notes

### Required environment variables (for evaluator)

- `API_BASE_URL`
- `MODEL_NAME`
- `API_KEY`

### Required LLM initialization pattern

```python
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["API_KEY"]

from openai import AsyncOpenAI
llm = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

> Important:
> - Do not hardcode API keys.
> - Do not use alternate provider auth paths for judged inference.
> - Proxy traffic must go through injected `API_BASE_URL` + `API_KEY`.

### Optional environment variables

- `ENV_BASE_URL` (defaults to this HF Space endpoint)
- `MY_ENV_TASK` (default: `data-engineer`)
- `MY_ENV_BENCHMARK` (default: `sqlite`)
- `MAX_STEPS`, `TEMPERATURE`, `MAX_TOKENS`, `SCORE_DENOM`

---

## 🖨️ Mandatory Stdout Contract

`inference.py` emits:

1. `[START]` exactly once  
2. `[STEP]` once per environment step  
3. `[END]` exactly once (always emitted)

Example:
```text
[START] task=data-engineer env=sqlite model=<provided-model-name>
[STEP] step=1 action=read_file({"filename":"users.json"}) reward=-0.01 done=false error=null
...
[END] success=true steps=15 score=1.00 rewards=-0.01,0.04,...
```

Rules:
- `reward` values formatted to 2 decimals
- `done` and `success` are lowercase booleans
- `error` is raw error string or `null`
- score is clamped to `[0,1]`

---

## 🧪 Integration Testing

Golden-path test:
```bash
python test_remote_full_game.py
```

If available in your local checkout:
```bash
python test_remote.py
```

---

## 🧩 Custom Datasets (Local Development)

You can run local custom scenarios by injecting virtual files:

```python
my_custom_data = {
    "sales_data.csv": "id,amount,status\n1,500,SUCCESS\n2,1000,FAILED",
    "events.json": '[{"event":"checkout","value":1200},{"event":"refund","value":-100}]'
}

env = DataEngineerEnv(custom_files=my_custom_data)
```

Useful for domain-specific benchmarks (fintech, ecommerce, telemetry, logs).

---

## 📂 Project Structure

```text
.
├── inference.py
├── autonomous_agent.py
├── client.py
├── models.py
├── server/
│   ├── app.py
│   └── environment.py
├── test_remote.py
├── test_remote_full_game.py
└── README.md
```

---

## 👤 Author

Created by **[@Aditya-myst](https://github.com/Aditya-myst)** for the Meta × Hugging Face OpenEnv Hackathon.