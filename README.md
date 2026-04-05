```markdown
---
title: Data Engineer Env Environment Server
emoji: 🗄️
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
---

# 🚀 AI Data Engineer Escape Room (OpenEnv)
**Built for the Meta x Hugging Face OpenEnv Hackathon**

[![Hosted on Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hosted%20on-Hugging%20Face-blue)](https://huggingface.co/spaces/aadiiityaa007/data_engineer_env)
[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-green)](#)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](#)

## 📖 Overview
The **Data Engineer Escape Room** is a Reinforcement Learning (RL) environment designed to evaluate and train autonomous AI agents in real-world data engineering tasks. 

Instead of simple text games, this environment provides a secure, in-memory **SQLite sandbox** and virtual file system. The agent must autonomously read raw data (JSON), construct relational databases, and clean corrupted datasets (e.g., negative ages, missing fields) using raw SQL commands.

## 🎯 Hackathon Criteria Met
1. **Built on OpenEnv:** Uses `openenv.core.env_server` for standardized WebSocket communication.
2. **Hugging Face Integration:** Fully dockerized and deployed live on Hugging Face Spaces.
3. **Strict Graders:** Tasks aren't graded by LLM vibes; they are graded by hidden SQL queries executing against the agent's database to verify exact row counts and data integrity.
4. **Resource-Optimized RL (Energy Penalties):** The agent is penalized `-0.01` reward points for every action (compute cycle) it takes, forcing it to write efficient SQL pipelines rather than blindly guessing or spamming the server.

---

## ⚙️ Environment Mechanics

### 1. The Action Space
The agent interacts with the environment by sending JSON payloads with one of three commands:
* **`read_file`**: Inspects virtual files.
  * *Example:* `{"command": "read_file", "parameters": {"filename": "users.json"}}`
* **`execute_sql`**: Runs SQL commands against the SQLite sandbox.
  * *Example:* `{"command": "execute_sql", "parameters": {"query": "CREATE TABLE users (id INTEGER, age INTEGER)"}}`
* **`submit_task`**: Triggers the Grader to evaluate the current database state.
  * *Example:* `{"command": "submit_task", "parameters": {"task": "easy"}}`

### 2. The Reward Function
This environment utilizes a strict reward system to train agents:
* 🔴 **-0.01**: Compute Penalty (Applied to *every* action taken).
* 🔴 **-0.05**: Syntax Penalty (Invalid SQL, missing files, or system errors).
* 🟢 **+0.05**: Successful DML Execution (Successfully altering data, e.g., `INSERT` or `CREATE`).
* 🟢 **+0.30 to +0.40**: Level Completion (Passing the strict SQL Grader).

### 3. The Levels
1. **Easy:** Create a table and insert raw data.
2. **Medium:** Create relational tables using `FOREIGN KEY` constraints.
3. **Hard (Data Cleaning):** Read corrupted data and write an `INSERT` statement that dynamically filters out invalid constraints (e.g., ages < 0) before insertion.

---

## 🛠️ How to Connect Your Own AI Agent

This environment is live and accepts remote WebSocket connections. You can attach *any* LLM (OpenAI, Anthropic, Gemini, Llama) to it.

### 1. Install Dependencies
```bash
pip install openenv
```

### 2. Python Client Example
```python
import asyncio
from openenv.core.env_client import EnvironmentClient
from models import SQLAction

REMOTE_URL = "https://aadiiityaa007-data-engineer-env.hf.space"

async def main():
    # Connect to the live Hugging Face Environment
    async with EnvironmentClient(base_url=REMOTE_URL) as client:
        
        # 1. Reset environment & get the first prompt
        obs = await client.reset()
        print("System:", obs.observation.result)
        
        # 2. Take an action
        action = SQLAction(command="read_file", parameters={"filename": "users.json"})
        result = await client.step(action)
        
        print("Result:", result.observation.result)
        print("Reward Earned:", result.reward)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🤖 Running the Built-In Autonomous Agent
This repository includes a pre-built autonomous agent powered by **Google Gemini**. It demonstrates how an LLM can navigate the environment, fix its own SQL errors, and beat all 3 levels.

1. Clone the repo.
2. Install requirements: `pip install requests openenv`
3. Add your Gemini API key inside `autonomous_agent.py`.
4. Run the agent:
```bash
python autonomous_agent.py
```
Watch the terminal as the AI reads the data, writes the SQL, and escapes the room!

---
*Created by [@codeviaditya](https://github.com/Aditya-myst) for the Meta x Hugging Face Hackathon.*
```
