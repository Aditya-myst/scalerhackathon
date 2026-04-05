import asyncio
import json
import requests
from client import DataEngineerClient
from models import SQLAction

# ==========================================
# 1. SETUP YOUR GEMINI API KEY HERE
# ==========================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

# Your live Hugging Face Environment
REMOTE_URL = "https://aadiiityaa007-data-engineer-env.hf.space"

# ==========================================
# 2. DIRECT API FUNCTION (Bypasses Google's broken SDK)
# ==========================================
def ask_gemini(prompt_text, history):
        # Replace the model name below with the exact one from your script!
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent?key={GEMINI_API_KEY}"
    
    # We combine the system prompt + history + current prompt
    full_prompt = history + "\n\nCURRENT OBSERVATION:\n" + prompt_text
    
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}]
    }
    
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        raise Exception(f"Gemini API Error: {response.text}")
        
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ==========================================
# 3. THE SYSTEM PROMPT
# ==========================================
SYSTEM_PROMPT = """
You are an autonomous AI Data Engineer Agent participating in an OpenEnv Hackathon.
Your goal is to solve a Data Engineering Escape Room by completing 3 levels (Easy, Medium, Hard).

You must interact with a SQLite database and files. Every action costs compute energy (-0.01 reward).
You can ONLY output a SINGLE valid JSON object. NO markdown formatting. NO conversational text.
ONLY raw JSON.

Available Commands:
1. To read a file: {"command": "read_file", "parameters": {"filename": "users.json"}}
2. To run SQL: {"command": "execute_sql", "parameters": {"query": "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)"}}
3. To submit: {"command": "submit_task", "parameters": {"task": "easy"}} (or "medium", "hard")

Strategy:
- Read the file mentioned in the task first so you know the exact data format.
- Execute SQL to create the table and INSERT the exact data you just read.
- If a task mentions "Data cleaning" (like ignoring negative ages), you MUST write your INSERT statement to skip those corrupted rows.
- Once you are confident the table matches the data perfectly, submit the task.
"""

async def main():
    print("🤖 Waking up the Autonomous AI Agent (REST API Mode)...")
    
    # We will build our own memory!
    conversation_history = SYSTEM_PROMPT

    async with DataEngineerClient(base_url=REMOTE_URL) as client:
        result = await client.reset()
        observation = result.observation.result
        
        print("\n🎮 [GAME STARTED]")
        print(f"Server: {observation}")
        
        step_count = 1
        total_score = 0.0

        while not result.done:
            print(f"\n--- Step {step_count} ---")
            
            prompt = f"{observation}\nWhat is your next action? (Reply in pure JSON only)"
            
            try:
                # 1. Ask Gemini directly via REST
                ai_text = ask_gemini(prompt, conversation_history)
                
                # Clean up markdown
                if ai_text.startswith("```json"): ai_text = ai_text[7:]
                if ai_text.startswith("```"): ai_text = ai_text[3:]
                if ai_text.endswith("```"): ai_text = ai_text[:-3]
                ai_text = ai_text.strip()
                
                action_data = json.loads(ai_text)
                command = action_data.get("command")
                parameters = action_data.get("parameters", {})
                
                print(f"🚀 AI Action: {command} -> {parameters}")
                
                # 2. Send to Hugging Face Server
                action = SQLAction(command=command, parameters=parameters)
                result = await client.step(action)
                
                observation = result.observation.result
                total_score += result.reward
                
                # Add successful action to memory so the AI doesn't repeat itself
                conversation_history += f"\n\nObservation: {observation}\nMy Action: {ai_text}"
                
                print(f"📊 Server Reply: {observation}")
                print(f"⭐ Reward: {result.reward:.2f} | Total Score: {total_score:.2f}")

            except json.JSONDecodeError:
                print(f"❌ AI output invalid JSON: {ai_text}")
                observation = "SYSTEM ERROR: You did not return valid JSON. Try again. Output ONLY pure JSON, no backticks."
            except Exception as e:
                print(f"❌ System Error: {e}")
                observation = f"SYSTEM ERROR: {e}"

            step_count += 1
            if step_count > 30:
                print("\n🛑 AI got stuck. Terminating.")
                break

        if result.done:
            print("\n🏆 THE AI BEAT THE ESCAPE ROOM! 🏆")
            print(f"Final Score: {total_score:.2f} in {step_count - 1} steps.")
        else:
            print("\n💀 THE AI FAILED TO BEAT THE ROOM.")

if __name__ == "__main__":
    asyncio.run(main())