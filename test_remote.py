import asyncio
from client import DataEngineerClient
from models import SQLAction

# Use your live Hugging Face Space URL
REMOTE_URL = "https://aadiiityaa007-data-engineer-env.hf.space"

async def main():
    print(f"Testing the AI Escape Room on {REMOTE_URL}...")
    
    # 1. Connect to the remote server
    async with DataEngineerClient(base_url=REMOTE_URL) as client:
        
        # 2. Reset the environment (Start the game)
        result = await client.reset()
        print("Observation from Reset:", result.observation.result)
        
        # 3. Take an action (Act like the AI)
        action = SQLAction(command="read_file", parameters={"filename": "users.json"})
        result = await client.step(action)
        
        print("\n--- The AI did something! ---")
        print("Observation from Step:", result.observation.result)
        print("Reward:", result.reward)
        print("Done:", result.done)

if __name__ == "__main__":
    asyncio.run(main())