import asyncio
from client import DataEngineerClient
from models import SQLAction

async def main():
    print("Testing the AI Escape Room...")
    
    # 1. Connect to the local web server
    async with DataEngineerClient(base_url="http://127.0.0.1:8000") as client:
        
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