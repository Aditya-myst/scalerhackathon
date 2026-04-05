import asyncio
from client import DataEngineerClient
from models import SQLAction

# Use your live Hugging Face Space URL
REMOTE_URL = "https://aadiiityaa007-data-engineer-env.hf.space"

async def main():
    async with DataEngineerClient(base_url=REMOTE_URL) as client:
        print("--- STARTING THE FULL GAME ON HUGGING FACE ---")
        obs = await client.reset()
        
        # We will keep a running total, just like the real judges do!
        total_score = 0.0

        print("\n--- LEVEL 1: EASY ---")
        obs = await client.step(SQLAction(command="execute_sql", parameters={"query": "CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)"}))
        total_score += obs.reward  # Add reward (+0.05 for good SQL)
        
        obs = await client.step(SQLAction(command="execute_sql", parameters={"query": "INSERT INTO users VALUES (1, 'Alice', 30), (2, 'Bob', 25)"}))
        total_score += obs.reward  # Add reward (+0.05 for good SQL)
        
        obs = await client.step(SQLAction(command="submit_task", parameters={"task": "easy"}))
        total_score += obs.reward  # Add reward (+0.3 for passing Easy)
        print("Grader Reply:", obs.observation.result)

        print("\n--- LEVEL 2: MEDIUM ---")
        obs = await client.step(SQLAction(command="execute_sql", parameters={"query": "CREATE TABLE orders (order_id INTEGER, user_id INTEGER, amount REAL)"}))
        total_score += obs.reward
        
        obs = await client.step(SQLAction(command="execute_sql", parameters={"query": "INSERT INTO orders VALUES (101, 1, 50.5), (102, 2, 12.0)"}))
        total_score += obs.reward
        
        obs = await client.step(SQLAction(command="submit_task", parameters={"task": "medium"}))
        total_score += obs.reward  # Add reward (+0.3 for passing Medium)
        print("Grader Reply:", obs.observation.result)

        print("\n--- LEVEL 3: HARD ---")
        obs = await client.step(SQLAction(command="execute_sql", parameters={"query": "INSERT INTO users VALUES (4, 'Dave', 40)"}))
        total_score += obs.reward
        
        obs = await client.step(SQLAction(command="submit_task", parameters={"task": "hard"}))
        total_score += obs.reward  # Add reward (+0.4 for passing Hard)
        print("Grader Reply:", obs.observation.result)
        
        print("\n--- FINAL RESULTS ---")
        print(f"Final Total Reward (Cumulative Score): {total_score:.2f}")
        print("Did AI Win? (Done):", obs.done)

if __name__ == "__main__":
    asyncio.run(main())