import json
import sqlite3
from openenv.core.env_server import Environment
from models import SQLAction, SQLObservation, SQLState
from openenv.core.env_server.types import State
import uuid

class DataEngineerEnv(Environment):
    # We now allow anyone to pass their own custom data into the environment!
    def __init__(self, custom_files=None):
        super().__init__()
        self._state = SQLState()
        self.db = None
        self.reward = 0.0
        self.done = False
        
        # Default data if none is provided, but now it can be swapped!
        self.files = custom_files or {
            "users.json": json.dumps([
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25}
            ]),
            "orders.json": json.dumps([
                {"order_id": 101, "user_id": 1, "amount": 50.5},
                {"order_id": 102, "user_id": 2, "amount": 12.0}
            ]),
            "corrupted.json": json.dumps([
                {"id": 3, "name": "Charlie", "age": -5},
                {"id": 4, "name": "Dave", "age": 40}
            ])
        }

        # DYNAMIC GRADER MATH: Calculate exactly how many rows SHOULD exist
        self.expected_users = len(json.loads(self.files["users.json"]))
        self.expected_orders = len(json.loads(self.files["orders.json"]))
        
        # Calculate how many valid rows are in the corrupted file (age >= 0)
        corrupted_list = json.loads(self.files["corrupted.json"])
        valid_corrupted_rows = len([row for row in corrupted_list if row.get("age", 0) >= 0])
        self.expected_final_users = self.expected_users + valid_corrupted_rows

    def reset(self) -> SQLObservation:
        if self.db:
            self.db.close()
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        
        self._state = SQLState()
        self.reward = 0.0
        self.done = False

        welcome_msg = (
            "SYSTEM: Data Engineering Pipeline Initialized.\n"
            "ENGINE: SQLite3 (In-Memory)\n"
            "TASK 1 [EASY]: Read 'users.json'. Create table 'users' (id INTEGER, name TEXT, age INTEGER). Insert data.\n"
            "COMMAND: When verified, execute command 'submit_task' with parameters {'task': 'easy'}."
        )
        return SQLObservation(result=welcome_msg, success=True, reward=self.reward, done=self.done)

    def step(self, action: SQLAction) -> SQLObservation:
        self.reward = -0.01
        self._state.step_count += 1
        cmd = action.command.lower()
        params = action.parameters

        try:
            if cmd == "read_file":
                filename = params.get("filename")
                if filename not in self.files:
                    return SQLObservation(result=f"Error: File '{filename}' does not exist.", success=False, reward=self.reward, done=self.done)
                return SQLObservation(result=self.files[filename], success=True, reward=self.reward, done=self.done)

            elif cmd == "execute_sql":
                query = params.get("query", "")
                cursor = self.db.cursor()
                cursor.execute(query)
                self.db.commit()
                
                
                query_upper = query.strip().upper()
                if query_upper.startswith("SELECT") or query_upper.startswith("PRAGMA"):
                    rows = [dict(row) for row in cursor.fetchall()]
                    return SQLObservation(result=json.dumps(rows), success=True, reward=self.reward, done=self.done)
                else:
                    self.reward += 0.05 
                    return SQLObservation(result=f"SQL executed successfully. Rows affected: {cursor.rowcount}", success=True, reward=self.reward, done=self.done)

            elif cmd == "submit_task":
                return self._grade_task(params.get("task", "easy"))

            else:
                return SQLObservation(result=f"Error: Unknown command '{cmd}'.", success=False, reward=self.reward, done=self.done)

        except sqlite3.Error as e:
            self.reward -= 0.05
            return SQLObservation(result=f"SQLite Error: {str(e)}", success=False, reward=self.reward, done=self.done)
        except Exception as e:
            self.reward -= 0.05
            return SQLObservation(result=f"System Error: {str(e)}", success=False, reward=self.reward, done=self.done)

    def _grade_task(self, task: str) -> SQLObservation:
        cursor = self.db.cursor()
        try:
            if task == "easy":
                cursor.execute("SELECT COUNT(*) as cnt FROM users")
                # DYNAMIC CHECK: Does it match the exact length of the JSON file?
                if cursor.fetchone()["cnt"] == self.expected_users:
                    self.reward += 0.3
                    self._state.current_level = "medium"
                    return SQLObservation(result="VALIDATION PASSED: Task 'easy' complete.\nTASK 2 [MEDIUM]: Read 'orders.json'. Create table 'orders' with FOREIGN KEY user_id referencing users(id). Insert data. Submit task 'medium'.", success=True, reward=self.reward, done=self.done)
                return SQLObservation(result=f"VALIDATION FAILED: Table 'users' should have {self.expected_users} rows.", success=False, reward=self.reward, done=self.done)
                    
            elif task == "medium":
                cursor.execute("SELECT COUNT(*) as cnt FROM orders")
                # DYNAMIC CHECK: Does it match the exact length of the orders file?
                if cursor.fetchone()["cnt"] == self.expected_orders:
                    self.reward += 0.3
                    self._state.current_level = "hard"
                    return SQLObservation(result="VALIDATION PASSED: Task 'medium' complete.\nTASK 3 [HARD]: Read 'corrupted.json'. Insert valid records into 'users'. Data cleaning required: IGNORE records with negative ages. Submit task 'hard'.", success=True, reward=self.reward, done=self.done)
                return SQLObservation(result=f"VALIDATION FAILED: Table 'orders' should have {self.expected_orders} rows.", success=False, reward=self.reward, done=self.done)
                    
            elif task == "hard":
                # DYNAMIC CHECK 1: Ensure absolutely ZERO negative ages made it into the database
                cursor.execute("SELECT COUNT(*) as cnt FROM users WHERE age < 0")
                corrupted_inserted = cursor.fetchone()["cnt"] > 0
                
                # DYNAMIC CHECK 2: Ensure the total row count is exactly correct
                cursor.execute("SELECT COUNT(*) as cnt FROM users")
                current_total_users = cursor.fetchone()["cnt"]
                
                if not corrupted_inserted and current_total_users == self.expected_final_users:
                    self.reward += 0.4
                    self.done = True
                    return SQLObservation(result="VALIDATION PASSED: Task 'hard' complete. Pipeline successful. ALL TASKS COMPLETED.", success=True, reward=self.reward, done=self.done)
                return SQLObservation(result=f"VALIDATION FAILED: Data cleaning constraints violated. Found corrupted data OR total row count is not {self.expected_final_users}.", success=False, reward=self.reward, done=self.done)
                    
        except sqlite3.OperationalError as e:
            return SQLObservation(result=f"VALIDATION ERROR: Could not query tables. SQLite Error: {str(e)}", success=False, reward=self.reward, done=self.done)

    @property
    def state(self) -> SQLState:
        # OpenEnv will automatically give us an episode_id and step_count
        if not hasattr(self, '_state_obj'):
            from openenv.core.env_server.types import State
            import uuid
            self._state_obj = State(episode_id=str(uuid.uuid4()), step_count=0)
        return self._state_obj