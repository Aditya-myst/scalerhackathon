from openenv.core.env_server import create_fastapi_app
from models import SQLAction, SQLObservation
from server.environment import DataEngineerEnv

# This creates the live web server!
env = DataEngineerEnv
app = create_fastapi_app(env, SQLAction, SQLObservation)

# Add a fake /web endpoint to satisfy Hugging Face's health checker
@app.get("/web")
def huggingface_health_check():
    return {"status": "ok", "message": "Health check passed!"}

# Also add a root endpoint just in case
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Server is running!"}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()