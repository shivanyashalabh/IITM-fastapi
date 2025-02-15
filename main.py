from fastapi import FastAPI, HTTPException, Query
from task_handlers import execute_function  # Ensure this import is correct
from utils import read_file
from fastapi.responses import PlainTextResponse
app = FastAPI()

@app.post("/run")
def run(task: str = Query(...)):
    if not task:
        raise HTTPException(status_code=400, detail="Task description missing")

    try:
        # Pass the task to execute_function to determine intent and execute the task
        result = execute_function(task)
        return {"status": "success", "message": "Task executed successfully", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/read",response_class=PlainTextResponse)
def read(path: str = Query(...)):
    print(path)
    """Returns the content of a specified file."""
    if not path or not path.startswith("/data/"):
        print("path")
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        print(path)
        content = read_file(path)
        return content
    except FileNotFoundError:
        raise HTTPException(status_code=200, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
