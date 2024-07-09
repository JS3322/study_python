from fastapi import FastAPI, Request
from src.analysis.application import AnalysisCleancodeApplication
from src.ai.application import DeepLearningExampleApplication
import uvicorn

from s615_aop.service.Logging_Example import log_activity

app = FastAPI()

app.include_router(AnalysisCleancodeApplication.router, prefix="/v1/read")
app.include_router(DeepLearningExampleApplication.router, prefix="/v1/execute")

@app.get("/items/{item_id}")
@log_activity
async def read_item(item_id: int, request: Request):
  """Endpoint to read an item by its ID."""
  return {"item_id": item_id, "info": "Details about item"}

@app.post("/items/")
@log_activity
async def create_item(item: dict, request: Request):
  """Endpoint to create a new item."""
  return {"item_id": 123, "info": "Item created successfully"}

# The server should be run in a proper environment, not in this code cell
if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
