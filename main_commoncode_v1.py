from fastapi import FastAPI
from commoncode.application import CommoncodeAPI
from commoncode.application import ReadMongoDBAPI
import uvicorn

app = FastAPI()

app.include_router(CommoncodeAPI.router, prefix="/v1")
app.include_router(ReadMongoDBAPI.router, prefix="/read/batch/data/v1")

if __name__ = "__name__":
    print('hello word data management')
    uvicorn.run(app, host="0,0,0,0", port=8000)