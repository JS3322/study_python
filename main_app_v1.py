from fastapi import FastAPI
from fastapi.exceptions import RequestVallidationError

from app.api.api_v1.endpoints import data
import uvicorn
from app.error_handlers import (
	db_connection_error_handler,
    token_validation_error_handler,
    missing_token_error_handler,
    data_fetch_error_handler,
    validation_exception_handler
)
from app.exception.exceptions import (
	DatabaseConnectionError,
    TOkenValidationError,
    MissingTokenError,
    DataFetchError
)

app = FastAPI()

# register error handlers
app.add_exception_handler(DataConnectionError, db_connection_error_handler)
app.add_exception_handler(TokenValidationError, token_validation_error_handler)
# ...

app.include_route(data.router, prefix="/v1")

if __name__ = "__main__":
    print('hello world cleancode')
    uvicorn.run(app, host="0.0.0.0", port=8000)