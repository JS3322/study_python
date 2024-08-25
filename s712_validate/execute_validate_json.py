import json
import jsonschema
from jsonschema import validate

# JSON Schema
schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "User Project Schema",
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for the user",
      "minLength": 1
    },
    "pw": {
      "type": "string",
      "description": "Password for the user",
      "minLength": 8
    },
    "projectName": {
      "type": "string",
      "description": "Name of the project",
      "minLength": 1
    },
    "githubUrl": {
      "type": "string",
      "description": "GitHub repository URL",
      "format": "uri"
    },
    "hyperparameters": {
      "type": "object",
      "description": "Hyperparameters for the machine learning model",
      "additionalProperties": {
        "type": ["number", "string"]
      }
    },
    "dataPath": {
      "type": "string",
      "description": "File path to the training data",
      "minLength": 1
    }
  },
  "required": ["id", "pw", "projectName", "githubUrl", "dataPath"],
  "additionalProperties": false
}

# Example JSON data
data = {
  "id": "user123",
  "pw": "securePassword123",
  "projectName": "MyMLProject",
  "githubUrl": "https://github.com/user123/mymlproject",
  "hyperparameters": {
    "learning_rate": 0.01,
    "batch_size": 32
  },
  "dataPath": "/data/mymlproject/dataset.csv"
}

# Validate JSON data
try:
  validate(instance=data, schema=schema)
  print("JSON data is valid.")
except jsonschema.exceptions.ValidationError as err:
  print("JSON data is invalid:", err.message)
