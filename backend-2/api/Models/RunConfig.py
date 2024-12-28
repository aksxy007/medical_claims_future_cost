from pydantic import BaseModel

class Config(BaseModel):
    config: dict  # The config is expected to be a dictionary
