# app.py
from fastapi import FastAPI
from endpoints import trigger

# Initialize the FastAPI app
app = FastAPI()

# Include the trigger route
app.include_router(trigger.router, prefix="/trigger", tags=["trigger"])