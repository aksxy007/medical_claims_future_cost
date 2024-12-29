# app.py
from fastapi import FastAPI
from endpoints import trigger
from fastapi.middleware.cors import CORSMiddleware
import os
from services.RabbitMQService import RabbitMQService
from services.RabbitMQConsumer import consume_messages
# Initialize the FastAPI app
from utils.trigger_airflow_pipeline import trigger_airflow_pipeline
import threading
app = FastAPI()

rabbitmq_service = RabbitMQService()

app.add_middleware(CORSMiddleware,
    allow_origins=['http://localhost:8000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def start_rabbitmq_consumer():
    """Start the RabbitMQ consumer in a separate thread."""
    try:
        rabbitmq_service.connect()
        rabbitmq_service.create_queue(os.getenv("RABBITMQ_QUEUE_NAME"))
        print("RabbitMQ connected and queue created")
        
        # Start consuming messages continuously
        consume_messages(
            rabbitmq_service=rabbitmq_service,
            queue_name=os.getenv("RABBITMQ_QUEUE_NAME"),
            helper_function=trigger_airflow_pipeline,
        )
    except Exception as e:
        print(f"Error in RabbitMQ consumer: {e}")

@app.on_event("startup")
def start_event():
    try:
        """Start RabbitMQ consumer when the application starts."""
        threading.Thread(target=start_rabbitmq_consumer, daemon=True).start()
        print("RabbitMQ consumer started in a background thread")
    except Exception as e:
        print("Error",e)
    
    
@app.on_event("shutdown")
def shutdown_event():
    # Close RabbitMQ connection
    if rabbitmq_service.connection:
        rabbitmq_service.connection.close()
        print("RabbitMQ connection closed")
# Include the trigger route
app.include_router(trigger.router, prefix="/trigger", tags=["trigger"])