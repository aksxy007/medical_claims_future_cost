import pika
import os
from dotenv import load_dotenv

load_dotenv()
class RabbitMQService:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.username = os.getenv("RABBITMQ_USERNAME")
        self.password =os.getenv("RABBITMQ_PASSWORD")

    def connect(self):
        try:
            
            if not self.username or not self.password:
                raise ValueError("RabbitMQ username or password is missing from environment variables.")
            
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host='localhost', credentials=pika.PlainCredentials(self.username,self.password))
            )
            self.channel = self.connection.channel()
            print("RabbitMQ connected in Backend 2")
        except Exception as e:
            print(f"Error connecting to RabbitMQ: {e}")

    def create_queue(self, queue_name):
        if not self.channel:
            raise Exception("Channel not initialized")
        self.channel.queue_declare(queue=queue_name, durable=True)

    def get_channel(self):
        return self.channel
