from services.RabbitMQService import RabbitMQService
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

def consume_messages(rabbitmq_service,queue_name,helper_function):
    channel = rabbitmq_service.get_channel()

    async def process_message(message):
        try:
            await helper_function(message)
        except Exception as e:
            print(f"Error processing message: {e}")

    def callback(ch, method, properties, body):
        print(f"Received message: {body.decode()}")
        # Process the message here (e.g., trigger a pipeline run)
        message = body.decode()
        loop = asyncio.new_event_loop()  # Create a new event loop for the thread
        asyncio.set_event_loop(loop)  # Set it as the current event loop for this thread

        # Run the async task on the newly created event loop
        loop.run_until_complete(process_message(message))

        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(queue=queue_name, on_message_callback=callback)
    print(f"Waiting for messages in queue: {queue_name}")
    try:
        # Start consuming messages
        channel.start_consuming()
    except Exception as e:
        print(f"Error consuming messages: {e}")
        channel.stop_consuming()
    

