import rabbitMQService from "./rabbitMQservice.js";

class ProducerService {
  async sendMessage(queue, message) {
    try {
      const channel = rabbitMQService.getChannel();
      if (!channel) throw new Error('RabbitMQ channel is not initialized');
      channel.sendToQueue(queue, Buffer.from(JSON.stringify(message)));
      console.log(`Message sent to queue ${queue}:`, message);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  }
}

export default ProducerService;

