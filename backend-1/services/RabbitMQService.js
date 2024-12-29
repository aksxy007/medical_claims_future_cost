import amqp from "amqplib"

class RabbitMQService{
    constructor(){
        this.connection = null,
        this.channel = null
        this.username = process.env.RABBITMQ_USERNAME
        this.password = process.env.RABBITMQ_PASSWORD
    }

    async connect(){
        try {
            this.connection = await amqp.connect(`amqp://${this.username}:${this.password}@localhost`)
            this.channel = await this.connection.createChannel();
            console.log('RabbitMQ connected in Backend 1')
        } catch (error) {
            console.log('RabbitMQ connection error:', error)
        }
    }

    async createQueue(queueName){
        if(!this.channel) throw new Error("Channel not initialized")
        await this.channel.assertQueue(queueName,{durable:true})
    }

    async closeConnection() {
        try {
          if (this.channel) {
            await channel.close();
            console.log('Channel closed');
          }
          if (this.connection) {
            await connection.close();
            console.log('Connection closed');
          }
        } catch (error) {
          console.error('Error closing connection:', error);
        }
    }

    getChannel() {
        return this.channel;
    }
}

const rabbitMQService = new RabbitMQService()

export default rabbitMQService;