import mongoose from "mongoose";
import dotenv from 'dotenv';

dotenv.config()
const connectDB = async ()=>{
    try {
        await mongoose.connect(process.env.MONGO_URL,{
            useNewUrlParser: true,
            useUnifiedTopology: true,
        }).then(() => {
            console.log("Connected to MongoDB successfully!");
        }).catch((err)=>{
            console.error("Mongo Connection error: ",err.message)
            process.exit(1)
        });

    } catch (error) {
        console.error("Mongo Connection error",error.message)
        process.exit(1)
    }
}

export default connectDB