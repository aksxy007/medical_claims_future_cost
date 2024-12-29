import bodyParser from "body-parser";
import express from "express";
import connectDB from "./db/mongo_connection.js";
import AuthRouter from "./routes/auth.js";
import DefaultConfigRouter from "./routes/configRoute.js"
import ProjectRouter from "./routes/projects.js"
import AirlfowRouter from "./routes/airflowRoutes.js"
import session from "express-session";
import cookieParser from "cookie-parser";
import MongoStore from "connect-mongo";
import cors from "cors";
import mongoose from "mongoose";
import rabbitMQService from "./services/rabbitMQservice.js";

const app = express();
const PORT = process.env.PORT || 8000;

// Middlewares
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cookieParser());

// CORS setup
app.use(
  cors({
    origin: ["http://localhost:3000"], // Adjust the allowed origin
    credentials: true,
  })
);



// Connect to MongoDB
connectDB();

// Session Management
app.use(
  session({
    secret: "your_session_secret", // Session secret
    store: MongoStore.create({
      client:mongoose.connection.getClient(), // Use the existing MongoDB client from mongoose
      collectionName: "sessions", // Custom collection for storing sessions
      ttl: 60 * 60 * 1000, // Session expiration time in seconds (1 hour)
      touchAfter: 24 * 3600, // Delay the session update if there are no changes
      autoRemove: "native", // Auto-remove expired sessions from MongoDB
    }),
    resave: false, // Do not force re-save the session if unmodified
    saveUninitialized: true, // Store session even if it is not modified
    cookie: {
      secure: process.env.NODE_ENV === "production", // Secure cookie in production
      maxAge: 1000 * 60* 60  , // Cookie expiration (1 hour)
    },
  })
);

const startApp = async () => {
  try {
    await rabbitMQService.connect();  // Connect to RabbitMQ when the app starts
    await rabbitMQService.createQueue(process.env.RABBITMQ_QUEUE_NAME);  // Create the necessary queues
    console.log('App is starting...');
  } catch (error) {
    console.error('Error starting the app:', error);
    process.exit(1);  // Exit if there is an issue with connecting to RabbitMQ
  }
};

startApp()

// Add routes
app.use("/auth", AuthRouter);
app.use("/config", DefaultConfigRouter);
app.use("/projects", ProjectRouter);
app.use("/run", AirlfowRouter);

// Start the server
app.listen(PORT, () => {
  console.log(`Server started at PORT: ${PORT}`);
});
