import bodyParser from "body-parser";
import express from "express";
import connectDB from "./db/mongo_connection.js";
import AuthRouter from "./routes/auth.js";
import session from "express-session";
import cookieParser from "cookie-parser";
import MongoStore from "connect-mongo";
import cors from "cors";
import mongoose from "mongoose";

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

// Add routes
app.use("/auth", AuthRouter);

// Start the server
app.listen(PORT, () => {
  console.log(`Server started at PORT: ${PORT}`);
});
