import bodyParser from "body-parser";
import express from "express";
import connectDB from "./db/mongo_connection.js";
import AuthRouter from "./routes/auth.js";
import session from "express-session";
import cookieParser from 'cookie-parser'; 
import MongoStore from 'connect-mongo';
import cors from 'cors';


const app = express()
const PORT = process.env.PORT || 8000;


//  Middlewares
app.use(bodyParser.urlencoded({extended:true}))
app.use(bodyParser.json())
app.use(cookieParser())


app.use(cors({
  origin:["http://localhost:3000"],
  credentials:true
}))

// Session Management
app.use(session({
    secret: 'your_session_secret', 
    store: MongoStore.create({
        mongoUrl:process.env.MONGO_URL,
        collectionName:"sessions",
        ttl: 24 * 60 * 60,
    }),
    resave: false,
    saveUninitialized: true,  // Store session even if the session is not modified
    cookie: { 
      secure: process.env.NODE_ENV === 'production',  // Secure cookies in production
      maxAge: 1000 * 60 * 60 * 24, // 1 day expiration
    }
  }));

// Add routes
app.use("/auth",AuthRouter)

// Connect to MongoDB
connectDB()

app.listen(PORT,()=>{
    console.log(`Server started at PORT: ${PORT}`)
})