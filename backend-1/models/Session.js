import mongoose from "mongoose";

const sessionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  accessToken: { type: String, required: true },
  refreshToken: { type: String, required: true },
  lastActive: { type: Date, default: Date.now }, // Track last active time
  createdAt: { type: Date, default: Date.now, expires: "1h" }, // Auto-delete after 7 days
});

const Session = mongoose.model("Session", sessionSchema);

export default Session;