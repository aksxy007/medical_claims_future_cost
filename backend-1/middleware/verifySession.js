import Session from "../models/Session.js";

const verifySession = async (req, res, next) => {
  const accessToken = req.headers.authorization?.split(" ")[1]; // Extract token from headers

  if (!accessToken) {
    return res.status(401).json({ success: false, message: "Access token missing" });
  }

  try {
    // Find session by access token
    const session = await Session.findOne({ accessToken });

    if (!session) {
      return res.status(401).json({ success: false, message: "Session expired or invalid" });
    }

    // Update lastActive timestamp for session
    session.lastActive = new Date();
    await session.save();

    // Attach userId to the request object
    req.user = session.userId;
    next();
  } catch (error) {
    console.error("Error in verifySession middleware", error);
    return res.status(500).json({ success: false, message: "Server error" });
  }
};

export default verifySession;
