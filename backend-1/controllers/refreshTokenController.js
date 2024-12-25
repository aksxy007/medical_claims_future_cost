// controllers/refreshToken.js
import User from '../models/Users.js';
import { verifyRefreshToken } from '../utils/token.js';
import { generateAccessToken } from '../utils/token.js';

export const refreshTokenController = async (req, res) => {
  const { refreshToken } = req.cookies;  // Retrieve refresh token from cookie

  if (!refreshToken) {
    return res.status(400).json({ success: false, message: 'No refresh token provided' });
  }

  try {
    // Verify refresh token
    console.log("Refresh Token Called")
    const decoded = verifyRefreshToken(refreshToken);
    if (!decoded) {
      return res.status(403).json({ success: false, message: 'Invalid or expired refresh token' });
    }

    // Generate a new access token
    const newAccessToken = generateAccessToken(decoded.userId);

    // Store the new access token in the session
    req.session.accessToken = newAccessToken;

    res.cookie("token", newAccessToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production", // Use HTTPS in production
      sameSite: "None", // Adjust based on your requirements
      maxAge:  60*60 * 1000, // 1 hour
    });

    const userId = decoded.userId
    const user = await User.findById({userId})

    if(!user){
        console.log(`User not found: ${user}`);
        return res.status(400).json({ success: false, message: 'User not found' });
    }

    req.session.user = {
      id: user._id,
      email: user.email,
      username: user.username,
  };


    return res.status(200).json({
      success: true,
      message: 'Access token refreshed successfully',
      accessToken: newAccessToken
    });
  } catch (err) {
    console.error('Error refreshing token', err);
    return res.status(500).json({ success: false, message: 'Server error' });
  }
};
