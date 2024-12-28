import { generateAccessToken, generateRefreshToken, verifyRefreshToken } from '../utils/token.js';
import User from '../models/Users.js';

const refreshToken = async (req, res) => {
  try {
    // Get refresh token from cookies
    const refreshToken = req.cookies?.refreshToken;

    // If no refresh token is found, log the event and return an error
    if (!refreshToken) {
      console.log('No refresh token provided in request');
      return res.status(403).json({ success: false, message: 'No refresh token provided' });
    }

    console.log('Received refresh token, attempting to verify it');

    // Verify the refresh token
    const decoded = verifyRefreshToken(refreshToken);  // Implement verifyRefreshToken function based on your token generation logic
    if (!decoded) {
      console.log('Invalid or expired refresh token');
      return res.status(403).json({ success: false, message: 'Invalid or expired refresh token' });
    }
    const userId = decoded.userId
    console.log(`Refresh token verified successfully, user ID: ${userId}`);

    // Find the user by ID
    const user = await User.findById(userId);
    if (!user) {
      console.log(`User not found with ID: ${userId}`);
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    console.log(`User found: ${user.email}, generating new tokens`);

    // Generate new access token and refresh token
    const newAccessToken = generateAccessToken(user._id);
    // const newRefreshToken = generateRefreshToken(user._id);

    // Set the new refresh token in the cookie
    // res.cookie('refreshToken', newRefreshToken, {
    //   httpOnly: true,
    //   secure: process.env.NODE_ENV === 'production',
    //   maxAge: 1000 * 60 * 60 * 24 * 7, // 1 day
    //   // sameSite: "None",
    // });

    req.session.accessToken = newAccessToken;

    console.log('New tokens generated and refresh token set in cookies');

    // Respond with the new access token and refresh token
    return res.status(200).json({
      success: true,
      message: 'Tokens refreshed successfully',
      accessToken: newAccessToken,
    });
  } catch (error) {
    console.error('Error during token refresh:', error);
    return res.status(500).json({ success: false, message: error.message || 'An error occurred' });
  }
};

export default refreshToken;
