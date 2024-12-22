// controllers/refreshToken.js
import { verifyRefreshToken } from '../utils/token.js';
import { generateAccessToken } from '../utils/token.js';

export const refreshTokenController = (req, res) => {
  const { refreshToken } = req.cookies;  // Retrieve refresh token from cookie

  if (!refreshToken) {
    return res.status(400).json({ success: false, message: 'No refresh token provided' });
  }

  try {
    // Verify refresh token
    const decoded = verifyRefreshToken(refreshToken);
    if (!decoded) {
      return res.status(403).json({ success: false, message: 'Invalid or expired refresh token' });
    }

    // Generate a new access token
    const newAccessToken = generateAccessToken(decoded.userId);

    // Store the new access token in the session
    req.session.accessToken = newAccessToken;

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
