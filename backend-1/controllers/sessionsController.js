// controllers/checkSessionController.js
import { verifyAccessToken } from '../utils/token.js';

const checkSession = (req, res) => {
  try {
    // Check if the user has a valid access token in the session
    const accessToken = req.session.accessToken;

    if (!accessToken) {
      return res.status(403).json({
        success: false,
        message: 'Session expired or user not logged in',
      });
    }

    // Verify the access token to check if it's still valid
    const decoded = verifyAccessToken(accessToken); // Assuming `verifyAccessToken` is in `token.js`
    if (!decoded) {
      return res.status(403).json({
        success: false,
        message: 'Invalid or expired access token',
      });
    }

    // console.log(req.session)

    // If access token is valid, return session details
    return res.status(200).json({
      success: true,
      message: 'Session is active',
      user: req.session?.user, // Return user data from the session
      accessToken: req.session.accessToken
    });
  } catch (error) {
    console.error('Error checking session', error);
    return res.status(500).json({
      success: false,
      message: 'Internal server error',
    });
  }
};

export default checkSession;
