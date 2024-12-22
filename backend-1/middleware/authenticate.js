// middleware/auth.js
import { verifyAccessToken } from '../utils/token.js';

export const authenticate = (req, res, next) => {
  const accessToken = req.session.accessToken;

  if (!accessToken) {
    return res.status(401).json({ success: false, message: 'No access token, authorization denied' });
  }

  // Verify the access token
  const decoded = verifyAccessToken(accessToken);
  if (!decoded) {
    return res.status(403).json({ success: false, message: 'Invalid or expired access token' });
  }

  // Attach user info to request object
  req.user = decoded;
  next();  // Proceed to the next middleware or route
};
