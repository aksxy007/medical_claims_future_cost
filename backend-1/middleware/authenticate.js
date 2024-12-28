import { verifyAccessToken } from '../utils/token.js';

const authenticate = (req, res, next) => {
  // Get the token from the Authorization header (format: Bearer <token>)
  const authHeader = req.headers.authorization;

  if (!authHeader) {
    console.log('No authorization header provided');
    return res.status(401).json({ success: false, message: 'Authorization token is required' });
  }

  // Extract the token from the header
  const token = authHeader.split(' ')[1];

  if (!token) {
    console.log('No token found in authorization header');
    return res.status(401).json({ success: false, message: 'Access token is missing' });
  }

  // Use the existing verifyAccessToken function to validate the token
  const decoded = verifyAccessToken(token);
  if (!decoded) {
    console.log('Invalid or expired access token');
    return res.status(403).json({ success: false, message: 'Invalid or expired access token' });
  }

  // Token is valid, attach user data to request object
  req.user = decoded; // You can store the user details (like user ID, email) here
  console.log('Token verified successfully, user ID:', decoded.userId);

  // Call the next middleware or route handler
  next();
};

export default authenticate;
