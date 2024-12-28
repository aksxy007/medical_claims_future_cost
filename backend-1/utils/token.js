// token.js
import jwt from 'jsonwebtoken';

const JWT_ACCESS_SECRET = process.env.ACCESS_TOKEN_SECRET;
const JWT_REFRESH_SECRET = process.env.REFRESH_TOKEN_SECRET;
const JWT_ACCESS_EXPIRY = process.env.ACCESS_TOKEN_EXPIRY; // 15min
const JWT_REFRESH_EXPIRY = process.env.REFRESH_TOKEN_EXPIRY; // 7days

/**
 * Generates an access token for the user.
 * @param {string} userId - The ID of the user.
 * @returns {string} - The signed JWT access token.
 */
export const generateAccessToken = (userId) => {
    try {
        const token = jwt.sign({ userId }, JWT_ACCESS_SECRET, {
            expiresIn: JWT_ACCESS_EXPIRY,
        });
        console.log(`Access token generated for user ${userId}`);
        return token;
    } catch (error) {
        console.error(`Error generating access token for user ${userId}:`, error);
        throw new Error('Could not generate access token');
    }
};

/**
 * Generates a refresh token for the user.
 * @param {string} userId - The ID of the user.
 * @returns {string} - The signed JWT refresh token.
 */
export const generateRefreshToken = (userId) => {
    try {
        const token = jwt.sign({ userId }, JWT_REFRESH_SECRET, {
            expiresIn: JWT_REFRESH_EXPIRY,
        });
        console.log(`Refresh token generated for user ${userId}`);
        return token;
    } catch (error) {
        console.error(`Error generating refresh token for user ${userId}:`, error);
        throw new Error('Could not generate refresh token');
    }
};

/**
 * Verifies an access token.
 * @param {string} token - The JWT access token to verify.
 * @returns {object} - The decoded token payload if valid.
 */
export const verifyAccessToken = (token) => {
    try {
        const decoded = jwt.verify(token, JWT_ACCESS_SECRET);
        if(decoded){
          console.log(`Access token verified:`, decoded);
          return decoded;
        }else{
          return null
        }
    } catch (error) {
        logTokenError(error);
        // throw new Error('Invalid or expired access token');
        return null
    }
};

/**
 * Verifies a refresh token.
 * @param {string} token - The JWT refresh token to verify.
 * @returns {object} - The decoded token payload if valid.
 */
export const verifyRefreshToken = (token) => {
    try {
        const decoded = jwt.verify(token, JWT_REFRESH_SECRET);
        if(decoded){
          console.log(`Refresh token verified:`, decoded);
          return decoded;
        }else{
          return null
        }
    } catch (error) {
        logTokenError(error);
        // throw new Error('Invalid or expired refresh token');
        return null
    }
};

/**
 * Logs detailed information about token generation or verification errors.
 * @param {Error} error - The error object.
 */
export const logTokenError = (error) => {
    console.error(`JWT Error: ${error.message}`);
    if (error.name === 'TokenExpiredError') {
        console.error('The token has expired');
    } else if (error.name === 'JsonWebTokenError') {
        console.error('The token is invalid');
    } else {
        console.error('An unknown JWT error occurred');
    }
};

export default {
    generateAccessToken,
    generateRefreshToken,
    verifyAccessToken,
    verifyRefreshToken,
    logTokenError,
};
