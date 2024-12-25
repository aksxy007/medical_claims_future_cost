import jwt from 'jsonwebtoken';

const ACCESS_TOKEN_SECRET = process.env.ACCESS_TOKEN_SECRET
const REFRESH_TOKEN_SECRET = process.env.REFRESH_TOKEN_SECRET;
const ACCESS_TOKEN_EXPIRATION = process.env.ACCESS_TOKEN_EXPIRATION; // 1 hour
const REFRESH_TOKEN_EXPIRATION = process.env.REFRESH_TOKEN_EXPIRATION; // 7 days


export const generateAccessToken = (userId)=>{
    return jwt.sign({userId},ACCESS_TOKEN_SECRET,{expiresIn:ACCESS_TOKEN_EXPIRATION})
}

export const generateRefreshToken = (userId)=>{
    return jwt.sign({userId},REFRESH_TOKEN_SECRET,{expiresIn:REFRESH_TOKEN_EXPIRATION})
}   


export const verifyAccessToken = (token) => {
    try {
      return jwt.verify(token, ACCESS_TOKEN_SECRET);
    } catch (error) {
      if (error.name === "TokenExpiredError") {
        console.log("Access token has expired");
        throw new Error("TokenExpired");
      }
      console.log("Invalid access token");
      throw new Error("InvalidToken");
    }
  };
  
  export const verifyRefreshToken = (token) => {
    try {
      return jwt.verify(token, REFRESH_TOKEN_SECRET);
    } catch (error) {
      if (error.name === "TokenExpiredError") {
        console.log("Refresh token has expired");
        throw new Error("TokenExpired");
      }
      console.log("Invalid refresh token");
      throw new Error("InvalidToken");
    }
  };