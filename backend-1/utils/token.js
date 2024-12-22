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


export const verifyAccessToken = (token)=>{
    try {
        return jwt.verify(token,ACCESS_TOKEN_SECRET)
    } catch (error) {
        console.log("User not autherised!!")
        return null
    }
}

export const verifyRefreshToken = (token)=>{
    try {
        return jwt.sign(token,REFRESH_TOKEN_SECRET)
    } catch (error) {
        console.error('Refresh token verification failed', err);
        return null;
    }
} 