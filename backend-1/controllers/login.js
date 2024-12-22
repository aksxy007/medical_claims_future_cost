import User from "../models/Users.js";
import bcrypt from 'bcryptjs'
import {generateAccessToken,generateRefreshToken} from '../utils/token.js'


const login = async (req,res)=>{

    try {
        
        const {email,password} = req.body

        const user = await User.findOne({email})

        if(!user){
            console.log(`User not found: ${email}`);
            return res.status(400).json({ success: false, message: 'User not found' });
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
        console.log(`Invalid credentials for: ${email}`);
        return res.status(400).json({ success: false, message: 'Invalid credentials' });
        }

        console.log(`User logged in: ${user.email}`);

        // Generate tokens
        const accessToken = generateAccessToken(user._id);
        const refreshToken = generateRefreshToken(user._id);

        res.cookie('refreshToken', refreshToken, {
            httpOnly: true, // Prevent JavaScript access
            secure: process.env.NODE_ENV === 'production', // Secure cookie in production (HTTPS)
            maxAge: 1000 * 60 * 60 * 24 * 7 // 1 week expiration
          });

        
        req.session.accessToken = accessToken;

        return res.status(200).json({
        success: true,
        message: 'Login successful',
        accessToken,
        refreshToken
        });
    }
    catch (error) {
        console.error('Error during login', error);
        return res.status(500).json({ success: false, message: error });
    }
    
}

export default login;


