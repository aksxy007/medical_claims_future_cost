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
    
        // Set refresh token in a secure cookie
        res.cookie('refreshToken', refreshToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        maxAge: 1000 * 60 *60*24*7 , // 7 day for refreshToken
        // sameSite: "None"
        });        

        req.session.accessToken = accessToken;
        req.session.user = {
            id: user._id,
            email: user.email,
            username: user.username,
        };
      

        return res.status(200).json({
        success: true,
        message: 'Login successful',
        user: req.session.user,
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


