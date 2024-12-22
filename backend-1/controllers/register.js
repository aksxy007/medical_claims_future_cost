import User from "../models/Users.js";
import bcrypt from 'bcryptjs'
import {generateAccessToken,generateRefreshToken} from '../utils/token.js'

const register = async (req,res)=>{
    const {username,email,password} = req.body;

    try {
        const existingUser =await User.findOne({email});
        if(existingUser){
            console.log(`${email} already in registered`)
            return res.status(400).json({success:false,message:'User already exists'});
        }

        const salt = await bcrypt.genSalt(10)
        const hashedPassword = await bcrypt.hash(password,salt)

        const newUser = new User({
            username,
            email,
            password:hashedPassword
        })

        await newUser.save()

        console.log(`User Registered!`);

        // const accessToken = generateAccessToken(newUser._id)
        // const refreshToken = generateRefreshToken(newUser._id)

        return res.status(201).json({
            success: true,
            message: 'User registered successfully',
            // accessToken,
            // refreshToken
          });

    } catch (error) {
        console.error('Error during registration', error);
        return res.status(500).json({ success: false, message: 'Server error' });
    }
}

export default register;