// register.js
import bcrypt from 'bcryptjs';
import User from '../models/Users.js';
import { generateAccessToken, generateRefreshToken } from '../utils/token.js';

/**
 * Handles user registration.
 * @param {object} req - Express request object.
 * @param {object} res - Express response object.
 */
const register = async (req, res) => {
    const { username, email, password } = req.body;

    try {
        // Check if the user already exists
        const userExists = await User.findOne({ email });
        if (userExists) {
            console.log(`Registration failed: User already exists with email ${email}`);
            return res.status(400).json({ success: false, message: 'User already exists' });
        }

        // Hash the user's password
        const salt = await bcrypt.genSalt(10); // Generate salt for hashing
        const hashedPassword = await bcrypt.hash(password, salt); // Hash password

        // Create new user
        const newUser = new User({
            username,
            email,
            password: hashedPassword, // Save the hashed password
        });

        // Save user to database
        await newUser.save();
        console.log(`User registered successfully: ${email}`);

        // Generate tokens
        const accessToken = generateAccessToken(newUser._id);
        const refreshToken = generateRefreshToken(newUser._id);

        // Send response with tokens
        res.cookie('refreshToken', refreshToken, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production',
            maxAge: 1000 * 60 * 60 * 24 * 7, // 7 day
            
        });

        return res.status(201).json({
            success: true,
            message: 'Registration successful',
            user: {
                id: newUser._id,
                username: newUser.username,
                email: newUser.email,
            },
            accessToken,
        });
    } catch (error) {
        console.error('Error during registration:', error);
        return res.status(500).json({ success: false, message: 'Internal server error' });
    }
};

export default register;
