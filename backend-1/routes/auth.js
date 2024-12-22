// routes/index.js
import express from 'express';
import login from '../controllers/login.js';
import register  from '../controllers/register.js';
import { refreshTokenController } from '../controllers/refreshTokenController.js';
import { logoutController } from '../controllers/logout.js';
import { authenticate } from '../middleware/authenticate.js';

const router = express.Router();

// POST route for user registration
router.post('/register', register);

// POST route for user login
router.post('/login', login);

router.post('/refresh-token', refreshTokenController);

// Logout Route
router.post('/logout', logoutController);


export default router;
