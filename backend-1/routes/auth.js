// routes/index.js
import express from 'express';
import login from '../controllers/login.js';
import register  from '../controllers/register.js';
import refreshToken from '../controllers/refreshTokenController.js';
import logout from '../controllers/logout.js'
import checkSession from '../controllers/sessionsController.js';

const router = express.Router();

// POST route for user registration
router.post('/register', register);

// POST route for user login
router.post('/login', login);

router.post("/logout",logout)

router.post('/refresh-token', refreshToken);

router.get("/check-session",checkSession)

export default router;
