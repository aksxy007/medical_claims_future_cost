// controllers/logout.js
export const logoutController = (req, res) => {
    // Destroy the session
    req.session.destroy((err) => {
      if (err) {
        return res.status(500).json({ success: false, message: 'Failed to log out' });
      }
  
      // Clear the refresh token cookie
      res.clearCookie('refreshToken', { httpOnly: true, secure: process.env.NODE_ENV === 'production' });
      console.log("User Logged out successfully!!!")
      return res.status(200).json({ success: true, message: 'Logged out successfully' });
    });
  };
  