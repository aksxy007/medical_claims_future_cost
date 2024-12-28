const logout = (req, res) => {
  try {
    // Clear the refresh token from the cookies
    res.clearCookie('refreshToken', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: "None",
    });

    // Destroy the session (if you're using express-session)
    req.session.destroy((err) => {
      if (err) {
        console.error('Error destroying session', err);
        return res.status(500).json({ success: false, message: 'Error logging out' });
      }

      console.log(`User logged out successfully, session destroyed.`);
      return res.status(200).json({ success: true, message: 'Logged out successfully' });
    });
  } catch (error) {
    console.error('Error during logout', error);
    return res.status(500).json({ success: false, message: 'Error during logout' });
  }
};

export default logout;
