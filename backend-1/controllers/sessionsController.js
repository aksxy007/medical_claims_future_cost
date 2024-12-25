import User from "../models/Users.js"; // Import the User model
import { verifyAccessToken } from "../utils/token.js";

export const getUserSession = async (req, res) => {
  const token = req.cookies?.token; // Assuming token is stored in cookies
  if (!token) {
    console.log("No token found");
    return res.status(401).json({ message: "Unauthorized" });
  }

  try {
    // Verify the token and extract the user ID
    const decoded = verifyAccessToken(token);
    if (!decoded) {
      return res.status(401).json({ message: "Invalid token" });
    }

    const userId = decoded.userId; // Extract userId from the decoded token
    console.log("Decoded userId:", userId);

    // Fetch user details from the database
    const user = await User.findById(userId).select("username email"); // Retrieve only username and email

    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Return the user information along with the access token
    console.log("session details sent!")
    return res.status(200).json({
      user: {
        id: user._id,
        username: user.username,
        email: user.email,
      },
      accessToken: token,
    });
  } catch (error) {
     // If token expired, explicitly return a 401 response with message
     if (error.message === "TokenExpired") {
      return res.status(401).json({ message: "Access token expired" });
    }

    return res.status(401).json({ message: "Invalid token" });
  }
};
