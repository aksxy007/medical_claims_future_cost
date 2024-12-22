
export const getUserSession = (req, res)=>{
    if (req.session.user) {
        res.json({ user: req.session.user }); // Return session user details
      } else {
        res.status(401).json({ error: "Not authenticated" });
      }
}
