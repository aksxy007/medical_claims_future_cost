
export const getUserSession = (req, res)=>{

    if (req.session.user) {
        res.json({ user: req.session.user, accessToken:req.session.accessToken }); // Return session user details()
        console.log("session details sent!!")
      } else {
        res.status(401).json({ error: "Not authenticated" });
      }
}
