import express from "express";
import authenticate from "../middleware/authenticate.js";
import { getConfig } from "../controllers/configController.js";


const router = express.Router()

router.get("/default-config",authenticate,getConfig)

export default router;