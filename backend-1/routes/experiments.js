import express from 'express'
import authenticate from '../middleware/authenticate.js';
import { getUserProjectsExperiments } from '../controllers/createExperimentController.js';

const router = express.Router()

router.get("/get-project-experiments",authenticate,getUserProjectsExperiments)


export default router;