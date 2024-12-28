import express from 'express'
import authenticate from '../middleware/authenticate.js'
import { createProject, deleteProject, getUserProjects } from '../controllers/createProjectController.js'

const router = express.Router()

router.post("/create-project",authenticate,createProject)

router.post("/delete-project",authenticate,deleteProject)

router.get("/get-user-projects",authenticate,getUserProjects)

export default router;