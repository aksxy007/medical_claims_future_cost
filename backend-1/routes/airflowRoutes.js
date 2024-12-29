import experss from 'express'
import authenticate from '../middleware/authenticate.js'
import { triggerAirflowRun } from '../controllers/triggerAirflowController.js'


const router= experss.Router()

router.post("/trigger-run",authenticate,triggerAirflowRun)


export default router