import Experiments from "../models/Experiment.js";
import Project from "../models/Project.js";
import User from "../models/Users.js";
import ProducerService from "../services/RabbitMQProducerService.js";
import apiClient from "../utils/api-client.js"

export const triggerAirflowRun = async (req,res)=>{
   try {

        const {runConfig,userId,projectId,experimentId} = req.body
        let customRunConfig = JSON.parse(runConfig)

        if(!userId || !projectId || !experimentId){
            console.log(`Missing required fields in the request body: ${userId}, ${projectId},${experimentId}`);
            return res.status(400).json({ message: "Missing required fields " });
        }
        const user = await User.findOne({_id:userId})
        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: "User not found" });
        }

        console.log(`User with ID ${userId} found:`, user);

        // Check if project already exists
        let project = await Project.findOne({ _id:projectId});
        if(!project){
            console.log(`Project with ID ${projectId} not found`);
            return res.status(404).json({ error: "Project not found" });
        }

        let experiment = await Experiments.findOne({_id:experimentId});
        if (!experiment) {
            console.log(`Experiment with ID ${experimentId} not found`);
            return res.status(404).json({ error: "Experiment not found" });
        }


        customRunConfig.input_folder = `${userId}/${projectId}/${experimentId}/${customRunConfig.input_folder}`
        customRunConfig.output_folder = `${userId}/${projectId}/${experimentId}/${customRunConfig.output_folder}`  
        // Add user related prefix for input and output paths 


        console.log("Final runConfig:", customRunConfig);

        // const response = await apiClient.post("/trigger/trigger-pipeline",{
        //     "config":customRunConfig
        // })
        // const response = await apiClient.get("/trigger/health")
        
        const message = {
            "config":customRunConfig
        }
        const producerService =new ProducerService()
        await producerService.sendMessage(process.env.RABBITMQ_QUEUE_NAME,message);

        console.log("pipeline triggered ")

        return res.status(200).json({message:"pipeline triggered"})
   } catch (error) {
        console.log("error",error)
        return res.status(500).json({error:error})
   }
}