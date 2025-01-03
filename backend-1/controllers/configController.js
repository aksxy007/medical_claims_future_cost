import Experiments from "../models/Experiment.js"
import { readJsonFile } from "../utils/readJsonFile.js"

export const getConfig = async (req,res) =>{
    try {
        // const data= json.load("config.json")
        // console.log("default config",data)
        console.log("Fetching experiment config")
        const {experimentId} = req.body;

        const experiment = await Experiments.findById({_id:experimentId})

        let data = experiment.config
        if(!data){
            console.log(`No config found in experimentID: ${experimentId}`)
            data =await readJsonFile()
        }

        console.log("Experiment config sent!")
        return res.status(200).json({message:data  })
    } catch (error) {
        console.log("Error in getConfig",error)
        return res.status(400).json({error:"Unable to push config"})
    }

}


export const pushDefaulConfig = async (req,res)=>{

    try {
        // const data= json.load("config.json")
        // console.log("default config",data)
    
        return res.status(200).json({message:data})
    } catch (error) {
        console.log("Error in pushDefaultConfig",error)
        return res.status(400).json({error:"Unable to push config"})
    }
   
}