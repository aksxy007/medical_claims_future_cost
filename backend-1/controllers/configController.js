import { readJsonFile } from "../utils/readJsonFile.js"

export const getConfig = async (req,res) =>{
    try {
        // const data= json.load("config.json")
        // console.log("default config",data)
        const data =await readJsonFile()
        return res.status(200).json({message:data  })
    } catch (error) {
        console.log("Error in pushDefaultConfig",error)
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