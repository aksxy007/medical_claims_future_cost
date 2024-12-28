import mongoose from "mongoose";

const ProjectSchema = new mongoose.Schema({
    name:{
      type:String,
      required:true,
      unique:true
    },
    user:{
      type:mongoose.Schema.Types.ObjectId,
      ref:"User",
      required:true
    },
    projectType:{
      type:String,
      required:true
    },
    experiments:[{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Experiment'
    }]
}, { timestamp: true });

const Project = mongoose.model("Project",ProjectSchema)

export default Project
