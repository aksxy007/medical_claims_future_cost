import mongoose from "mongoose";

const ExperimentSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true,
    },
    user:{
        type:mongoose.Schema.Types.ObjectId,
        ref:"User",
        required:true
    },
    status: {
      type: String,
      enum: ["pending", "running", "completed", "failed"],
      default: "pending",
    },
    project:{
        type:mongoose.Schema.Types.ObjectId,
        ref:"Project",
        required:true
    },
    createdAt: {
      type: Date,
      default: Date.now,
    },
    updatedAt: {
      type: Date,
      default: Date.now,
    },
    logsPath: {
      type: String, // Path to the logs file
    },
    resultsPath: {
      type: String, // Path to the results file or folder
    },
    tags: {
      type: [String], // Optional categorization
    }
  },
  {
    timestamps: true, // Automatically add createdAt and updatedAt
  }
);

const Experiments = mongoose.model("Experiment", ExperimentSchema);

export default Experiments
