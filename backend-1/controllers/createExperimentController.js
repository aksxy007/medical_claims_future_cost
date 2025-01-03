import Experiments from "../models/Experiment.js";
import Project from "../models/Project.js";
import User from "../models/Users.js";


export const createExperiment = async (req, res) => {
    const { userId, projectId, experimentName } = req.body;

    try {
        console.log("Received request to create experiment:", { userId, projectId, experimentName });

        // Check if required fields are present
        if (!userId || !projectId || !experimentName) {
            console.log("Missing required fields: userId, projectId or experimentName");
            return res.status(400).json({ message: "Missing required fields" });
        }

        // Find the user to make sure they exist
        const user = await User.findById(userId);
        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: "User not found" });
        }

        console.log(`User with ID ${userId} found`);

        // Find the project to which the experiment will be added
        const project = await Project.findOne({ _id: projectId, user: userId });
        if (!project) {
            console.log(`Project with ID ${projectId} not found or does not belong to user ${userId}`);
            return res.status(404).json({ error: "Project not found or unauthorized" });
        }

        const projectType = project.projectType;

        console.log(`Project with ID ${projectId} found`);

        // Check if the experiment already exists in the project
        let experiment = await Experiments.findOne({
            name: experimentName,
            user:userId,
            project: projectId
        });

        if (!experiment) {
            // Create a new experiment
            experiment = new Experiments({
                name: experimentName,
                user:userId,
                project: projectId,
                status: "pending", // Default status
            });

            await experiment.save();
            console.log(`Experiment with name ${experimentName} created`);

            // Add the experiment reference to the project
            project.experiments.push(experiment._id);
            await project.save();
            console.log(`Experiment reference added to project with ID ${projectId}`);
        } else {
            console.log(`Experiment with name ${experimentName} already exists in the project`);
        }

        // Return success response
        return res.status(200).json({
            message: "Experiment created or already exists in the project",
            experiment,
            projectType
        });
    } catch (error) {
        console.error("Error in createExperiment:", error);
        return res.status(500).json({ error: "Server error" });
    }
};


export const deleteExperiment = async (req, res) => {
    const { userId, projectId, experimentId } = req.body;

    try {
        console.log("Received request to delete experiment:", { userId, projectId, experimentId });

        // Check if required fields are present
        if (!userId || !projectId || !experimentId) {
            console.log("Missing required fields: userId, projectId, or experimentId");
            return res.status(400).json({ message: "Missing required fields" });
        }

        // Find the user to make sure they exist
        const user = await User.findById(userId);
        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: "User not found" });
        }

        console.log(`User with ID ${userId} found`);

        // Find the project to which the experiment is linked
        const project = await Project.findOne({ _id: projectId, user: userId });
        if (!project) {
            console.log(`Project with ID ${projectId} not found or does not belong to user ${userId}`);
            return res.status(404).json({ error: "Project not found or unauthorized" });
        }

        console.log(`Project with ID ${projectId} found`);

        // Find the experiment to be deleted
        const experiment = await Experiments.findOne({ _id: experimentId, project: projectId });
        if (!experiment) {
            console.log(`Experiment with ID ${experimentId} not found in project ${projectId}`);
            return res.status(404).json({ error: "Experiment not found in project" });
        }

        console.log(`Experiment with ID ${experimentId} found`);

        // Remove the experiment reference from the project
        project.experiments.pull(experimentId);
        await project.save();
        console.log(`Experiment reference removed from project with ID ${projectId}`);

        // Delete the experiment itself
        await experiment.remove();
        console.log(`Experiment with ID ${experimentId} deleted successfully`);

        // Return success response
        return res.status(200).json({
            message: "Experiment deleted successfully",
        });
    } catch (error) {
        console.error("Error in deleteExperiment:", error);
        return res.status(500).json({ error: "Server error" });
    }
};



/**
 * Fetch user's projects and their related experiments.
 * @param {object} req - Express request object.
 * @param {object} res - Express response object.
 */
export const getUserProjectsExperiments = async (req, res) => {
    const {userId,projectId} = req.query;

    console.log("userId",userId)
    console.log("ProjectId",projectId)
    if (!userId) {
        return res.status(400).json({ success: false, message: 'User ID is required' });
    }

    try {
        // Find the user
        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ success: false, message: 'User not found' });
        }

        // Fetch the user's projects
        const project = await Project.findOne({ user: userId, _id:projectId });
        // console.log(project)
        // Prepare the response data structure
        let responseData=[];

        if (project) {
            // Fetch experiments and structure data if projects are found

                // Fetch the experiments related to the current project
                const experiments = await Experiments.find({ project: project._id }).sort({updatedAt:-1});

                // Map experiments to the desired structure
                // const projectData = {
                //     experiments.map(experiment => ({
                //         title: experiment.name,
                //         projectName:project.name,
                //         id:experiment._id,
                //         projectId:project._id,
                //         url: `/dashboard/${project._id}/${experiment._id}`,
                //     })),
                // };

                // Add the project data to the response array
                responseData=(experiments.map(experiment => ({
                    title: experiment.name,
                    projectName:project.name,
                    id:experiment._id,
                    createdAt:experiment.createdAt,
                    lastRun:experiment.updatedAt,
                    projectId:project._id,
                    url: `/dashboard/${project._id}/${experiment._id}`,
                })));
        } 

        console.log(responseData)
        // Send the response with the data structure
        console.log("Experiments Data sent!!")
        return res.status(200).json({ data: responseData });

    } catch (error) {
        console.error('Error fetching user project experiments:', error);
        return res.status(500).json({ success: false, message: 'Internal server error' });
    }
};
