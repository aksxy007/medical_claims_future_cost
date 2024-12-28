import Experiments from "../models/Experiment.js";
import Project from "../models/Project.js";
import User from "../models/Users.js";

export const createProject = async (req, res) => {
    const { userId, projectType, projectName, experimentName } = req.body;

    try {
        // Log initial request data
        console.log("Request received to create or update project:", {
            userId,
            projectType,
            projectName,
            experimentName,
        });

        // Check for missing required fields
        if (!userId || !projectType || !projectName || !experimentName) {
            console.log("Missing required fields in the request body");

            return res.status(400).json({ message: "Missing required fields" });
        }

        // Fetch user to check if they exist
        const user = await User.findById(userId);
        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: "User not found" });
        }

        // Log user found
        console.log(`User with ID ${userId} found:`, user);

        // Check if project already exists
        let project = await Project.findOne({ name: projectName, user: userId });
        if (!project) {
            // Create new project if it does not exist
            console.log(`Project with name "${projectName}" not found. Creating new project...`);

            project = new Project({
                name: projectName,
                user: userId,
                projectType: projectType,
                experiments: [],
            });

            await project.save();
            user.projects.push(project._id)
            await user.save()
            console.log(`New project created:`, project);
        } else {
            console.log(`Project with name "${projectName}" already exists.`);
        }

        // Check if experiment already exists in the project
        let experiment = await Experiments.findOne({ name: experimentName, project: project._id });
        if (!experiment) {
            // Create new experiment if it does not exist
            console.log(`Experiment with name "${experimentName}" not found in project. Creating new experiment...`);

            experiment = new Experiments({
                name: experimentName,
                user:userId,
                project: project._id,
                status: "pending", // Default status
            });

            await experiment.save();
            console.log(`New experiment created:`, experiment);

            // Add the new experiment to the project's experiments array
            project.experiments.push(experiment._id);
            await project.save();
            console.log(`Experiment added to project:`, project);
        } else {
            console.log(`Experiment with name "${experimentName}" already exists in the project.`);
        }

        // Return success response with experiment data
        return res.status(200).json({
            message: "Experiment created or updated successfully",
            experiment,
        });

    } catch (error) {
        // Log error
        console.error("Error in createOrUpdateExperiment:", error);
        return res.status(500).json({ error: "Server error" });
    }
};

export const deleteProject = async (req, res) => {
    const { userId, projectId } = req.body;

    try {
        console.log("Received request to delete project:", { userId, projectId });

        // Check if userId and projectId are provided
        if (!userId || !projectId) {
            console.log("Missing required fields: userId or projectId");
            return res.status(400).json({ message: "Missing required fields" });
        }

        // Find the user to make sure they exist
        const user = await User.findById(userId);
        if (!user) {
            console.log(`User with ID ${userId} not found`);
            return res.status(404).json({ error: "User not found" });
        }

        console.log(`User with ID ${userId} found`);

        // Find the project to be deleted and make sure it's owned by the user
        const project = await Project.findOne({ _id: projectId, user: userId });
        if (!project) {
            console.log(`Project with ID ${projectId} not found or does not belong to user ${userId}`);
            return res.status(404).json({ error: "Project not found or unauthorized" });
        }

        console.log(`Project with ID ${projectId} found:`, project);

        // Delete all experiments related to the project
        const experimentDeleteResult = await Experiments.deleteMany({ project: projectId });
        console.log(`Deleted ${experimentDeleteResult.deletedCount} experiments related to project ${projectId}`);

        // Delete the project itself
        await project.remove();
        console.log(`Project with ID ${projectId} deleted successfully`);

        // Return success response
        return res.status(200).json({
            message: "Project and related experiments deleted successfully",
        });
    } catch (error) {
        console.error("Error in deleteProject:", error);
        return res.status(500).json({ error: "Server error" });
    }
};


/**
 * Fetch user's projects and their related experiments.
 * @param {object} req - Express request object.
 * @param {object} res - Express response object.
 */
export const getUserProjects = async (req, res) => {
    const {userId} = req.query;

    console.log("userId",userId)
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
        const projects = await Project.find({ user: userId });

        // Prepare the response data structure
        const responseData = [];

        if (projects.length > 0) {
            // Fetch experiments and structure data if projects are found
            for (const project of projects) {
                // Fetch the experiments related to the current project
                const experiments = await Experiments.find({ project: project._id });

                // Map experiments to the desired structure
                const projectData = {
                    title: project.name,
                    experiments: experiments.map(experiment => ({
                        title: experiment.name,
                        url: `/dashboard/${experiment.name}`,
                    })),
                };

                // Add the project data to the response array
                responseData.push(projectData);
            }
        } else {
            // If no projects are found, return an empty data structure
            responseData.push({});
        }

        // Send the response with the data structure
        return res.status(200).json({ data: responseData });

    } catch (error) {
        console.error('Error fetching user projects and experiments:', error);
        return res.status(500).json({ success: false, message: 'Internal server error' });
    }
};

