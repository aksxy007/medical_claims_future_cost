import Experiments from "../models/Experiment.js";
import Project from "../models/Project.js";
import User from "../models/Users.js";

export const createProject = async (req, res) => {
    const { userId, projectType, projectName, experimentName } = req.body;

    try {
        if (!userId || !projectType || !projectName || !experimentName) {
            return res.status(400).json({ message: "Missing required fields" });
        }

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).json({ error: "User not found" });
        }

        let project = await Project.findOne({ name: projectName, user: userId });

        if (!project) {
            // Create a new project if it does not exist
            project = new Project({
                name: projectName,
                user: userId,
                projectType,
                experiments: [],
            });

            await project.save();
            user.projects.push(project._id);
            await user.save();
        }

        // Check if the experiment already exists in the project
        let experiment = await Experiments.findOne({ name: experimentName, project: project._id });

        if (!experiment) {
            // Create a new experiment if it does not exist
            experiment = new Experiments({
                name: experimentName,
                user: userId,
                project: project._id,
                status: "pending",
                tags: [projectType],
            });

            await experiment.save();
            project.experiments.push(experiment._id);
            await project.save();
        }

        return res.status(200).json({
            message: "Project and experiment created or updated successfully",
            projectId: project._id,
            experiment,
            projectType,
        });
    } catch (error) {
        console.error("Error in createProject:", error);
        return res.status(500).json({ error: "Internal server error" });
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
        await project.deleteOne({_id:projectId});
        console.log(`Project with ID ${projectId} deleted successfully`);

        user.projects.pull(projectId)
        await user.save()
        // Return success response
        return res.status(200).json({
            message: "Project and related experiments deleted successfully",
        });
    } catch (error) {
        console.error("Error in deleteProject:", error);
        return res.status(500).json({ error: error });
    }
};


/**
 * Fetch user's projects and their related experiments.
 * @param {object} req - Express request object.
 * @param {object} res - Express response object.
 */

export const getUserProjects = async (req, res) => {
    const {userId,projectType} = req.query;

    console.log("userId",userId)
    console.log("ProjectType",projectType)
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
        const projects = await Project.find({ user: userId, projectType:projectType });

        // Prepare the response data structure
        const responseData = [];

        if (projects.length > 0) {
            // Fetch experiments and structure data if projects are found
            for (const project of projects) {
                // Fetch the experiments related to the current project
                // const experiments = await Experiments.find({ project: project._id }).sort({updatedAt:-1});

                // Map experiments to the desired structure
                const projectData = {
                    title: project.name,
                    url:"#",
                    id:project._id,
                    // items: experiments.map(experiment => ({
                    //     title: experiment.name,
                    //     url: `/dashboard/${project._id}/${experiment._id}`,
                    // })),
                };

                // Add the project data to the response array
                responseData.push(projectData);
            }
        } 
        // Send the response with the data structure
        console.log("Project Data sent!!")
        return res.status(200).json({ data: responseData });

    } catch (error) {
        console.error('Error fetching user projects and experiments:', error);
        return res.status(500).json({ success: false, message: 'Internal server error' });
    }
};
