import { useState, useEffect, createContext, useContext } from 'react';
import apiClient from '@/lib/api-client';
import { useAuth } from './use-auth';


const ProjectsContext = createContext();

// Create a custom hook to access the context
export const ProjectsProvider = ({children}) => {
    const {user} = useAuth()
  const [selectedPipeline,setSelectedPipeline] = useState("Modelling")
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
//   console.log("Selected Pipeline",selectedPipeline)
  // Fetch projects

  const setCurrentPipeline =(value)=>{
    setSelectedPipeline(()=>value)
  }

  const fetchProjects = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get(`/projects/get-user-projects`, {
        params: {
          userId:user.id,
          projectType: selectedPipeline,
        },
      });
      const data = response.data?.data || [];
      setProjects(data);
    } catch (err) {
      setError("Failed to fetch projects.");
      console.error("Error fetching projects:", err);
    } finally {
      setLoading(false);
    }
  };

  // Add a new project or experiment
  const addNewProject = async (newProjectName, newExperimentName) => {
    try {
      // Check if the project already exists
      const existingProject = projects.find(project => project.title === newProjectName);
  
      if (existingProject) {
        // If the project exists, check if the experiment exists
        const existingExperiment = existingProject.experiments.find(experiment => experiment.title === newExperimentName);
  
        // If the experiment exists, do nothing
        if (existingExperiment) {
          console.log("Both project and experiment already exist. Doing nothing.");
          return; // Exit function early
        }
  
        // If the experiment does not exist, call API to create the experiment
        const response = await apiClient.post('/projects/create-project', {
          userId: user.id,
          projectName: newProjectName, // Use the existing project ID
          experimentName: newExperimentName,
          projectType: selectedPipeline,
        });
        
        if(response.status===200){
            const newExperiment = {
            title: newExperimentName,
            url: `/dashboard/${newExperimentName}`,
            };
  
        // Update the projects state to reflect the new experiment
            setProjects((prevProjects) => {
            const updatedProjects = prevProjects.map(project =>
                project.id === existingProject.id
                ? { ...project, experiments: [...project.experiments, newExperiment] }
                : project
            );
            console.log("Updated project with new experiment:", updatedProjects);
            return updatedProjects;
            });
        }
        setError(null)
        return;
      }
  
      // If the project does not exist, call API to create the project and the experiment
      const response = await apiClient.post('/projects/create-project', {
        userId: user.id,
        projectName: newProjectName,
        projectType: selectedPipeline,
        experimentName: newExperimentName,
      });
  
      const newProject = {
        id: response.data?.experiment?.project, // Ensure this matches the project ID from the backend
        title: newProjectName,
        experiments: [
          {
            title: newExperimentName,
            url: `/dashboard/${newExperimentName}`,
          },
        ],
      };
  
      // Add the new project and experiment
      setProjects((prevProjects) => {
        const updatedProjects = [...prevProjects, newProject];
        console.log("Added new project with experiment:", updatedProjects);
        return updatedProjects;
      });

      setError(null)
  
    } catch (err) {
      setError('Failed to add new project or experiment.');
      console.error('Error adding new project/experiment:', err);
    }
  };
  
  

  // Delete a project
  const deleteProject = async (projectId) => {
    try {
      const response = await apiClient.post('/projects/delete-project', {
        userId:user.id,
        projectId,
      });

      console.log('Project Deleted Successfully:', response.data.message);

      // Re-fetch projects after deletion
    //   await fetchProjects();
    setError(null)
    setProjects((prevProjects) => prevProjects.filter((p) => p.id !== projectId));
    } catch (err) {
      setError('Failed to delete project.');
      console.error('Error deleting project:', err);
    }
  };

  // Fetch projects on mount or when dependencies change
  useEffect(() => {
    if (user && user?.id && selectedPipeline) {
      fetchProjects();
    }
  }, [user?.id, selectedPipeline]);

  return (<ProjectsContext.Provider value={{
    projects,
    loading,
    error,
    selectedPipeline,
    setCurrentPipeline,
    fetchProjects,
    addNewProject,
    deleteProject,
  }}>
    {children}
  </ProjectsContext.Provider >
    )
};

export const useProjects = ()=>useContext(ProjectsContext)
