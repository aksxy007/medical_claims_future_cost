import { useState, useEffect, createContext, useContext } from 'react';
import apiClient from '@/lib/api-client';
import { useAuth } from './use-auth';


const ProjectsContext = createContext();

// Create a custom hook to access the context
export const ProjectsProvider = ({children}) => {
    const {user} = useAuth()
  const [selectedPipeline,setSelectedPipeline] = useState("Modelling")
  const [projects, setProjects] = useState([]);
  const [experiments,setExpriments] = useState([])
  const [projectId,setProjectId] = useState(null)
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
//   console.log("Selected Pipeline",selectedPipeline)
  // Fetch projects

  console.log(user)
  const setCurrentPipeline = (value)=>{
    setSelectedPipeline(()=>value)
  }

  // console.log("ProjectId",projectId)

  const fetchProjectExperiments= async ()=>{
    setLoading(true)
    setError(null)

    try {
      const response = await apiClient.get(`/projects/experiments/get-project-experiments`, {
        params: {
          userId:user.id,
          projectId: projectId,
        },
      });
      const data = response.data?.data || [];
      setExpriments(data);
    }  catch (err) {
      setError("Failed to fetch projects.");
      console.error("Error fetching projects:", err);
    } finally {
      setLoading(false);
    }
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
        setLoading(true);
        setError(null);

        // Check if the project already exists in the current state
        const existingProject = projects.find(project => project.title === newProjectName);
        console.log("existingProject",existingProject)
        if (existingProject) {
            // If the project exists, fetch its experiments
            const response = await apiClient.get("/projects/experiments/get-project-experiments", {
                params: {
                    userId: user.id,
                    projectId: existingProject.id,
                },
            });

            const existingExperiments = response.data?.data || [];
            console.log("Existing Experiments",existingExperiments)
            const existingExperiment = existingExperiments.find(experiment => experiment.title === newExperimentName);

            if (existingExperiment) {
                console.log("Experiment already exists in the project.");
                return existingExperiment; // Return the existing experiment
            }

            // Add a new experiment to the existing project
            const experimentResponse = await apiClient.post("/projects/create-project", {
                userId: user.id,
                projectName: newProjectName,
                experimentName: newExperimentName,
                projectType: selectedPipeline,
            });

            const newExperiment = experimentResponse.data?.experiment;
            const newExperimentData= {
              title: newExperiment.name,
              id: newExperiment._id,
              projectId: existingProject.id,
              createdAt: newExperiment.createdAt,
              lastRun:newExperiment.updatedAt,
              url: `/dashboard/${existingProject.id}/${newExperiment._id}`,
          }
            if (newExperiment) {
                setExpriments((prevExperiments) => [...prevExperiments,newExperimentData]);
                console.log("Added new experiment to the existing project.");
                return newExperimentData;
            }
        } else {
            // If the project does not exist, create both project and experiment
            const response = await apiClient.post("/projects/create-project", {
                userId: user.id,
                projectName: newProjectName,
                experimentName: newExperimentName,
                projectType: selectedPipeline,
            });

            const newProjectId = response.data?.projectId;
            const newExperiment = response.data?.experiment;

            if (newProjectId && newExperiment) {
                // Update projects and experiments state
                setProjects((prevProjects) => [...prevProjects, {
                    id: newProjectId,
                    url:"#",
                    title: newProjectName,
                }]);

                const newExperimentData=  {
                  title: newExperiment.name,
                  id: newExperiment._id,
                  projectId: newProjectId,
                  createdAt: newExperiment.createdAt,
                  lastRun:newExperiment.updatedAt,
                  url: `/dashboard/${newProjectId}/${newExperiment.name}`,
              }

                setExpriments((prevExperiments) => [...prevExperiments, {
                    title: newExperiment.name,
                    id: newExperiment._id,
                    projectId: newProjectId,
                    createdAt: newExperiment.createdAt,
                    lastRun:newExperiment.updatedAt,
                    url: `/dashboard/${newProjectId}/${newExperiment.name}`,
                }]);

                console.log("Created new project and experiment.");
                return newExperimentData;
            }
        }
    } catch (err) {
        setError("Failed to add new project or experiment.");
        console.error("Error adding project/experiment:", err);
    } finally {
        setLoading(false);
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
      console.log("Fetch Projects Called")
      fetchProjects();
    }
  }, [user?.id, selectedPipeline]);

  useEffect(() => {
    if (user && user?.id && projectId) {
      console.log("Fetch Project Experiments Called")
      fetchProjectExperiments();
    }
  }, [user?.id, projectId]);

  return (<ProjectsContext.Provider value={{
    projects,
    loading,
    error,
    experiments,
    setProjectId,
    selectedPipeline,
    setCurrentPipeline,
    fetchProjects,
    fetchProjectExperiments,
    addNewProject,
    deleteProject,
  }}>
    {children}
  </ProjectsContext.Provider >
    )
};

export const useProjects = ()=>useContext(ProjectsContext)
