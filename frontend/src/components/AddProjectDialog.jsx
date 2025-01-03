// components/AddProjectDialog.js
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Plus, PlusCircle } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { useState } from "react";
import apiClient from "@/lib/api-client";
import {useProjects} from "@/hooks/use-projects";
import { useRouter } from "next/navigation";

export function AddProjectDialog({
  isDialogOpen,
  setIsDialogOpen,
  newProjectName,
  setNewProjectName
}) {


  const pipelines = [
    {
        title:"Model Builds",
    },{
        title:"Production Runs",
    }]

  const {user} = useAuth()
  const [projectType,setProjectType] = useState(pipelines[0].title)
  const [newExprimentName,setNewExperimentName] = useState("")
  const {addNewProject,error,setCurrentPipeline} = useProjects()
  const router = useRouter()

  console.log("isDialogOpenState",isDialogOpen)

  const handleSetProjectType = (value)=>{
    setProjectType(value)
    // const projectType = value === "Model Builds"?"Modelling":"Production"
    // setCurrentPipeline(projectType)
  }

  const handleAddNewProject = async ()=>{
      try {
        const response = await addNewProject(newProjectName,newExprimentName)
        if(!error){
          console.log("Added new project/experiment successfully!")
          setIsDialogOpen(false)
          console.log(response)
          router.push(`/dashboard/${response.projectId}/${response.id}`)
        }else{
          console.error("error",error)
          setIsDialogOpen(false)
        }
      } catch (error) {
        console.error("Error in adding new project",error)
      }
  }

  return (
    <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
      {/* <DialogTrigger asChild>
        
      </DialogTrigger> */}
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add New Project</DialogTitle>
        </DialogHeader>
        <DialogDescription>Choose a pipeline and add your project name</DialogDescription>
        {/* ShadCN Select for Pipeline Selection */}
        <Select value={projectType} onValueChange={(value)=>handleSetProjectType(value)}>
          <SelectTrigger className="mb-4 w-full">
            <SelectValue placeholder="Select Pipeline" />
          </SelectTrigger>
          <SelectContent>
            {pipelines.map((pipeline) => (
              <SelectItem key={pipeline.title} value={pipeline.title}>
                {pipeline.title}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* ShadCN Input for Project Name */}
        <Input
          type="text"
          value={newProjectName}
          onChange={(e) => setNewProjectName(e.target.value)}
          placeholder="Enter project name"
          className="mb-4"
        />

        <Input
          type="text"
          value={newExprimentName}
          onChange={(e) => setNewExperimentName(e.target.value)}
          placeholder="Enter project name"
          className="mb-4"
        />

        <DialogFooter>
          <Button onClick={() => setIsDialogOpen(false)}>Cancel</Button>
          <Button
            variant="default"
            onClick={handleAddNewProject}
            className="bg-customButton  dark:bg-customButton dark:text-white hover:dark:text-black"
          >
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
