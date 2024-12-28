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

export function AddProjectDialog({
  isDialogOpen,
  setIsDialogOpen,
  pipelines,
  selectedPipeline,
  setSelectedPipeline,
  newProjectName,
  setNewProjectName,
  addNewProject,
}) {

  console.log("isDialogOpenState",isDialogOpen)

  return (
    <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
      <DialogTrigger asChild>
        <div
          className="flex p-2 gap-2 cursor-pointer bg-background dark:hover:bg-[#27272A]"
          onClick={() => setIsDialogOpen(true)}
        >
          <div className="flex size-6 items-center justify-center rounded-md border bg-background">
            <Plus className="size-4" />
          </div>
          <div className="font-medium text-muted-foreground">Add Project</div>
        </div>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add New Project</DialogTitle>
        </DialogHeader>
        <DialogDescription>Choose a pipeline and add your project name</DialogDescription>
        {/* ShadCN Select for Pipeline Selection */}
        <Select value={selectedPipeline} onValueChange={setSelectedPipeline}>
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

        <DialogFooter>
          <Button onClick={() => setIsDialogOpen(false)}>Cancel</Button>
          <Button
            variant="default"
            onClick={addNewProject}
            className="bg-customButton"
          >
            Add
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
