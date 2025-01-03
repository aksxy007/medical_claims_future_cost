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
import { ClipboardCopyIcon, Plus, PlusCircle } from "lucide-react";
import { useAuth } from "@/hooks/use-auth";
import { useState } from "react";
import apiClient from "@/lib/api-client";
import { useProjects } from "@/hooks/use-projects";
import { useRouter } from "next/navigation";
import { useToast } from "@/hooks/use-toast";
import { Progress } from "./ui/progress";

export function FileUploadDialog({
  isOpenFileUploadDialog,
  setIsOpenFileUploadDialog,
  experimentId
}) {
  const DatasetType = ["Modelling Data", "Scoring Data"];

  const { user } = useAuth();
  const [datasetType, setDatasetType] = useState(DatasetType[0]);
  const [file, setFile] = useState(null);
  const [uploadState, setUploadState] = useState("choose"); //"choose","uploading","done"

  const [uploadedFileUrl,setUploadFileUrl] = useState("s3://dummy_file_location");

  const [progress, setProgress] = useState(10);
  const { showToast } = useToast();
  const router = useRouter();

  console.log("isFileUploadDialogOpenState", isOpenFileUploadDialog);

  const handleSetDatasetType = (value) => {
    setDatasetType(value);
  };

  const handleFileChange = (event) => {
    try {
      const selectFile = event.target.files[0];

      if (selectFile) {
        const fileType = selectFile.type;

        if (fileType !== "text/csv") {
          showToast({ message: "Upload only csv type files!", type: "error" });
          event.target.value = null;
          setUploadState("choose");
          return;
        }
        setFile(selectFile);
      }
      console.log("Uploaded file:", file);
    } catch (error) {
      console.error("Error uploading file!!");
      showToast({ message: "Error uploading file", type: "error" });
      setUploadState("choose");
    }
  };

  const handleFileUpload = async () => {
    try {
      setUploadState("uploading");
      let uploadProgress = 0;

      // Simulate file upload progress
      const interval = setInterval(() => {
        uploadProgress += 10; // Increment progress
        setProgress(uploadProgress);

        if (uploadProgress >= 100) {
          clearInterval(interval);
          setUploadState("done")
        }
      }, 500);



    } catch (error) {
      console.log("Error in uploading file");
      showToast({ message: "Error in uploading file", type: "error" });
      setFile(null)
      setUploadState("choose");
    }
  };

  const handleCopy = ()=>{
    navigator.clipboard.writeText(uploadedFileUrl).then(()=>{
        showToast({message:"File path copied!",type:"success"})
    }).catch((error)=>{
        showToast({message:"Could not copy file path! try again",type:"error"})
    })
  }

  return (
    <Dialog
      open={isOpenFileUploadDialog}
      onOpenChange={setIsOpenFileUploadDialog}
    >
      {/* <DialogTrigger asChild>
          
        </DialogTrigger> */}
      <DialogContent>
        {uploadState === "choose" && (
          <>
            <DialogHeader>
              <DialogTitle>Upload Your Dataset</DialogTitle>
            </DialogHeader>
            <DialogDescription>Choose Dataset Type</DialogDescription>
            {/* ShadCN Select for Pipeline Selection */}
            <Select
              value={datasetType}
              onValueChange={(value) => handleSetDatasetType(value)}
            >
              <SelectTrigger className="mb-4 w-full">
                <SelectValue placeholder="Select Pipeline" />
              </SelectTrigger>
              <SelectContent>
                {DatasetType.map((item) => (
                  <SelectItem key={item} value={item}>
                    {item}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {/* React DropZone */}
            <DialogDescription>Choose dataset file </DialogDescription>
            <Input
              type="file"
              name="dataset"
              accept=".csv"
              multiple={false}
              onChange={handleFileChange}
            />

            <DialogFooter>
              <Button onClick={() => setIsOpenFileUploadDialog(false)}>
                Cancel
              </Button>
              <Button
                variant="default"
                className="bg-customButton  dark:bg-customButton dark:text-white hover:dark:text-black"
                onClick={handleFileUpload}
                disabled={!file}
              >
                Upload Data
              </Button>
            </DialogFooter>
          </>
        )}

        {/* Uploading State */}

        {uploadState === "uploading" && (
          <>
            <DialogHeader>
              <DialogTitle>Uploading Your Dataset ...</DialogTitle>
            </DialogHeader>
            <DialogDescription>File: {file?.name}</DialogDescription>
            <DialogDescription>
              Size: {(file?.size / 1024).toFixed(2)}KB
            </DialogDescription>
            <Progress value={progress} />
          </>
        )}

        {uploadState === "done" && (
            <>
            <DialogHeader>
              <DialogTitle>Dataset file uploaded</DialogTitle>
            </DialogHeader>
            <DialogDescription>File: {file?.name}</DialogDescription>
            <DialogDescription>
              Size: {(file?.size / 1024).toFixed(2)}KB
            </DialogDescription>
            <DialogDescription>Copy the url and paste in local file in config</DialogDescription>
            <div className="flex w-full justify-between border border-gray-400 rounded-sm p-1" >
            <p className="">
                File path: <span>{uploadedFileUrl}</span>
            </p>
                <ClipboardCopyIcon className="rounded-md hover:text-gray-400" onClick={handleCopy}/>
            </div>
            <DialogFooter>
                <Button className="bg-customButton  dark:bg-customButton dark:text-white hover:dark:text-black" 
                    onClick={() => {
                        setProgress(0)
                        setFile(null)
                        setUploadFileUrl("s3://dummy_file_location")
                        setUploadState("choose")
                        setIsOpenFileUploadDialog(false)
                    }}
                >
                    Done
                </Button>
            </DialogFooter>
            
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
