"use client";

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import apiClient from '@/lib/api-client';
import { useCodeEditor } from '@/hooks/use-code-editor';
import { useTheme } from 'next-themes';
import { Button } from './ui/button';
import { useAuth } from '@/hooks/use-auth';
import { useToast } from '@/hooks/use-toast';
import { FileUploadDialog } from './FileUploader';

const Editor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

const CodeEditor= ({projectId,experimentId,setTab}) => {
  console.log("ExperimentID",experimentId)
  const [isOpenFileUploadDialog,setIsOpenFileUploadDialog] = useState(false)
  const { editorValue, setEditorValue } = useCodeEditor();
  const {theme} = useTheme() 
  const {user} = useAuth()
  const editorTheme = theme==='dark'?'vs-dark':'vs'
  const {showToast} = useToast()

  const fetchDefaultConfig = async () => {
    try {
      const response = await apiClient.post('/config/default-config',{
        experimentId
      });
      const defaultConfig = response.data?.message;

      // if (!editorValue) {
      setEditorValue(JSON.stringify(defaultConfig, null, '\t'));
      // }
    } catch (error) {
      console.error('Error in getting config:', error);
    }
  };

  useEffect(() => {
    console.log("Fetching experiment config")
    fetchDefaultConfig();
  }, [experimentId]);

  const handleEditorTextChange = (newValue) => {
    setEditorValue(newValue || '');
  };

  const handleFileUpload = ()=>{
    setIsOpenFileUploadDialog(true)
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const updatedConfig = JSON.parse(editorValue);
      console.log('Updated config:', updatedConfig);

      const response = await apiClient.post("/run/trigger-run",{
        runConfig:JSON.stringify(updatedConfig),
        userId:user.id,
        projectId:projectId,
        experimentId:experimentId

      })

      const data =response.data.message
      if(response.status===200){
        showToast({message:"Triggered experiment",type:"success"})

        setTimeout(()=>{
          setTab(()=>"dagflow")
        },501)
      }
        
      console.log("Trigger run",data)
      // Make your API call to save the config here

    } catch (error) {
      console.error('Error parsing JSON:', error);
      showToast({message:"Triggered experiment failed",type:"error"})
    }
  };

  return (
        <form onSubmit={handleSubmit} className='flex flex-col gap-2 w-full h-full p-4 my-4'>
          <div className="flex-1 h-[90%] overflow-hidden rounded-md ">
            <Editor
              height="100%"
              width="100%"
              defaultLanguage="json"
              value={editorValue}
              onChange={handleEditorTextChange}
              theme={editorTheme}
              options={{
                readOnly: false,
                lineNumbers: 'on',
                fontSize: 16,
                automaticLayout: true,
                minimap: { enabled: true },
                scrollBeyondLastLine: false,
                wordWrap: 'off',
                tabSize: 4,
                cursorStyle: 'line',
                formatOnPaste: true,
              }
              
            }
            
            />
          </div>
          <div className="flex h-[10%] justify-end gap-3 mt-2">
            <Button
              type="submit"
            >
              Run
            </Button>
            <Button
              type="button"
              variant={"default"}
              className="bg-customButton dark:bg-customButton dark:text-white hover:dark:text-black"
              onClick = {handleFileUpload}
            >
              Upload Dataset
            </Button>
          </div>
          <FileUploadDialog isOpenFileUploadDialog={isOpenFileUploadDialog} setIsOpenFileUploadDialog={setIsOpenFileUploadDialog} experimentId={experimentId}/>
        </form>
  );
};

export default CodeEditor;
