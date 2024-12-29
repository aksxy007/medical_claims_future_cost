"use client";

import { useEffect } from 'react';
import dynamic from 'next/dynamic';
import apiClient from '@/lib/api-client';
import { useCodeEditor } from '@/hooks/use-code-editor';
import { useTheme } from 'next-themes';
import { Button } from './ui/button';
import { useAuth } from '@/hooks/use-auth';

const Editor = dynamic(() => import('@monaco-editor/react'), { ssr: false });

const CodeEditor= ({projectId,experimentId}) => {
  console.log("ExperimentID",experimentId)
  const { editorValue, setEditorValue } = useCodeEditor();
  const {theme} = useTheme() 
  const {user} = useAuth()
  const editorTheme = theme==='dark'?'vs-dark':'vs'

  const fetchDefaultConfig = async () => {
    try {
      const response = await apiClient.get('/config/default-config');
      const defaultConfig = response.data.message;

      if (!editorValue) {
        setEditorValue(JSON.stringify(defaultConfig, null, '\t'));
      }
    } catch (error) {
      console.error('Error in getting config:', error);
    }
  };

  useEffect(() => {
    fetchDefaultConfig();
  }, []);

  const handleEditorTextChange = (newValue) => {
    setEditorValue(newValue || '');
  };

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

      const data =response.data

      console.log("Trigger run",data)
      // Make your API call to save the config here

    } catch (error) {
      console.error('Error parsing JSON:', error);
    }
  };

  return (
    <div className="flex justify-center items-start pt-1 h-full rounded-md">
      <div className="w-full p-4">
        <form onSubmit={handleSubmit}>
          <div className="max-w-screen">
            <Editor
              height="80vh"
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
          <div className="flex justify-end mt-2">
            <Button
              type="submit"

            >
              Run
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CodeEditor;
