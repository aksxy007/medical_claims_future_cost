"use client";

import CodeEditor from "@/components/CodeEditorPage";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { usePathname } from "next/navigation";
import { useState } from "react";

const RunDetails = () => {

  const [tab, setTab] = useState("codeeditor");

  const onTabChange = (value) => {
    setTab(value);
  }

  const pathname = usePathname();
  const projectId = pathname.split("/")[2];
  const experimentId = pathname.split("/")[3];


  console.log("Project ID:", projectId, "Experiment ID:", experimentId);

  return (
    <div className="flex w-full h-full justify-center overflow-hidden">
      <div className="flex w-full h-full">
        <Tabs defaultValue="codeeditor" value={tab} onValueChange={onTabChange} className="w-full h-full">
          <TabsList className="grid w-full grid-cols-3 p-2 h-fit text-black">
            <TabsTrigger value="codeeditor">Code Editor</TabsTrigger>
            <TabsTrigger value="dagflow">Dag Flow</TabsTrigger>
            <TabsTrigger value="logs">Dag Logs</TabsTrigger>
          </TabsList>
          <TabsContent value="codeeditor" className="h-full relative">
            <div className="absolute inset-0 flex flex-col h-full w-full">
              <CodeEditor projectId={projectId} experimentId={experimentId} setTab={onTabChange}/>
            </div>
          </TabsContent>
          <TabsContent value="dagflow" className="h-full">
            Dag Flow
          </TabsContent>
          <TabsContent value="logs" className="h-full">
            Logs
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default RunDetails;
