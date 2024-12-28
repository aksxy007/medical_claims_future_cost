import CodeEditor from "@/components/CodeEditorPage"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs"


const RunDeatils = ()=> {
  return (
    <div className="flex w-screenh h-screen">
      <div className="flex m-auto w-full h-screen rounded-md dark:bg-customSubBackground bg-gray-200">
      <Tabs defaultValue="codeeditor" className="w-full">
      <TabsList className="grid w-full grid-cols-3 p-2 h-fit text-black ">
        <TabsTrigger value="codeeditor">Code Editor</TabsTrigger>
        <TabsTrigger value="dagflow">Dag Flow</TabsTrigger>
        <TabsTrigger value="logs">Dag Logs</TabsTrigger>
      </TabsList>
      <TabsContent value="codeeditor">
        <CodeEditor/>
      </TabsContent>
      <TabsContent value="dagflow">
       Dag Flow
      </TabsContent>
      <TabsContent value="logs">
        Logs
      </TabsContent>
    </Tabs>
    </div>
    </div>
    
    
  )
}

export default RunDeatils;
