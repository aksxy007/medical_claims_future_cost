"use client"

import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { CodeEditorProvider } from "@/hooks/use-code-editor";
import { Separator } from "@/components/ui/separator";
import { ProjectsProvider } from "@/hooks/use-projects";


export default function DashboardLayout({ children }) {
  return (
    <>
      <ProtectedRoute>
        <ProjectsProvider>
        <SidebarProvider>
        <AppSidebar />
        <SidebarInset>
        <CodeEditorProvider>
          <main>{children} </main>
        </CodeEditorProvider>
        
      </SidebarInset>
      </SidebarProvider>
        </ProjectsProvider>
       
      </ProtectedRoute>
    </>
  );
}
