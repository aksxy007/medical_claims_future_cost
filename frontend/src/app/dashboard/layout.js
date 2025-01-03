"use client";

import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import { CodeEditorProvider } from "@/hooks/use-code-editor";
import { Separator } from "@/components/ui/separator";
import { ProjectsProvider } from "@/hooks/use-projects";

export default function DashboardLayout({ children }) {
  return (
    <>
      <ProtectedRoute>
        <SidebarProvider>
          <ProjectsProvider>
            <AppSidebar />
            <SidebarInset>
              <CodeEditorProvider>
                <main className="bg-gray-200 dark:bg-customPageBackground w-full h-full">{children} </main>
              </CodeEditorProvider>
            </SidebarInset>
          </ProjectsProvider>
        </SidebarProvider>
      </ProtectedRoute>
    </>
  );
}
