"use client"

import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { ProtectedRoute } from "@/components/ProtectedRoute";

export default function DashboardLayout({ children }) {
  return (
    <>
      <ProtectedRoute>
      <SidebarProvider>
        <AppSidebar />
        <main className="w-screen h-screen flex">{children}</main> 
      </SidebarProvider>
      </ProtectedRoute>
    </>
  );
}
