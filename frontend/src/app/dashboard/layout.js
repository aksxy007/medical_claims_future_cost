"use client"

import { SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";

export default function DashboardLayout({ children }) {
  return (
    <>
      <SidebarProvider>
        <AppSidebar />
        <main className="w-screen h-screen flex">{children}</main> 
      </SidebarProvider>
    </>
  );
}
