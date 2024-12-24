"use client";

import {
  DropdownMenu,
  DropdownMenuItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/use-auth";
import { Mail, LogOut, ArrowDownIcon, User2, ChevronUp, } from "lucide-react";
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "./ui/sidebar";

// type Checked = DropdownMenuCheckboxItemProps["checked"]

export function SideBarUser() {
  const {user,logout } = useAuth();

  const handleLogout = async ()=>{
    try {
        await logout()
        console.log("User logged out successfully!")
    } catch (error) {
        console.error("Error in logging out",error)
    }
    
  }

  if (!user){
    return <></>
  }

  return (
    <SidebarMenu>
    <SidebarMenuItem>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <SidebarMenuButton className="flex items-center gap-2">
            <User2 size={20} />
            <span>Hi {user?.username || " Hi UNK"}</span>
            <ChevronUp className="ml-auto" />
          </SidebarMenuButton>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-56">
          <DropdownMenuLabel>Account</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem>
            <Mail />
            <span>{user?.email}</span>
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={handleLogout}>
            <LogOut />
            <span>Log Out</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  </SidebarMenu>
  );
}
