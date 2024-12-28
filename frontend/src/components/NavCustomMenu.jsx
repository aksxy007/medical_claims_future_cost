"use client";

import { ChevronRight, MoreHorizontal } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { Separator } from "@radix-ui/react-dropdown-menu";
import { ScrollArea } from "./ui/scroll-area";
import Link from "next/link";
import { useEffect, useState } from "react";
import apiClient from "@/lib/api-client";
import { useAuth } from "@/hooks/use-auth";



export function NavMain() {

  const {user}  = useAuth()
  const [items,setItems] = useState(null)

  const fetchUserProjects = async ()=>{
    try {
      const response = await apiClient.get(`/projects/get-user-projects`,{
        params:{
          userId:user.id
        }
      })
      console.log(response.data)
      const data = response.data?.data
      setItems([...data]);
    } catch (error) {
      console.log("error fetching user data",error)
    }
  }

  useEffect(()=>{
    fetchUserProjects()
  },[])

  return (
    <SidebarGroup className="overflow-hidden">
      <SidebarGroupLabel>Projects</SidebarGroupLabel>
      <SidebarMenu className="ml-auto">
        {items?.map((item) => (
          <Collapsible
            key={item.title}
            asChild
            // defaultOpen={item.isActive}
            className="group/collapsible"
          >
            <SidebarMenuItem>
              <CollapsibleTrigger asChild>
                <SidebarMenuButton
                  className="dark:hover:bg-[#27272A] hover:bg-[#f4f4f5]"
                  tooltip={item.title}
                >
                  {item.icon && <item.icon className="size-6" />}
                  <span className="text-lg">{item.title}</span>
                  <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <SidebarMenuAction>
                        <MoreHorizontal />
                      </SidebarMenuAction>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent side="right" align="start">
                      <DropdownMenuItem>
                        <span>Edit Project</span>
                      </DropdownMenuItem>
                      <DropdownMenuItem>
                        <span>Delete Project</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub className="w-full">
                  <ScrollArea className="h-screen w-full rounded-md pr-1">
                    {item.experiments?.map((subItem,index) => (
                       <SidebarMenuSubItem key={`${subItem.title}-${index}`} className="dark:hover:bg-[#27272A] hover:bg-[#f4f4f5] focus:[#27272A]">
                       <Separator className="my-2"/>
                        <SidebarMenuSubButton
                          className="bg-sidebar-foreground"
                          asChild
                        >
                          <Link href={subItem.url}>
                            <span>{subItem.title}</span>
                          </Link>
                        </SidebarMenuSubButton>
                      </SidebarMenuSubItem>
                    ))}
                  </ScrollArea>
                </SidebarMenuSub>
              </CollapsibleContent>
            </SidebarMenuItem>
          </Collapsible>
        ))}
      </SidebarMenu>
    </SidebarGroup>
  );
}
