"use client";

import { ChevronRight, Loader2, MoreHorizontal } from "lucide-react";

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
import { Separator } from "./ui/separator";
import { ScrollArea } from "./ui/scroll-area";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useProjects } from "@/hooks/use-projects";



export function NavMain({selectedPipeline}) {
  const {projects,loading,error,deleteProject} = useProjects(selectedPipeline)

  const router = useRouter()

  const handleDeleteProject = async (projectId)=>{
    try {
      deleteProject(projectId)
      console.log("Project Deleted Sucessfully!")
      if(!error){
        router.back()
      }
    } catch (error) {
      console.error("Error in deleting project",data.error)
    }
  }

  if (loading){
    return (
    <div className="flex justify-center items-center">
      <Loader2 />
    </div>
    )
  }

  return (
    <SidebarGroup className="overflow-hidden" key={projects.length}>
      <SidebarGroupLabel>Projects</SidebarGroupLabel>
      <Separator className="my-1" color="white"/>
      <SidebarMenu className="ml-auto">
        {projects.map((item) => (
          <Collapsible
            key={item.id}
            asChild
            defaultOpen={true}
            className="group/collapsible max-h-fit"
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
                      <DropdownMenuItem onClick = {()=>handleDeleteProject(item.id)}>
                        <span>Delete Project</span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </SidebarMenuButton>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <SidebarMenuSub className="w-full">
                  <ScrollArea className="max-h-[200px] w-full rounded-md pr-1">
                    {item.experiments?.map((subItem,index) => (
                       <SidebarMenuSubItem key={`${subItem.title}-${index}`} className="dark:hover:bg-[#27272A] hover:bg-[#f4f4f5] focus:[#27272A] rounded-sm">
                        <SidebarMenuSubButton
                          className="bg-sidebar-foreground"
                          asChild
                        >
                          <Link href={subItem.url}>
                            <span>{subItem.title}</span>
                          </Link>
                        </SidebarMenuSubButton>
                        <Separator className="my-1"/>
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
