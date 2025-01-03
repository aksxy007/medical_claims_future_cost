"use client"

import * as React from "react"
import { AudioWaveform, Bot, ChevronRight, Loader2, Plus } from "lucide-react"

import { SearchForm } from "@/components/search-form"
import { VersionSwitcher } from "@/components/version-switcher"
import {TeamSwitcher} from "@/components/ModellingTypeSwitch"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"

import { Separator } from "./ui/separator"
import { NavUser } from "./SideBarUser"
import { useAuth } from "@/hooks/use-auth"
import { NavMain } from "./NavCustomMenu"
import { useProjects } from "@/hooks/use-projects"
import Link from "next/link"
import { ScrollArea } from "./ui/scroll-area"
import { AddProjectDialog } from "./AddProjectDialog"

// This is sample data.


export function AppSidebar({
  ...props
}) {

  const {user} = useAuth()
  const {projects,loading} = useProjects()

  const [newProjectName,setNewProjectName] = React.useState("")
  const [isDialogOpen,setIsDialogOpen] = React.useState(false)

  if (loading){
    return (
    <div className="flex justify-center items-center">
      <Loader2 />
    </div>
    )
  }




  const data = {
    versions: ["1.0.1", "1.1.0-alpha", "2.0.0-beta1"],
    teams: [
      {
        name: "Modelling Runs",
        logo: Bot,
        plan: "Automated Model Training",
      },
      {
        name: "Production Runs",
        logo: AudioWaveform,
        plan: "Automate Production Run",
      }
    ],
    navMain: [
      {
        title: "Getting Started",
        url: "#",
        items: [
          {
            title: "Quick Guide",
            url: "#",
          },
        ],
      },
      {
        title:"Your Projects",
        url:"#",
        items:projects
      },
      {
        title: "Building Your Application",
        url: "#",
        items: [
          {
            title: "Routing",
            url: "#",
          },
          {
            title: "Data Fetching",
            url: "#",
            isActive: true,
          },
          {
            title: "Rendering",
            url: "#",
          },
          {
            title: "Caching",
            url: "#",
          },
          {
            title: "Styling",
            url: "#",
          },
          {
            title: "Optimizing",
            url: "#",
          },
          {
            title: "Configuring",
            url: "#",
          },
          {
            title: "Testing",
            url: "#",
          },
          {
            title: "Authentication",
            url: "#",
          },
          {
            title: "Deploying",
            url: "#",
          },
          {
            title: "Upgrading",
            url: "#",
          },
          {
            title: "Examples",
            url: "#",
          },
        ],
      },
      {
        title: "API Reference",
        url: "#",
        items: [
          {
            title: "Components",
            url: "#",
          },
          {
            title: "File Conventions",
            url: "#",
          },
          {
            title: "Functions",
            url: "#",
          },
          {
            title: "next.config.js Options",
            url: "#",
          },
          {
            title: "CLI",
            url: "#",
          },
          {
            title: "Edge Runtime",
            url: "#",
          },
        ],
      },
      {
        title: "Architecture",
        url: "#",
        items: [
          {
            title: "Accessibility",
            url: "#",
          },
          {
            title: "Fast Refresh",
            url: "#",
          },
          {
            title: "Next.js Compiler",
            url: "#",
          },
          {
            title: "Supported Browsers",
            url: "#",
          },
          {
            title: "Turbopack",
            url: "#",
          },
        ],
      },
      {
        title: "Community",
        url: "#",
        items: [
          {
            title: "Contribution Guide",
            url: "#",
          },
        ],
      },
    ],
  }
  console.log(projects)

  return (
    (<Sidebar {...props}>
      <SidebarHeader>
        <TeamSwitcher  teams={data.teams} />
        <div className="flex mt-1"></div>
        <SearchForm />
      </SidebarHeader>
      <Separator className="my-2"/>
      <SidebarContent className="gap-0">
        {/* We create a collapsible SidebarGroup for each parent. */}
        {/* <NavMain/> */}
        <ScrollArea className="h-full">
       {data.navMain.map((item) => (
          <Collapsible
            key={item.title}
            title={item.title}
            defaultOpen
            className="group/collapsible">
            <SidebarGroup>
              <SidebarGroupLabel
                asChild
                className="group/label text-sm text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground">
                <CollapsibleTrigger className="flex justify-between">
                  {item.title}
                  <div className="flex justify-center items-center">
                  { item.title =="Your Projects" ?
                      <Plus className="h-4 w-4 " onClick={()=>setIsDialogOpen(true)}/>
                    :<></>}
                  <ChevronRight
                    className="ml-auto h-4 w-4 transition-transform group-data-[state=open]/collapsible:rotate-90" />
                  </div>
                  
                </CollapsibleTrigger>
              </SidebarGroupLabel>
              <CollapsibleContent>
                <SidebarGroupContent>
                <ScrollArea className=" rounded-md pr-1">
                  <SidebarMenu>
                    {item.items.map((item) => (
                      <SidebarMenuItem key={item.title} className="ml-1">
                        <SidebarMenuButton asChild isActive={item.isActive}>
                          <Link href={`/dashboard/${item.id}`}>{item.title}</Link>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                      ))}
                     
                  </SidebarMenu>
                  </ScrollArea>
                </SidebarGroupContent>
                
              </CollapsibleContent>
            </SidebarGroup>
          </Collapsible>
        ))}
      <AddProjectDialog isDialogOpen={isDialogOpen} setIsDialogOpen={setIsDialogOpen} newProjectName={newProjectName} setNewProjectName={setNewProjectName}/>            
      </ScrollArea>
      </SidebarContent>
      <Separator className="my-2"/>
      <SidebarFooter>
          <NavUser user={user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>)
    
  );
}
