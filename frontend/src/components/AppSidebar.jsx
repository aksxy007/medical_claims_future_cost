"use client"

import * as React from "react"
import {
  AudioWaveform,
  BookOpen,
  Bot,
  Command,
  Frame,
  GalleryVerticalEnd,
  Map,
  PieChart,
  Settings2,
  SquareTerminal,
} from "lucide-react"

import { NavMain } from "@/components/NavCustomMenu"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar"
import { NavUser } from "./SideBarUser"
import { TeamSwitcher } from "./ModellingTypeSwitch"
import { useAuth } from "@/hooks/use-auth"
import { Separator } from "./ui/separator"
import { ProjectsProvider } from "@/hooks/use-projects"

// This is sample data.
const data = {
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
  ]
}

export function AppSidebar({ ...props }) {
  const {user,token} = useAuth()

  return (
      <Sidebar collapsible="None" {...props}>
      <SidebarHeader>
        <TeamSwitcher teams={data.teams} />
      </SidebarHeader>
      <Separator className="my-2"/>
      <SidebarContent>
        <NavMain/>
      </SidebarContent>
      <Separator className="my-2"/>
      <SidebarFooter>
        <NavUser user={user} />
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
