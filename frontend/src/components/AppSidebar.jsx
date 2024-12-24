import {
  BookOpen,
  Bot,
  ChevronUp,
  LogOut,
  Mail,
  Settings2,
  SquareTerminal,
  User2,
} from "lucide-react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

import { NavMain } from "./NavCusotmMenu";
import { SideBarUser } from "./SideBarUser";

// Menu items.
const data = {
  navMain: [
    {
      title: "Modelling Pipeline",
      url: "#",
      icon: SquareTerminal,
      isActive: true,
      items: [
            {
              title: "Project Name 1",
              items: [
                {
                  title: "Project Run Name 1",
                },
                {
                  title: "Project Run Name 2",
                },
              ],
            },
            {
              title: "Project Name 2",
              items: [
                {
                  title: "Project Run Name 1",
                },
              ],
        },
      ],
    },
    {
      title: "Production Pipeline",
      url: "#",
      icon: Bot,
      items: [
        {
          title: "Deployed Projects",
          url: "#",
          items: [
            {
              title: "Deployed Project 1",
              url: "#",
            },
          ],
        },
      ],
    },
  ],
};


export function AppSidebar() {

  return (
    <Sidebar className="pt-4 h-full text-white flex items-center justify-center bg-customBackground">
      {/* Sidebar Content */}
      <SidebarContent>
      <NavMain items={data.navMain}/>
      </SidebarContent>
        

      {/* Sidebar Footer */}
      <SidebarFooter className="flex-shrink-0 p-4 border-t border-gray-700">
        <SideBarUser/>
      </SidebarFooter>
    </Sidebar>
  );
}
