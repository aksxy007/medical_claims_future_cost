// "use client";

// import {
//   DropdownMenu,
//   DropdownMenuItem,
//   DropdownMenuContent,
//   DropdownMenuLabel,
//   DropdownMenuSeparator,
//   DropdownMenuTrigger,
// } from "@/components/ui/dropdown-menu";
// import { useAuth } from "@/hooks/use-auth";
// import { Mail, LogOut, ArrowDownIcon, User2, ChevronUp, Sun, Moon, } from "lucide-react";
// import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from "./ui/sidebar";
// import { useTheme } from "next-themes";
// import { Button } from "./ui/button";
// import { cn } from "@/lib/utils";

// // type Checked = DropdownMenuCheckboxItemProps["checked"]

// export function SideBarUser() {
//   const {user,logout } = useAuth();
//   const { setTheme, theme} = useTheme()
//   const handleLogout = async ()=>{
//     try {
//         await logout()
//         console.log("User logged out successfully!")
//     } catch (error) {
//         console.error("Error in logging out",error)
//     }
    
//   }

//   if (!user){
//     return <></>
//   }

//   return (
//     <SidebarMenu className="flex flex-row justify-between">
//     <SidebarMenuItem>
//       <DropdownMenu>
//         <DropdownMenuTrigger asChild>
//           <SidebarMenuButton className="flex items-center gap-2 justify-between">
//             <User2 size={20} />
//             <span className="">{user?.username}</span>
//             <ChevronUp className="ml-auto" />
//           </SidebarMenuButton>
//         </DropdownMenuTrigger>
//         <DropdownMenuContent className="w-56">
//           <DropdownMenuLabel>Account</DropdownMenuLabel>
//           <DropdownMenuSeparator />
//           <DropdownMenuItem>
//             <Mail />
//             <span>{user?.email}</span>
//           </DropdownMenuItem>
//           <DropdownMenuItem onSelect={handleLogout}>
//             <LogOut />
//             <span>Log Out</span>
//           </DropdownMenuItem>
//         </DropdownMenuContent>
//       </DropdownMenu>
//     </SidebarMenuItem>
//     <SidebarMenuItem>
//     <DropdownMenu>
//       <DropdownMenuTrigger asChild>
//         <Button variant="outline" size="icon">
//           <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
//           <Moon className="absolute h-[1.2rem] text-white w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
//           <span className="sr-only">Toggle theme</span>
//         </Button>
//       </DropdownMenuTrigger>
//       <DropdownMenuContent align="end">
//         <DropdownMenuItem onClick={() => setTheme("light")}>
//           Light
//         </DropdownMenuItem>
//         <DropdownMenuItem onClick={() => setTheme("dark")}>
//           Dark
//         </DropdownMenuItem>
//         <DropdownMenuItem onClick={() => setTheme("system")}>
//           System
//         </DropdownMenuItem>
//       </DropdownMenuContent>
//     </DropdownMenu>
//     </SidebarMenuItem>
//   </SidebarMenu>
//   );
// }


"use client"

import {
  BadgeCheck,
  Bell,
  ChevronsUpDown,
  CreditCard,
  LogOut,
  Sparkles,
} from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"
import { useAuth } from "@/hooks/use-auth"
import { ModeToggle } from "./ModeToggel"

export function NavUser({
  user,
}) {
  const { isMobile } = useSidebar()
  const {logout} = useAuth()
  const handleLogout = async ()=>{
        try {
            await logout()
            console.log("User logged out successfully!")
        } catch (error) {
            console.error("Error in logging out",error)
        }
        
      }
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton
              size="xl"
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
            >
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-semibold">{user?.username}</span>
                <span className="truncate text-xs">{user?.email}</span>
              </div>
              <ChevronsUpDown className="ml-auto size-4" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
            side={isMobile ? "bottom" : "right"}
            align="end"
            sideOffset={4}
          >
            <DropdownMenuLabel className="p-0 font-normal">
              <div className="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">{user?.username}</span>
                  <span className="truncate text-xs">{user?.email}</span>
                </div>
                <div>
                  <ModeToggle/>
                </div>
              </div>
            </DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem>
                <Sparkles />
                Upgrade to Pro
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuGroup>
              <DropdownMenuItem>
                <BadgeCheck />
                Account
              </DropdownMenuItem>
              <DropdownMenuItem>
                <CreditCard />
                Billing
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Bell />
                Notifications
              </DropdownMenuItem>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={handleLogout}>
              <LogOut  />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
