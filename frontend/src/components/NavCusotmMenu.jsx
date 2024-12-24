"use client";

import { ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar";
import Link from "next/link";

export function NavMain({ items }) {
  const renderItems = (items) => {
    return items.map((item) => {
      return (
        <SidebarMenuItem key={item.title} className="p-0">
          {item.items ? (
            <Collapsible asChild defaultOpen={item.isActive} className="group/collapsible">
              <div>
                <CollapsibleTrigger asChild>
                  <SidebarMenuButton className="flex items-center gap-2 hover:bg-gray-700 focus:bg-gray-700">
                    {item.icon && <item.icon className="w-4 h-4" />}
                    <span>{item.title}</span>
                    <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                  </SidebarMenuButton>
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <SidebarMenuSub className="pl-1 ml-0 border-l-0 space-y-1">
                    {renderItems(item.items)}
                  </SidebarMenuSub>
                </CollapsibleContent>
              </div>
            </Collapsible>
          ) : (
            <SidebarMenuSubItem>
              <SidebarMenuSubButton asChild>
                <Link
                  href={`/dashboard/${item.title}`}
                  className="flex items-center gap-2 hover:bg-customButton focus:bg-customButton"
                >
                  {item.icon && <item.icon className="w-4 h-4" />}
                  <span>{item.title}</span>
                </Link>
              </SidebarMenuSubButton>
            </SidebarMenuSubItem>
          )}
        </SidebarMenuItem>
      );
    });
  };

  return (
    <SidebarGroup className="flex flex-col items-start space-y-4 p-4">
      <SidebarGroupLabel className="text-2xl text-gray-400 font-bold">AutoML</SidebarGroupLabel>
      <SidebarMenu>{renderItems(items)}</SidebarMenu>
    </SidebarGroup>
  );
}
