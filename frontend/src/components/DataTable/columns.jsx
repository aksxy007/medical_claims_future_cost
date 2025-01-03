import { CheckCircle, MoreHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import Link from "next/link";

export const columns = [
  {
    accessorKey: "title",
    header: "Experiment Title",
  },
  {
    accessorKey: "createdAt",
    header: "Created At",
  },
  {
    accessorKey: "lastRun",
    header: "Last Run",
  },
  {
    accessorKey: "url",
    header: "View",
    cell: ({ row }) => (
      <Link href={row.getValue("url")} className="text-blue-500">
        View Details
      </Link>
    ),
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: () => (
      <CheckCircle className="text-green-500 w-5 h-5" />
    ),
  },
  {
    id: "details",
    header: "Details",
    cell: ({ row }) => (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon">
            <MoreHorizontal className="w-5 h-5" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuItem onClick={() => console.log("View:", row.original)}>
            View
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => console.log("Delete:", row.original)}>
            Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    ),
  },
];
