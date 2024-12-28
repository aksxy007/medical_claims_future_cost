"use client";

import { useAuth } from "@/hooks/use-auth";
import { Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    console.log("Protected Route user",user)

    if (!loading && !user) {
      router.push("/login"); // If user is not logged in, redirect to login page
    }
  }, [user,loading]);

  if (loading) {
    return (
      <div className=" h-screen w-screen flex justify-center items-center">
        <Loader2 scale={20} color="white"/>
      </div>
    );
  }

  return <>{children}</>;
};
