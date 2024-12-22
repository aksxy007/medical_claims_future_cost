"use client"

import { useAuth } from "@/hooks/useAuth";
import { Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";


export const ProtectedRoute =({children})=>{
    const {user, loading} = useAuth()
    const router = useRouter() 


    useEffect(()=>{
        if(!loading && !user){
            router.push("/login")
        }
    },[user,loading])

    if(loading){
        return (
            <div>
                <Loader2 />
            </div>
        )
    }

    return <>{children}</>
}