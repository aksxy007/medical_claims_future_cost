"use client"

import { columns } from '@/components/DataTable/columns'
import { DataTable } from '@/components/DataTable/data-table'
import { useProjects } from '@/hooks/use-projects'
import { Loader2 } from 'lucide-react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useRef } from 'react'

const ProjectExperiments = () => {

    const {loading,projectId,experiments,setProjectId} = useProjects()
    const hasMounted = useRef(false);

    const pathname = usePathname()
    // const projectId = pathname.split("/")[2]
    const currentProjectId = pathname.split("/")[2];
    console.log("project Id in project page", currentProjectId)

   

    useEffect(() => {
        if (!hasMounted.current) {
          hasMounted.current = true;
          return;
        }
    
        if (currentProjectId !== projectId) {
          setProjectId(currentProjectId);
        }
      }, [currentProjectId, projectId]);


    console.log("Experiments",experiments)


    if(loading){
        return(
            <div className='w-full h-full flex justify-center items-center'>
                <Loader2 size={20} />
            </div>
        )
    }


  return (
    <div className='flex flex-col w-full h-full m-auto p-10 space-y-2'>
        <header className='text-start text-xl'>
            Your Experiments
        </header>
        <DataTable columns={columns} data={experiments}/>
    </div>
  )
}

export default ProjectExperiments