"use client"

import React from 'react'

const Dashboard = ({children}) => {

  return (
    <div className='w-full h-screen flex flex-col justify-center items-center  space-y-7'>
        <h1 className='text-5xl text-black'>No runs to Display</h1>
        <h2 className='text-2xl text-black'>Select or create a project to get started...</h2>
        <div>
          {children}
        </div>
    </div>
  )
}

export default Dashboard