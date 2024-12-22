"use client"

import Link from 'next/link';  // Link component for navigation
import {Button} from '@/components/ui/button';  // ShadCN Button component


export default function Home() {
  return (
      <section className="bg-customBackground flex flex-col justify-center items-center h-screen px-20">
      <h1 className="text-7xl font-bold mb-4 text-white">Welcome to AutoML</h1>
      <p className="text-lg text-gray-200 mb-8">
        Build and explore machine learning models with ease.
      </p>
      <div className="flex space-x-4">
        <Link href="/login">
          <Button className="bg-customButton hover:bg-black" variant="default">Login</Button>  {/* Button to Login page */}
        </Link>
        <Link href="/register">
          <Button className="bg-white hover:bg-black text-black hover:text-white" variant="default">Register</Button>  {/* Button to Register page */}
        </Link>
      </div>
    </section>
    
  );
}
