"use client"

import Link from 'next/link';  // Link component for navigation
import {Button} from '@/components/ui/button';  // ShadCN Button component
import { useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/use-auth';
import { useEffect } from 'react';


export default function Home() {

  const { user, loading } = useAuth();  // Get user and loading state from context
  const router = useRouter();  // To redirect the user to the dashboard if logged in

  const isAuthenticatedLogin= user ? "/dashboard":"/login"
  const isAuthenticateRegister = user ? "/dashbaord":"/register"

  return (
      <section className=" flex flex-col justify-center items-center h-screen px-20">
      <h1 className="text-7xl font-bold mb-4 dark:text-white text-black">Welcome to AutoML</h1>
      <p className="text-lg text-gray-400 mb-8">
        Build and explore machine learning models with ease.
      </p>
      <div className="flex space-x-4">
        <Link href={isAuthenticatedLogin}>
          <Button className="bg-customButton  dark:bg-customButton dark:text-white hover:dark:text-black text-white hover:bg-white hover:text-black" variant="secondary">Login</Button>  {/* Button to Login page */}
        </Link>
        <Link href={isAuthenticateRegister}>
          <Button className="bg-white text-black hover:bg-gray-300" variant="secondary">Register</Button>  {/* Button to Register page */}
        </Link>
      </div>
    </section>
    
  );
}
