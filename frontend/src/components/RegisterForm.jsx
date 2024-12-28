"use client";

import Link from "next/link";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Label } from "./ui/label";
import apiClient from "@/lib/api-client";
import { useRouter } from "next/navigation";


// Define your form schema using Zod
const formSchema = z.object({
  username: z.string().min(2).max(50),
  email: z.string().email().min(1, { message: "Kindly enter a valid email.." }),
  password: z
    .string()
    .min(4, { message: "Password must have at least 4 characters" })
    .max(10, { message: "Password cannot have more than 10 characters" }),
});

export const RegisterForm = () => {
  const router = useRouter()

  const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      username: "",
      email: "",
      password: "",
    },
  });

  const handleSubmit = async (formData)=>{
    try {
        const response = await apiClient.post("/auth/register",formData)
        const userData = response.data
        console.log("User registered successfully!")
        console.log("User Data: ",userData)
        router.push('/dashboard')
    } catch (error) {
        console.error("Error in registration: ",error)
    }
        
  }

  return (
    <div className=" w-[30%] flex flex-col justify-center items-center m-4 rounded-3xl p-10 border-2 border-gray-700 shadow-lg">
      <div className="flex p-5 mb-5">
        <h1 className="text-3xl ">
          Register
        </h1>
      </div>
      <Form {...form}>
        <form  onSubmit = {form.handleSubmit(handleSubmit)} className="w-[80%] space-y-6">
          {/* Username Field */}
          <FormField
            control={form.control}
            name="username"
            render={({ field }) => (
              <FormItem>
                <FormLabel className=" text-lg" > Username</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Enter your username"
                    {...field}
                    className="w-full h-12 p-4 bg-transparent border-gray-700  rounded-md focus:outline-none hover:border-gray-500"
                  />
                </FormControl>
                <FormDescription className="text-gray-400">This is your public display name.</FormDescription>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Email Field */}
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel className=" text-lg">Email</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Your Email..."
                    {...field}
                    className="w-full h-12 p-4 bg-transparent border-gray-700  rounded-md focus:ring-2 outline-none hover:border-gray-500"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Password Field */}
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <FormItem>
                <FormLabel className=" text-lg">Password</FormLabel>
                <FormControl>
                  <Input
                    type="password"
                    placeholder="Your Password..."
                    {...field}
                    className="w-full h-12 p-4 bg-transparent border-gray-700  outline-none rounded-md focus:ring-2 hover:border-gray-500"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Submit Button */}
          <Button type="submit" className="w-full h-10 mt-4 py-3 rounded-md">
            Submit
          </Button>
          <div className=" flex text-sm  mt-4 justify-center items-end">
            Already have an account? 
            <Link href={'/login'} className="text-customButton">
                Sign In
            </Link>
          </div>
          
        </form>
      </Form>
    </div>
  );
};
