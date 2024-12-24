"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import Link from "next/link";

import apiClient from "@/lib/api-client";
import { useAuth } from "@/hooks/use-auth";
import { useRouter } from "next/navigation";

// Define your form schema using Zod
const formSchema = z.object({
  email: z.string().email().min(1, { message: "Kindly enter a valid email.." }),
  password: z
    .string()
    .min(4, { message: "Password must have at least 4 characters" })
    .max(10, { message: "Password cannot have more than 10 characters" }),
});

export const LoginForm = () => {
    const {login} = useAuth()
    const router = useRouter()
  
  
    const form = useForm({
    resolver: zodResolver(formSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });

  const handleSubmit = async (formData)=>{
    try {
        await login(formData)
        console.log("User logged in successfully!")
    } catch (error) {
        console.error("Error in logging in: ",error)
    }
        
  }

  return (
    <div className="bg-customBackground w-[30%] flex flex-col justify-center items-center m-4 rounded-3xl p-10 border-2 border-gray-700 shadow-lg">
      <div className="flex p-5 mb-5">
        <h1 className="text-3xl text-white">
          Login
        </h1>
      </div>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(handleSubmit)} className="w-[80%] space-y-6">
          {/* Email Field */}
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="text-white text-lg">Email</FormLabel>
                <FormControl>
                  <Input
                    placeholder="Your Email..."
                    {...field}
                    className="w-full h-12 p-4 bg-transparent border-gray-700 text-white rounded-md focus:ring-2 outline-none hover:border-gray-500"
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
                <FormLabel className="text-white text-lg">Password</FormLabel>
                <FormControl>
                  <Input
                    type="password"
                    placeholder="Your Password..."
                    {...field}
                    className="w-full h-12 p-4 bg-transparent border-gray-700 text-white outline-none rounded-md focus:ring-2 hover:border-gray-500"
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          {/* Submit Button */}
          <Button type="submit" className="w-full h-10 mt-4 bg-customButton text-white py-3 rounded-md hover:bg-white hover:text-black">
            Submit
          </Button>
          <div className=" flex text-sm text-white mt-4 justify-center items-end ">
            Don't have an account? 
            <Link href={'/register'} className="text-customButton">
                Register here
            </Link>
          </div>
        </form>
      </Form>
    </div>
  );
};
