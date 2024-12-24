// middleware.js
import { NextResponse } from "next/server";

export function middleware(req) {
  // Get the token from the cookies
  const token = req.cookies.get("token");

  // If there's no token and trying to access a protected route
  if (!token && req.nextUrl.pathname.startsWith("/dashboard")) {
    // Redirect to the login page if not authenticated
    const loginUrl = new URL("/login", req.url); // Change to your login page route if needed
    return NextResponse.redirect(loginUrl);
  }

  // Allow the request to proceed if the token exists
  return NextResponse.next();
}

// Define which paths the middleware should run on
export const config = {
  matcher: ["/:path*"], // Match routes starting with /dashboard
};
