import './globals.css';  // Global styles (e.g., ShadCN CSS or custom styles)
import { AuthProvider } from '@/hooks/use-auth';
import { ThemeProvider } from '@/components/ThemeProvider';
import {Roboto} from 'next/font/google'
import { cn } from '@/lib/utils';
import { ToastProvider } from '@/hooks/use-toast';


export const metadata = {
  title: 'AutoML',
  description: 'Web UI for the AutoML',
};


const roboto = Roboto({
  subsets:['latin'],
  weight:['400','700']
})

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={roboto.className} >
        <ThemeProvider 
        attribute="class"
        defaultTheme="system"
        enableSystem
        // disableTransitionOnChange
        >
          <ToastProvider>
        <AuthProvider>
          {/* <ProtectedRoute> */}
            {children}
         
          {/* </ProtectedRoute> */}
          
        </AuthProvider>
        </ToastProvider>
        </ThemeProvider>
        
        
      </body>
    </html>
  );
}

