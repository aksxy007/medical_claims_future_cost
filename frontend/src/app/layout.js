import { ProtectedRoute } from '@/components/ProtectedRoute';
import '../globals.css';  // Global styles (e.g., ShadCN CSS or custom styles)
import { AuthProvider } from '@/hooks/useAuth';


export const metadata = {
  title: 'AutoML',
  description: 'Web UI for the AutoML',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          <ProtectedRoute>
          {children}
          </ProtectedRoute>
          
        </AuthProvider>
        
      </body>
    </html>
  );
}
