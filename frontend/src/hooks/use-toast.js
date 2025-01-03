"use client"

import { useTheme } from "next-themes";
import { createContext, useContext } from "react";
import { ToastContainer,toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';

const ToastContext = createContext({
    showToast: () => {},
})

export const ToastProvider = ({children})=>{
    
    const {theme} = useTheme()

    const showToast = ({message,type='info'})=>{
        console.log(`Toast: ${message} (${type})`);
        toast[type](message)
    }


    return (
        <ToastContext.Provider value={{showToast}}>
            {children}

            <ToastContainer 
                position="top-right"
                autoClose={1000}
                hideProgressBar={false}
                newestOnTop={true}
                closeOnClick={true}
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme={theme}
            />
        </ToastContext.Provider>
    )
}

export const useToast = ()=>useContext(ToastContext);
