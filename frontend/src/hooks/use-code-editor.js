import React, { createContext, useState, useContext } from 'react';

const CodeEditorContext = createContext(null);

export const CodeEditorProvider = ({ children }) => {
  const [editorValue, setEditorValue] = useState('');

  return (
    <CodeEditorContext.Provider value={{ editorValue, setEditorValue }}>
      {children}
    </CodeEditorContext.Provider>
  );
};

// Custom hook for accessing context
export const useCodeEditor = () => useContext(CodeEditorContext);
