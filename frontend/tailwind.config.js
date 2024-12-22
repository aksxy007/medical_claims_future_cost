/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
    content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
  	extend: {
  		colors: {
  			background: 'var(--background)',
  			foreground: 'var(--foreground)',
			customButton: "#00A550",
			customBackground:"#0A0A0A"
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
      fontFamily: {
        roboto: ['Roboto'],
        poppins:['Poppins']
      },
  	}
  },
  plugins: [require("tailwindcss-animate")],
}

