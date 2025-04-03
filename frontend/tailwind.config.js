/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#3B82F6', // Blue
          dark: '#2563EB',
          light: '#60A5FA',
        },
        secondary: {
          DEFAULT: '#10B981', // Green
          dark: '#059669',
          light: '#34D399',
        },
      },
      borderWidth: {
        '3': '3px',
      },
    },
  },
  plugins: [],
}; 