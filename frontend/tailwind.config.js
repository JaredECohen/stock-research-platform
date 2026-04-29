/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0A0F1C",
          900: "#0E1525",
          800: "#131B30",
          700: "#1A2440",
          600: "#243056",
        },
        accent: {
          500: "#52E0C4",
          600: "#2BC4A4",
          700: "#149E83",
        },
        warn: {
          500: "#F2B045",
        },
        danger: {
          500: "#EF6F6F",
        },
      },
      fontFamily: {
        sans: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
