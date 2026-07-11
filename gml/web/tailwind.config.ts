import type { Config } from "tailwindcss";

/**
 * Every value resolves to a CSS variable defined in styles/tokens.css, so the
 * design tokens stay the single source of truth and theming never forks.
 */
const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./hooks/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "bg-0": "var(--bg-0)",
        "bg-1": "var(--bg-1)",
        "bg-2": "var(--bg-2)",
        "bg-3": "var(--bg-3)",
        border: "var(--border)",
        "border-strong": "var(--border-strong)",
        "text-0": "var(--text-0)",
        "text-1": "var(--text-1)",
        "text-2": "var(--text-2)",
        accent: "var(--accent)",
        "accent-glow": "var(--accent-glow)",
        "cluster-1": "var(--cluster-1)",
        "cluster-2": "var(--cluster-2)",
        "cluster-3": "var(--cluster-3)",
        "cluster-4": "var(--cluster-4)",
        "cluster-5": "var(--cluster-5)",
        "cluster-6": "var(--cluster-6)",
      },
      fontFamily: {
        display: "var(--font-display)",
        clash: ['"Clash Display"', "var(--font-display)", "sans-serif"],
        mono: "var(--font-mono)",
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
        xl: "var(--radius-xl)",
      },
      transitionTimingFunction: {
        out: "var(--ease-out)",
        "in-out": "var(--ease-in-out)",
      },
      transitionDuration: {
        DEFAULT: "180ms",
      },
      boxShadow: {
        glow: "0 0 0 1px var(--accent), 0 0 24px -4px var(--accent-glow)",
      },
    },
  },
  // No tailwindcss-animate: we only use core (animate-pulse/spin) + our own
  // .animate-rise. Including it registers a second `duration-*` utility
  // (animation-duration) that collides with core transition-duration.
  plugins: [],
};
export default config;
