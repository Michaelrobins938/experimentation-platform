/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        carbon: {
          core: '#0a0a0a',
          surface: '#111111',
          elevated: '#1a1a1a',
          muted: '#252525',
        },
        tactical: {
          indigo: '#6366f1',
          emerald: '#10b981',
          amber: '#f59e0b',
          red: '#ef4444',
          cyan: '#06b6d4',
          purple: '#a855f7',
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'SF Mono', 'Menlo', 'Consolas', 'monospace'],
      }
    },
  },
  plugins: [],
}
