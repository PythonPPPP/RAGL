/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          900: '#0b1220',
          800: '#0f1a2f',
          700: '#152341'
        },
        card: {
          900: '#0f1a2f',
          850: '#111e36'
        },
        border: {
          700: 'rgba(255,255,255,0.08)'
        }
      },
      boxShadow: {
        soft: '0 18px 40px rgba(0,0,0,0.35)'
      }
    }
  },
  plugins: []
}
