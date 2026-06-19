/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      // Academic/clinical palette: cool navy + warm gold + linen background.
      // Migrated from the Dora Designer mock under /Wiz3 arayüz iyileştirmesi/.
      colors: {
        // Page + surface
        page: "#EDF1F6",        // soft warm-blue page background
        surface: "#FBFCFE",     // panel/card surface
        // Borders + dividers
        line: "#DCE3EC",        // default border
        divider: "#EBEFF4",     // table-row / subtle hairlines
        // Soft accents
        tint: "#F1F5FA",        // table-header strip / footer band
        chip: "#EEF2F7",        // resting chip background
        // Brand navy
        ink: {
          50:  "#EAF1FA",
          100: "#E5EEF9",
          150: "#E0EBF8",
          200: "#BFD3EC",
          300: "#7FA0C8",
          500: "#2A5DA0",       // primary
          600: "#1F4880",       // deep
          700: "#18345E",
        },
        // Warm gold accent (Power Analysis / academic)
        gold: {
          50:  "#FBF6EC",
          100: "#FBF4E2",
          200: "#E8D9B4",
          400: "#C9A86A",
          600: "#B0851F",
        },
        // Calm green (success / save)
        moss: {
          100: "#E3F0E8",
          400: "#3E9D6A",
          600: "#2E7D5B",
        },
        // Soft red (destructive / close)
        clay: {
          100: "#F7E6E2",
          600: "#C0492F",
        },
      },
      fontFamily: {
        // Serif for headings (academic feel), Hanken Grotesk for body.
        serif: ["Newsreader", "Source Serif Pro", "ui-serif", "Georgia", "serif"],
        sans:  ["Hanken Grotesk", "system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
      },
      boxShadow: {
        // Very subtle card lift, matching the mock.
        card: "0 1px 2px rgba(27,36,48,0.04)",
        soft: "0 9px 22px rgba(31,72,128,0.18)",
      },
      borderRadius: {
        card: "14px",
      },
    },
  },
  plugins: [],
}
