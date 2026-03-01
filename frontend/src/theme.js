import { createTheme } from '@mui/material/styles';

export const getDesignTokens = (mode) => ({
  palette: {
    mode: 'dark', // We force dark mode to match Stitch design
    primary: {
      main: "#137fec", // Bright Blue from Stitch
      light: "#60A5FA",
      dark: "#0b5bba",
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: "#0bda5b", // Emerald Green / Success
      light: "#34D399",
      dark: "#059669",
      contrastText: "#FFFFFF",
    },
    accent: {
      main: "#f59e0b", // Amber / Warning
      dark: "#D97706",
      light: "#FCD34D",
    },
    error: {
      main: "#ef4444", // Red / Error
    },
    background: {
      default: "#101922", // Background dark
      paper: "#1c252e",   // Surface Dark / Card Dark
    },
    text: {
      primary: "#FFFFFF", 
      secondary: "#92adc9", 
    },
    divider: "#2a3b4d", // Card border
    gradients: {
      primary: "linear-gradient(135deg, #137fec 0%, #0b5bba 100%)",
      glass: "rgba(16, 25, 34, 0.9)",
    }
  },
  typography: {
    fontFamily: '"Lexend", "Noto Sans", sans-serif',
    h1: { fontSize: "3rem", fontWeight: 900, letterSpacing: '-0.033em', lineHeight: 1.2 },
    h2: { fontSize: "2.25rem", fontWeight: 700, letterSpacing: '-0.015em', lineHeight: 1.3 },
    h3: { fontSize: "1.875rem", fontWeight: 700, lineHeight: 1.3 },
    h4: { fontSize: "1.5rem", fontWeight: 700, lineHeight: 1.4 },
    h5: { fontSize: "1.25rem", fontWeight: 700 },
    h6: { fontSize: "1.125rem", fontWeight: 700 },
    body1: { fontSize: "1rem", lineHeight: 1.6, fontWeight: 400 },
    body2: { fontSize: "0.875rem", lineHeight: 1.6, fontWeight: 400 },
    button: { textTransform: "none", fontWeight: 700, fontSize: "0.875rem" },
    caption: { fontSize: "0.75rem", fontWeight: 500 },
  },
  shape: {
    borderRadius: 8, // 0.5rem defaults
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          boxShadow: "none",
          padding: "8px 16px",
        },
        containedPrimary: {
          background: "#137fec",
          color: "white",
          "&:hover": {
            background: "#0b5bba",
            boxShadow: "0 8px 16px rgba(19, 127, 236, 0.25)",
            transform: "translateY(-1px)",
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: 16, // xl
          boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
          border: `1px solid ${theme.palette.divider}`,
          background: theme.palette.background.paper,
          overflow: 'hidden',
        }),
      },
    },
    MuiPaper: {
      styleOverrides: {
        rounded: {
          borderRadius: 16,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: ({ theme }) => ({
          background: "#111a22",
          backdropFilter: "blur(12px)",
          color: theme.palette.text.primary,
          boxShadow: "none",
          borderBottom: "1px solid",
          borderColor: theme.palette.divider,
        }),
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: ({ theme }) => ({
          backgroundColor: "#111a22",
          color: theme.palette.text.primary,
          borderRight: `1px solid ${theme.palette.divider}`,
          boxShadow: "none",
        }),
      },
    },
    MuiCssBaseline: {
      styleOverrides: (theme) => ({
        body: {
          backgroundColor: theme.palette.background.default,
          color: theme.palette.text.primary,
          backgroundImage: 'none',
        },
      }),
    },
  },
});

const theme = createTheme(getDesignTokens('dark')); 
export default theme;
