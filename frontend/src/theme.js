import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    fontSize: 13, // Default is 14. Scaling down slightly.
    h1: { fontSize: "2.2rem" }, // Scaled down from default
    h2: { fontSize: "1.8rem" },
    h3: { fontSize: "1.5rem" },
    h4: { fontSize: "1.25rem" },
    h5: { fontSize: "1.1rem" },
    h6: { fontSize: "1rem" },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: "none", // Modern look
          borderRadius: 8,
          fontWeight: 600,
        },
        containedPrimary: {
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          color: "white",
          boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)",
          transition: "all 0.3s ease",
          "&:hover": {
            boxShadow: "0 6px 20px rgba(102, 126, 234, 0.6)",
            transform: "translateY(-2px)",
          },
          "&.Mui-disabled": {
            background: "#e0e0e0",
            color: "#9e9e9e",
            boxShadow: "none",
          },
        },
        outlined: {
          borderColor: "#667eea",
          color: "#667eea",
          "&:hover": {
            borderColor: "#764ba2",
            backgroundColor: "rgba(102, 126, 234, 0.05)",
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

export default theme;
