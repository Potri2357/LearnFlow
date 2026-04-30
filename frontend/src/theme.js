import { createTheme } from "@mui/material/styles";

// ── LearnFlow earth-tone palette ──────────────────────────────────────────────
export const PALETTE = {
  ink: "#264653",
  teal: "#2A9D8F",
  sand: "#E9C46A",
  clay: "#F4A261",
  coral: "#E76F51",
};

export const COLORS = {
  ink: PALETTE.ink,
  teal: PALETTE.teal,
  sand: PALETTE.sand,
  clay: PALETTE.clay,
  coral: PALETTE.coral,
};

export const GRADIENTS = {
  tide: "linear-gradient(135deg, #264653 0%, #2A9D8F 100%)",
  sunset: "linear-gradient(135deg, #F4A261 0%, #E76F51 100%)",
  harvest: "linear-gradient(135deg, #E9C46A 0%, #F4A261 100%)",
  dune: "linear-gradient(135deg, #E9C46A 0%, #2A9D8F 100%)",
  ember: "linear-gradient(135deg, #E76F51 0%, #264653 100%)",
  prism:
    "linear-gradient(130deg, #264653 0%, #2A9D8F 35%, #F4A261 72%, #E76F51 100%)",
};

export const SUBJECT_COLORS = [
  "#264653", // ink
  "#2A9D8F", // teal
  "#E9C46A", // sand
  "#F4A261", // clay
  "#E76F51", // coral
  "#4A6875", // ink light
  "#58B8AC", // teal light
  "#F1D587", // sand light
  "#F29A86", // coral light
  "#8C5E58", // muted mix
];

export const getDesignTokens = (mode) => ({
  learnflow: {
    spacingBase: 8,
    radius: { sm: 8, md: 12, lg: 16, xl: 20 },
    contentMax: 1280,
    motionEasing: "cubic-bezier(0.22, 1, 0.36, 1)",
    motionFast: "160ms",
    motionMedium: "260ms",
  },
  palette: {
    mode: "light",
    primary: {
      main: PALETTE.teal,
      light: "#58B8AC",
      dark: "#1F7F73",
      contrastText: "#FFFFFF",
    },
    secondary: {
      main: PALETTE.coral,
      light: "#F29A86",
      dark: "#C94E35",
      contrastText: "#FFFFFF",
    },
    success: {
      main: PALETTE.teal,
      light: "#58B8AC",
      dark: "#1F7F73",
      contrastText: "#FFFFFF",
    },
    error: {
      main: PALETTE.coral,
      light: "#F29A86",
    },
    warning: {
      main: PALETTE.sand,
      light: "#F1D587",
    },
    info: {
      main: PALETTE.ink,
      light: "#4A6875",
    },
    background: {
      default: "#F6F1E6",
      paper: "#FFFDF8",
    },
    text: {
      primary: "#264653",
      secondary: "#556B72",
      disabled: "#9CA3AF",
    },
    divider: "#E8DCC9",
    action: {
      hover: "rgba(42,157,143,0.08)",
      selected: "rgba(42,157,143,0.15)",
    },
  },
  typography: {
    fontFamily: '"Lexend", "Noto Sans", sans-serif',
    h1: {
      fontSize: "2.5rem",
      fontWeight: 900,
      letterSpacing: "-0.03em",
      lineHeight: 1.2,
    },
    h2: {
      fontSize: "2rem",
      fontWeight: 800,
      letterSpacing: "-0.025em",
      lineHeight: 1.25,
    },
    h3: {
      fontSize: "1.625rem",
      fontWeight: 700,
      letterSpacing: "-0.018em",
      lineHeight: 1.3,
    },
    h4: { fontSize: "1.375rem", fontWeight: 700, lineHeight: 1.35 },
    h5: { fontSize: "1.2rem", fontWeight: 700 },
    h6: { fontSize: "1.04rem", fontWeight: 700 },
    body1: { fontSize: "1rem", lineHeight: 1.625, fontWeight: 400 },
    body2: { fontSize: "0.875rem", lineHeight: 1.57, fontWeight: 400 },
    button: {
      textTransform: "none",
      fontWeight: 700,
      fontSize: "0.875rem",
      letterSpacing: "0.01em",
    },
    caption: { fontSize: "0.75rem", fontWeight: 500 },
    overline: { fontSize: "0.7rem", fontWeight: 700, letterSpacing: "0.1em" },
    subtitle1: { fontWeight: 600 },
    subtitle2: { fontWeight: 600 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 14,
          boxShadow: "none",
          padding: "10px 20px",
          fontWeight: 700,
          letterSpacing: "0.012em",
          position: "relative",
          overflow: "hidden",
          transition:
            "transform 180ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 180ms cubic-bezier(0.22, 1, 0.36, 1), border-color 180ms cubic-bezier(0.22, 1, 0.36, 1), background-color 180ms cubic-bezier(0.22, 1, 0.36, 1), filter 180ms cubic-bezier(0.22, 1, 0.36, 1)",
          "&:hover": { boxShadow: "none", transform: "translateY(-1px)" },
          "&:active": { transform: "translateY(0) scale(0.99)" },
          "&.Mui-disabled": {
            transform: "none",
            filter: "grayscale(0.15)",
            opacity: 0.72,
          },
        },
        contained: {
          color: "#FFFFFF",
          boxShadow: "0 10px 24px rgba(15,23,42,0.16)",
          "&:hover": {
            boxShadow: "0 14px 30px rgba(15,23,42,0.2)",
          },
          "&.Mui-disabled": {
            color: "rgba(255,255,255,0.9)",
            backgroundColor: "#C7D2FE",
            boxShadow: "none",
          },
        },
        containedPrimary: {
          background: GRADIENTS.tide,
          boxShadow: "0 12px 28px rgba(42,157,143,0.28)",
          "&::after": {
            content: '""',
            position: "absolute",
            inset: 0,
            background:
              "linear-gradient(110deg, transparent 0%, rgba(255,255,255,0.20) 38%, transparent 68%)",
            transform: "translateX(-140%)",
            transition: "transform 0.55s ease",
          },
          "&:hover": {
            background:
              "linear-gradient(135deg, #1F7F73 0%, #2A9D8F 50%, #264653 100%)",
            boxShadow: "0 16px 32px rgba(42,157,143,0.36)",
            "&::after": { transform: "translateX(140%)" },
          },
        },
        containedSecondary: {
          background: GRADIENTS.sunset,
          boxShadow: "0 12px 28px rgba(231,111,81,0.28)",
          "&:hover": {
            background: "linear-gradient(135deg, #E76F51 0%, #F4A261 100%)",
            boxShadow: "0 16px 34px rgba(231,111,81,0.34)",
          },
        },
        outlined: {
          borderWidth: 1.5,
          backgroundColor: "rgba(255,253,248,0.76)",
          backdropFilter: "blur(6px)",
          WebkitBackdropFilter: "blur(6px)",
          "&:hover": {
            borderWidth: 1.5,
          },
          "&.Mui-disabled": {
            borderColor: "rgba(102,126,133,0.45)",
            backgroundColor: "rgba(255,253,248,0.58)",
          },
        },
        outlinedPrimary: {
          borderColor: "rgba(42,157,143,0.34)",
          backgroundColor: "rgba(255,253,248,0.68)",
          backdropFilter: "blur(6px)",
          WebkitBackdropFilter: "blur(6px)",
          "&:hover": {
            backgroundColor: "rgba(42,157,143,0.09)",
            borderColor: "#2A9D8F",
            boxShadow: "0 10px 22px rgba(42,157,143,0.18)",
          },
        },
        outlinedSecondary: {
          borderColor: "rgba(231,111,81,0.35)",
          color: "#C94E35",
          "&:hover": {
            borderColor: "#E76F51",
            backgroundColor: "rgba(231,111,81,0.10)",
            boxShadow: "0 10px 22px rgba(231,111,81,0.18)",
          },
        },
        text: {
          borderRadius: 12,
          "&:hover": {
            backgroundColor: "rgba(42,157,143,0.06)",
          },
          "&.Mui-disabled": {
            color: "#9CA3AF",
          },
        },
        textPrimary: {
          borderRadius: 12,
          "&:hover": {
            backgroundColor: "rgba(42,157,143,0.08)",
          },
        },
        textSecondary: {
          "&:hover": {
            backgroundColor: "rgba(231,111,81,0.10)",
          },
        },
        sizeSmall: { padding: "6px 14px", fontSize: "0.8rem" },
        sizeMedium: { padding: "9px 18px", fontSize: "0.88rem" },
        sizeLarge: { padding: "12px 28px", fontSize: "0.98rem" },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 20,
          boxShadow: "0 12px 32px rgba(38,70,83,0.08)",
          border: "1px solid rgba(232,220,201,0.88)",
          background: "linear-gradient(150deg, #FFFDF8 0%, #F7F0E3 100%)",
          overflow: "hidden",
          transition: "box-shadow 0.25s ease, transform 0.25s ease",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
        rounded: { borderRadius: 18 },
        elevation1: { boxShadow: "0 2px 8px rgba(37,99,235,0.08)" },
        elevation2: { boxShadow: "0 6px 18px rgba(37,99,235,0.12)" },
        elevation4: { boxShadow: "0 12px 30px rgba(37,99,235,0.14)" },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: "rgba(255,253,248,0.9)",
          backdropFilter: "blur(16px)",
          WebkitBackdropFilter: "blur(16px)",
          color: "#264653",
          boxShadow: "none",
          borderBottom: "1px solid rgba(232,220,201,0.9)",
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: "linear-gradient(180deg, #FFFDF8 0%, #F7F0E3 100%)",
          color: "#264653",
          borderRight: "1px solid rgba(232,220,201,0.9)",
          boxShadow: "4px 0 28px rgba(38,70,83,0.14)",
        },
      },
    },
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: "#F6F1E6",
          color: "#264653",
          backgroundImage:
            "radial-gradient(circle at 12% 15%, rgba(42,157,143,0.14) 0%, transparent 36%), radial-gradient(circle at 88% 20%, rgba(233,196,106,0.14) 0%, transparent 34%), radial-gradient(circle at 70% 88%, rgba(231,111,81,0.13) 0%, transparent 35%), radial-gradient(circle at 18% 78%, rgba(244,162,97,0.1) 0%, transparent 32%)",
        },
        "::selection": {
          background: "rgba(42,157,143,0.2)",
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 700,
          borderRadius: 10,
          transition: "all 0.15s ease",
        },
        colorPrimary: {
          backgroundColor: "rgba(42,157,143,0.14)",
          color: "#1F7F73",
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: { backgroundColor: "#F0E3C1", borderRadius: 8, height: 8 },
        bar: { borderRadius: 8 },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": {
            borderRadius: 12,
            backgroundColor: "#FFFDF8",
            transition: "all 0.2s ease",
            "& fieldset": {
              borderColor: "#E8DCC9",
              transition: "border-color 0.2s ease",
            },
            "&:hover fieldset": { borderColor: "#2A9D8F" },
            "&.Mui-focused": {
              backgroundColor: "#FFFDF8",
              "& fieldset": { borderColor: "#2A9D8F", borderWidth: 2 },
            },
          },
          "& .MuiInputLabel-root": { fontWeight: 600 },
        },
      },
    },
    MuiDialog: {
      styleOverrides: {
        paper: {
          borderRadius: 22,
          boxShadow: "0 32px 72px rgba(38,70,83,0.28)",
        },
        backdrop: {
          backdropFilter: "blur(6px)",
          backgroundColor: "rgba(38,70,83,0.45)",
        },
      },
    },
    MuiTooltip: {
      styleOverrides: {
        tooltip: {
          borderRadius: 8,
          fontWeight: 600,
          fontSize: "0.75rem",
          backgroundColor: "#264653",
          padding: "6px 12px",
        },
        arrow: { color: "#264653" },
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          transition: "all 0.15s ease",
          "&.Mui-selected": {
            backgroundColor: "rgba(42,157,143,0.12)",
            "&:hover": { backgroundColor: "rgba(42,157,143,0.16)" },
          },
        },
      },
    },
    MuiSelect: { styleOverrides: { outlined: { borderRadius: 10 } } },
    MuiAccordion: {
      styleOverrides: {
        root: { boxShadow: "none", "&:before": { display: "none" } },
      },
    },
    MuiDivider: { styleOverrides: { root: { borderColor: "#DCE7FF" } } },
    MuiAvatar: { styleOverrides: { root: { fontWeight: 700 } } },
    MuiMenu: {
      styleOverrides: {
        paper: {
          borderRadius: 14,
          border: "1px solid #E8DCC9",
          boxShadow: "0 12px 36px rgba(38,70,83,0.18)",
        },
      },
    },
    MuiMenuItem: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          margin: "2px 6px",
          fontWeight: 500,
          transition: "all 0.15s ease",
          "&:hover": { backgroundColor: "rgba(42,157,143,0.07)" },
          "&.Mui-selected": {
            backgroundColor: "rgba(42,157,143,0.12)",
            fontWeight: 700,
            "&:hover": { backgroundColor: "rgba(42,157,143,0.17)" },
          },
        },
      },
    },
    MuiAlert: {
      styleOverrides: { root: { borderRadius: 12, fontWeight: 500 } },
    },
    MuiDialogTitle: {
      styleOverrides: {
        root: {
          fontWeight: 800,
          fontSize: "1.125rem",
          padding: "24px 24px 16px",
        },
      },
    },
    MuiDialogContent: { styleOverrides: { root: { padding: "0 24px 16px" } } },
    MuiDialogActions: {
      styleOverrides: { root: { padding: "12px 24px 24px", gap: 8 } },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          backgroundColor: "#FAFCFF",
          transition: "all 0.2s ease",
          "& fieldset": {
            borderColor: "#D6E4FF",
            transition: "border-color 0.2s ease",
          },
          "&:hover fieldset": { borderColor: "#93C5FD" },
          "&.Mui-focused": {
            backgroundColor: "#FFFFFF",
            "& fieldset": { borderColor: "#2563EB", borderWidth: 2 },
          },
        },
      },
    },
    MuiFormLabel: {
      styleOverrides: {
        root: { fontWeight: 600, "&.Mui-focused": { color: "#2563EB" } },
      },
    },
    MuiBadge: {
      styleOverrides: { badge: { fontWeight: 800, fontSize: "0.65rem" } },
    },
    MuiTabs: {
      styleOverrides: {
        root: { minHeight: 44 },
        indicator: { height: 3, borderRadius: 2, background: GRADIENTS.prism },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          fontWeight: 600,
          fontSize: "0.875rem",
          textTransform: "none",
          minHeight: 44,
          transition: "color 0.2s ease",
          "&.Mui-selected": { fontWeight: 800, color: "#2563EB" },
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        head: {
          fontWeight: 700,
          color: "#41516D",
          fontSize: "0.78rem",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        },
        root: { borderColor: "#DCE7FF" },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          transition: "all 0.2s ease",
          "&:hover": { transform: "scale(1.08)" },
        },
      },
    },
  },
});

const theme = createTheme(getDesignTokens("light"));
export default theme;
