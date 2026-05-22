import React, {
  createContext,
  useState,
  useMemo,
  useContext,
  useEffect,
} from "react";
import {
  createTheme,
  ThemeProvider as MUIThemeProvider,
} from "@mui/material/styles";
import { getDesignTokens } from "../theme";

const ColorModeContext = createContext({
  mode: "light",
  fontLevel: 0,
  setFontLevel: () => {},
  toggleColorMode: () => {},
});

export const useColorMode = () => useContext(ColorModeContext);

export const ThemeProvider = ({ children }) => {
  // Always use light mode
  const mode = "light";

  // fontLevel: -1 (Small), 0 (Normal), 1 (Large)
  const [fontLevel, setFontLevel] = useState(() => {
    const savedLevel = localStorage.getItem("fontLevel");
    return savedLevel ? parseInt(savedLevel, 10) : 0;
  });

  useEffect(() => {
    localStorage.setItem("fontLevel", fontLevel);
  }, [fontLevel]);

  const colorMode = useMemo(
    () => ({
      mode,
      fontLevel,
      setFontLevel, // Function to set specific level (-1, 0, 1)
      increaseFont: () => setFontLevel((prev) => Math.min(prev + 1, 2)),
      decreaseFont: () => setFontLevel((prev) => Math.max(prev - 1, -1)),
      toggleColorMode: () => {},
    }),
    [fontLevel],
  );

  const themeDescriptor = useMemo(() => {
    // Calculate font size multiplier based on level
    // Level -1: 12px (0.85 approx), 0: 14px (1), 1: 16px (1.15), 2: 18px (1.3)
    // Standard MUI body1 is 1rem (16px) usuall, let's adjust htmlFontSize or base

    const designTokens = getDesignTokens(mode);

    // adjust typography
    let fontSizeMod = 14;
    if (fontLevel === -1) fontSizeMod = 12; // Small
    if (fontLevel === 0) fontSizeMod = 14; // Medium (Default)
    if (fontLevel === 1) fontSizeMod = 16; // Large
    if (fontLevel === 2) fontSizeMod = 18; // Extra Large

    return createTheme({
      ...designTokens,
      typography: {
        ...designTokens.typography,
        fontSize: fontSizeMod,
      },
    });
  }, [mode, fontLevel]);

  return (
    <ColorModeContext.Provider value={colorMode}>
      <MUIThemeProvider theme={themeDescriptor}>{children}</MUIThemeProvider>
    </ColorModeContext.Provider>
  );
};
