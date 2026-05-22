import React from "react";
import { Box } from "@mui/material";

export default function ActionBar({ children, sticky = false, sx }) {
  return (
    <Box
      sx={{
        display: "flex",
        gap: 1,
        flexWrap: "wrap",
        justifyContent: "flex-end",
        position: sticky ? "sticky" : "static",
        bottom: sticky ? 12 : "auto",
        zIndex: sticky ? 5 : "auto",
        bgcolor: sticky ? "rgba(255,255,255,0.88)" : "transparent",
        backdropFilter: sticky ? "blur(6px)" : "none",
        border: sticky ? "1px solid" : "none",
        borderColor: sticky ? "divider" : "transparent",
        p: sticky ? 1 : 0,
        borderRadius: sticky ? 2 : 0,
        ...sx,
      }}
    >
      {children}
    </Box>
  );
}
