import React from "react";
import { Box, Typography, Button } from "@mui/material";
import ErrorOutlineRoundedIcon from "@mui/icons-material/ErrorOutlineRounded";
import SurfaceCard from "./SurfaceCard";

export default function ErrorState({
  title = "Something went wrong",
  message = "Please try again. If this continues, check your connection.",
  onRetry,
  retryLabel = "Retry",
  fallbackAction,
}) {
  return (
    <SurfaceCard sx={{ borderColor: "rgba(244,63,94,0.25)" }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, mb: 1.5 }}>
        <ErrorOutlineRoundedIcon color="error" />
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        {message}
      </Typography>
      <Box sx={{ display: "flex", gap: 1 }}>
        {onRetry ? (
          <Button variant="contained" color="error" onClick={onRetry}>
            {retryLabel}
          </Button>
        ) : null}
        {fallbackAction || null}
      </Box>
    </SurfaceCard>
  );
}
