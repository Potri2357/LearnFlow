import React from "react";
import { Box, Typography, LinearProgress } from "@mui/material";

export default function StepProgress({ step = 1, total = 4, label }) {
  const percent = Math.max(0, Math.min(100, (step / total) * 100));

  return (
    <Box sx={{ mb: 2.5 }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 0.8,
        }}
      >
        <Typography
          variant="caption"
          sx={{ fontWeight: 700, color: "text.secondary" }}
        >
          {label || "Progress"}
        </Typography>
        <Typography
          variant="caption"
          sx={{ fontWeight: 700, color: "primary.main" }}
        >
          Step {step} of {total}
        </Typography>
      </Box>
      <LinearProgress
        value={percent}
        variant="determinate"
        sx={{ height: 9, borderRadius: 99 }}
      />
    </Box>
  );
}
