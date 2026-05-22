import React from "react";
import { Chip } from "@mui/material";

const toneMap = {
  primary: { bg: "rgba(37,99,235,0.12)", color: "#2563EB" },
  success: { bg: "rgba(16,185,129,0.14)", color: "#047857" },
  warning: { bg: "rgba(245,158,11,0.18)", color: "#B45309" },
  error: { bg: "rgba(244,63,94,0.14)", color: "#BE123C" },
  info: { bg: "rgba(6,182,212,0.14)", color: "#0E7490" },
};

export default function InsightBadge({ label, tone = "primary" }) {
  const style = toneMap[tone] || toneMap.primary;
  return (
    <Chip
      size="small"
      label={label}
      sx={{
        bgcolor: style.bg,
        color: style.color,
        fontWeight: 700,
        borderRadius: 2,
      }}
    />
  );
}
