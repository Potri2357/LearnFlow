import React from "react";
import { Box, Typography } from "@mui/material";
import SurfaceCard from "./SurfaceCard";

export default function MetricCard({
  label,
  value,
  hint,
  icon,
  tone = "primary",
  sx,
}) {
  return (
    <SurfaceCard
      sx={{
        position: "relative",
        overflow: "hidden",
        background: "linear-gradient(150deg, #FFFFFF 0%, #F8FBFF 100%)",
        ...sx,
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
        }}
      >
        <Box>
          <Typography
            variant="caption"
            sx={{ color: "text.secondary", fontWeight: 600 }}
          >
            {label}
          </Typography>
          <Typography variant="h4" sx={{ mt: 0.8, fontWeight: 800 }}>
            {value}
          </Typography>
          {hint ? (
            <Typography
              variant="caption"
              sx={{ color: `${tone}.main`, fontWeight: 700 }}
            >
              {hint}
            </Typography>
          ) : null}
        </Box>
        {icon ? (
          <Box
            sx={{
              width: 42,
              height: 42,
              borderRadius: 1.5,
              display: "grid",
              placeItems: "center",
              color: `${tone}.main`,
              bgcolor: `${tone}.main15`,
              backgroundColor: "rgba(37,99,235,0.1)",
            }}
          >
            {icon}
          </Box>
        ) : null}
      </Box>
    </SurfaceCard>
  );
}
