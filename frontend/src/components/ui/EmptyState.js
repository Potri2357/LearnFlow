import React from "react";
import { Box, Typography, Button } from "@mui/material";
import InboxRoundedIcon from "@mui/icons-material/InboxRounded";
import SurfaceCard from "./SurfaceCard";

export default function EmptyState({
  title = "Nothing here yet",
  message = "Start by creating your first item.",
  actionLabel,
  onAction,
  icon,
}) {
  return (
    <SurfaceCard sx={{ textAlign: "center" }}>
      <Box
        sx={{
          width: 56,
          height: 56,
          borderRadius: "50%",
          mx: "auto",
          mb: 1.5,
          display: "grid",
          placeItems: "center",
          bgcolor: "rgba(37,99,235,0.12)",
          color: "primary.main",
        }}
      >
        {icon || <InboxRoundedIcon />}
      </Box>
      <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
        {title}
      </Typography>
      <Typography
        variant="body2"
        color="text.secondary"
        sx={{ maxWidth: 460, mx: "auto", mb: 2 }}
      >
        {message}
      </Typography>
      {actionLabel ? (
        <Button variant="contained" onClick={onAction}>
          {actionLabel}
        </Button>
      ) : null}
    </SurfaceCard>
  );
}
