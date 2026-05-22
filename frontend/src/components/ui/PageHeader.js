import React from "react";
import { Box, Typography, Chip } from "@mui/material";

export default function PageHeader({ title, subtitle, badge, actions, sx }) {
  return (
    <Box
      sx={{
        display: "flex",
        alignItems: { xs: "flex-start", md: "center" },
        justifyContent: "space-between",
        flexDirection: { xs: "column", md: "row" },
        gap: 2,
        mb: 3,
        ...sx,
      }}
    >
      <Box>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.75 }}>
          <Typography variant="h4" sx={{ fontWeight: 800 }}>
            {title}
          </Typography>
          {badge ? (
            <Chip
              size="small"
              label={badge}
              color="primary"
              sx={{ fontWeight: 700, borderRadius: 2 }}
            />
          ) : null}
        </Box>
        {subtitle ? (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        ) : null}
      </Box>
      {actions ? (
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>{actions}</Box>
      ) : null}
    </Box>
  );
}
