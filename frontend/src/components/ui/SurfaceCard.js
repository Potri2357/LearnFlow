import React from "react";
import { Card, CardContent } from "@mui/material";

export default function SurfaceCard({ children, sx, contentSx, ...props }) {
  return (
    <Card
      sx={{
        borderRadius: 2.5,
        border: "1px solid",
        borderColor: "divider",
        boxShadow: "0 10px 28px rgba(19,32,58,0.08)",
        ...sx,
      }}
      {...props}
    >
      <CardContent sx={{ p: 3, ...contentSx }}>{children}</CardContent>
    </Card>
  );
}
