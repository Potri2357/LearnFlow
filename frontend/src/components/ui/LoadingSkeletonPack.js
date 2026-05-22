import React from "react";
import { Grid, Skeleton } from "@mui/material";

export default function LoadingSkeletonPack({ rows = 3, cardHeight = 120 }) {
  return (
    <Grid container spacing={2.5}>
      {Array.from({ length: rows }).map((_, idx) => (
        <Grid item xs={12} md={rows > 2 ? 4 : 6} key={idx}>
          <Skeleton
            variant="rounded"
            animation="wave"
            height={cardHeight}
            sx={{ borderRadius: 2.5, bgcolor: "rgba(148,163,184,0.18)" }}
          />
        </Grid>
      ))}
    </Grid>
  );
}
