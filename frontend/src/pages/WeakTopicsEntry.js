import React, { useState } from "react";
import { Button, Typography, Box, Stack } from "@mui/material";
import LectureSelect from "../components/LectureSelect";
import { useNavigate } from "react-router-dom";
import { ArrowForward as ArrowForwardIcon } from "@mui/icons-material";
import { PageHeader, SurfaceCard } from "../components/ui";

const PRIMARY_CTA_SX = {
  minHeight: 56,
  px: 3,
  borderRadius: 2.5,
  fontWeight: 800,
  letterSpacing: "0.01em",
  whiteSpace: "nowrap",
  color: "#fff",
  background: "linear-gradient(135deg, #2563EB 0%, #7C3AED 100%)",
  boxShadow: "0 10px 24px rgba(37,99,235,0.26)",
  transition: "all 180ms cubic-bezier(0.22, 1, 0.36, 1)",
  "&:hover": {
    background: "linear-gradient(135deg, #1D4ED8 0%, #6D28D9 100%)",
    boxShadow: "0 14px 30px rgba(37,99,235,0.34)",
    transform: "translateY(-1px)",
  },
  "&:active": {
    transform: "translateY(0)",
  },
};

export default function WeakTopicsEntry() {
  const [noteId, setNoteId] = useState("");
  const navigate = useNavigate();

  return (
    <Box sx={{ maxWidth: 860, mx: "auto", display: "grid", gap: 2.5 }}>
      <PageHeader
        title="Weak Topics Analysis"
        subtitle="Select a lecture to identify weak concepts and jump straight into focused practice."
      />

      <SurfaceCard>
        <Stack spacing={2.25}>
          <Typography variant="body2" color="text.secondary">
            Choose your lecture note and generate a ranked list of weak topics.
          </Typography>

          <Box
            sx={{
              display: "flex",
              gap: 1.5,
              alignItems: "stretch",
              flexWrap: { xs: "wrap", md: "nowrap" },
            }}
          >
            <Box sx={{ flex: 1, minWidth: 260 }}>
              <LectureSelect
                value={noteId}
                onChange={(value) => setNoteId(value)}
              />
            </Box>

            <Button
              variant="contained"
              endIcon={<ArrowForwardIcon />}
              onClick={() => noteId && navigate(`/weak-topics/${noteId}`)}
              disabled={!noteId}
              sx={PRIMARY_CTA_SX}
            >
              Analyze Weak Topics
            </Button>
          </Box>
        </Stack>
      </SurfaceCard>
    </Box>
  );
}
