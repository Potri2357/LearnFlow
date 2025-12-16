import React, { useState } from "react";
import {
  TextField,
  Button,
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  InputAdornment,
  Tooltip,
  IconButton,
} from "@mui/material";
import LectureSelect from "../components/LectureSelect";
import { useNavigate } from "react-router-dom";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import ClearIcon from "@mui/icons-material/Clear";

export default function WeakTopicsEntry() {
  const [noteId, setNoteId] = useState("");
  const navigate = useNavigate();

  return (
    <Container maxWidth="sm" sx={{ mt: 5, mb: 5 }}>
      <Typography
        variant="h3"
        gutterBottom
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          mb: 4,
        }}
      >
        📊 Weak Topics Analysis
      </Typography>

      <Card
        sx={{
          p: 4,
          background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
          borderRadius: 3,
        }}
      >
        <CardContent sx={{ p: 0 }}>
          <Typography
            variant="h6"
            sx={{ mb: 3, fontWeight: 700, color: "#333" }}
          >
            🎯 Enter Lecture Note ID
          </Typography>

          <Box
            sx={{
              display: "flex",
              gap: 2,
              alignItems: "stretch",
              flexWrap: { xs: "wrap", sm: "nowrap" },
            }}
          >
            <LectureSelect value={noteId} onChange={(v) => setNoteId(v)} />

            <Button
              variant="contained"
              onClick={() => noteId && navigate(`/weak-topics/${noteId}`)}
              disabled={!noteId}
              sx={{
                height: "56px",
                minWidth: "180px",
                whiteSpace: "nowrap",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              View Topics
            </Button>
          </Box>
        </CardContent>
      </Card>
    </Container>
  );
}
