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
  CircularProgress,
  Stack,
} from "@mui/material";
import LectureSelect from "../components/LectureSelect";
import { useNavigate } from "react-router-dom";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import QuizIcon from "@mui/icons-material/Quiz";
import ClearIcon from "@mui/icons-material/Clear";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";

export default function QuizEntry() {
  const [noteId, setNoteId] = useState("");
  const [numQuestions, setNumQuestions] = useState(10);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const startQuiz = () => {
    if (!noteId) return;
    setLoading(true);
    // Simulate a small delay or check if note exists (optional)
    setTimeout(() => {
      setLoading(false);
      navigate(`/quiz?noteId=${noteId}&n=${numQuestions}`);
    }, 500);
  };

  return (
    <Container maxWidth="sm" sx={{ mt: 8, mb: 5 }}>
      <Typography
        variant="h3"
        gutterBottom
        align="center"
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          mb: 1,
        }}
      >
        🚀 Ready to Quiz?
      </Typography>
      <Typography
        variant="subtitle1"
        align="center"
        color="text.secondary"
        sx={{ mb: 6 }}
      >
        Enter your lecture details to begin testing your knowledge.
      </Typography>

      <Card
        sx={{
          p: 4,
          background: "linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)",
          boxShadow: "0 20px 40px rgba(0,0,0,0.1)",
          borderRadius: 4,
          border: "1px solid rgba(255,255,255,0.5)",
        }}
      >
        <CardContent sx={{ p: 0 }}>
          <Stack spacing={4}>
            <Box>
              <Typography
                variant="subtitle2"
                fontWeight="700"
                color="#4a5568"
                sx={{ mb: 1, ml: 1 }}
              >
                Lecture Note
              </Typography>
              <LectureSelect value={noteId} onChange={(v) => setNoteId(v)} />
            </Box>

            <Box>
              <Typography
                variant="subtitle2"
                fontWeight="700"
                color="#4a5568"
                sx={{ mb: 1, ml: 1 }}
              >
                Number of Questions
              </Typography>
              <TextField
                type="number"
                value={numQuestions}
                onChange={(e) => setNumQuestions(e.target.value)}
                fullWidth
                sx={{
                  "& .MuiOutlinedInput-root": {
                    height: "56px",
                    background: "white",
                    borderRadius: 2,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                    transition: "all 0.3s ease",
                    "&:hover": {
                      boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                    },
                    "&.Mui-focused": {
                      boxShadow: "0 4px 12px rgba(102, 126, 234, 0.15)",
                    },
                  },
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <QuizIcon sx={{ color: "#667eea" }} />
                    </InputAdornment>
                  ),
                  inputProps: { min: 1, max: 50 },
                }}
              />
            </Box>

            <Button
              variant="contained"
              fullWidth
              onClick={startQuiz}
              disabled={loading || !noteId || !numQuestions}
              endIcon={
                loading ? (
                  <CircularProgress size={20} color="inherit" />
                ) : (
                  <ArrowForwardIcon />
                )
              }
              sx={{
                height: "60px",
                fontSize: "1.1rem",
                borderRadius: 3,
                mt: 2,
              }}
            >
              Start Quiz
            </Button>
          </Stack>
        </CardContent>
      </Card>
    </Container>
  );
}
