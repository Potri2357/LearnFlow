import React, { useState } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  Container,
  Alert,
  CircularProgress,
  Stack,
  IconButton,
  Tooltip,
  Paper,
  Chip,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from "@mui/material";
import API from "../api/api";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import ContentPasteIcon from "@mui/icons-material/ContentPaste";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import QuizIcon from "@mui/icons-material/Quiz";
import TopicIcon from "@mui/icons-material/Topic";
import WarningIcon from "@mui/icons-material/Warning";
import { useNavigate } from "react-router-dom";

import { useDropzone } from 'react-dropzone';

export default function UploadNotes() {
  const [mode, setMode] = useState("text"); // "text" | "pdf"
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [pdfFile, setPdfFile] = useState(null);

  const [loading, setLoading] = useState(false);
  const [successData, setSuccessData] = useState(null);
  const [error, setError] = useState("");
  
  const [quizDialogOpen, setQuizDialogOpen] = useState(false);
  const [numQuestions, setNumQuestions] = useState(10);

  const navigate = useNavigate();

  const onDrop = (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      setPdfFile(acceptedFiles[0]);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false
  });

  const uploadNote = async () => {
    if (!title.trim()) {
      setError("Please enter a title for the note.");
      return;
    }

    setError("");
    setSuccessData(null);
    setLoading(true);

    try {
      let res;

      if (mode === "text") {
        // Upload text mode
        res = await API.post("upload-note/", {
          title,
          content,
        });
      } else {
        // Upload PDF mode
        const formData = new FormData();
        formData.append("title", title);
        formData.append("file", pdfFile);

        res = await API.post("upload-pdf/", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
      }

      setSuccessData(res.data);
    } catch (err) {
      console.error(err);
      setError(err?.response?.data?.error || "Upload failed.");
    } finally {
      setLoading(false);
    }
  };

  const copyNoteId = () => {
    navigator.clipboard.writeText(successData.note_id.toString());
  };

  return (
    <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
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
        📚 Upload Lecture Notes
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
            sx={{ mb: 3, fontWeight: 600, color: "#333" }}
          >
            📝 Choose Upload Method
          </Typography>

          {/* Mode Switch Buttons */}
          <Stack direction="row" spacing={2} sx={{ mb: 4 }}>
            <Button
              variant={mode === "text" ? "contained" : "outlined"}
              startIcon={<ContentPasteIcon />}
              onClick={() => setMode("text")}
              sx={{
                flex: 1,
                height: "56px",
                borderRadius: 2,
                fontWeight: 600,
                ...(mode === "text" && {
                  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)",
                }),
              }}
            >
              Paste Text
            </Button>

            <Button
              variant={mode === "pdf" ? "contained" : "outlined"}
              startIcon={<UploadFileIcon />}
              onClick={() => setMode("pdf")}
              sx={{
                flex: 1,
                height: "56px",
                borderRadius: 2,
                fontWeight: 600,
                ...(mode === "pdf" && {
                  background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                  boxShadow: "0 4px 15px rgba(102, 126, 234, 0.4)",
                }),
              }}
            >
              Upload PDF
            </Button>
          </Stack>

          <Stack spacing={3}>
            {/* Title Input */}
            <TextField
              label="Lecture Title"
              placeholder="e.g., Introduction to Machine Learning"
              fullWidth
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              sx={{
                "& .MuiOutlinedInput-root": {
                  height: "56px",
                  background: "white",
                  borderRadius: 2,
                  boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                  transition: "all 0.3s ease",
                  "&:hover": {
                    boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
                  },
                },
              }}
            />

            {/* Conditional Inputs */}
            {mode === "text" ? (
              <TextField
                label="Paste Lecture Content"
                placeholder="Paste your lecture notes here..."
                fullWidth
                multiline
                rows={10}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                sx={{
                  "& .MuiOutlinedInput-root": {
                    background: "white",
                    borderRadius: 2,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                    transition: "all 0.3s ease",
                    "&:hover": {
                      boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
                    },
                  },
                }}
              />
            ) : (
              <Box>
                <Paper
                  {...getRootProps()}
                  elevation={0}
                  sx={{
                    p: 6,
                    background: isDragActive ? "#f0fdf4" : "white",
                    borderRadius: 2,
                    border: "2px dashed",
                    borderColor: isDragActive ? "#16a34a" : "#cbd5e0",
                    textAlign: "center",
                    cursor: 'pointer',
                    transition: "all 0.3s ease",
                    "&:hover": {
                      borderColor: "#667eea",
                      background: "#f8f9ff",
                    },
                  }}
                >
                  <input {...getInputProps()} />
                  <UploadFileIcon sx={{ fontSize: 48, color: isDragActive ? "#16a34a" : "#cbd5e0", mb: 2 }} />
                  <Typography variant="h6" color={isDragActive ? "primary" : "textSecondary"} gutterBottom>
                    {isDragActive ? "Drop the PDF here..." : "Drag & Drop PDF here"}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    or click to browse files
                  </Typography>
                </Paper>
                
                {pdfFile && (
                  <Chip
                    label={pdfFile.name}
                    onDelete={() => setPdfFile(null)}
                    color="primary"
                    sx={{ mt: 2, height: 32, fontSize: '0.9rem' }}
                    icon={<CheckCircleIcon />}
                  />
                )}
              </Box>
            )}

            {/* Upload Button */}
            <Button
              variant="contained"
              fullWidth
              onClick={uploadNote}
              disabled={loading || !title.trim() || (mode === 'text' && !content.trim()) || (mode === 'pdf' && !pdfFile)}
              sx={{
                px: 3,
                py: 1.5,
                height: "56px",
              }}
            >
              {loading ? (
                <CircularProgress size={24} color="inherit" />
              ) : (
                "Upload Lecture Note"
              )}
            </Button>
          </Stack>

          {/* Error */}
          {error && (
            <Alert severity="error" sx={{ mt: 3 }}>
              {error}
            </Alert>
          )}

          {/* SUCCESS BLOCK */}
          {successData && (
            <Box sx={{ mt: 4 }}>
              <Divider sx={{ mb: 3 }} />
              <Paper
                elevation={0}
                sx={{
                  p: 3,
                  background: successData.message?.includes("already exists") 
                    ? "linear-gradient(135deg, #fff7ed 0%, #ffedd5 100%)" 
                    : "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)",
                  borderRadius: 3,
                  border: successData.message?.includes("already exists")
                    ? "2px solid #f97316"
                    : "2px solid #10b981",
                }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
                  {successData.message?.includes("already exists") ? (
                    <WarningIcon sx={{ color: "#c2410c", fontSize: 32 }} />
                  ) : (
                    <CheckCircleIcon sx={{ color: "#059669", fontSize: 32 }} />
                  )}
                  <Typography variant="h5" sx={{ 
                    color: successData.message?.includes("already exists") ? "#9a3412" : "#065f46", 
                    fontWeight: 700 
                  }}>
                    {successData.message?.includes("already exists") 
                      ? "Duplicate Content Detected!" 
                      : "Upload Successful!"}
                  </Typography>
                </Box>

                {successData.message?.includes("already exists") && (
                   <Typography variant="body1" sx={{ mb: 2, color: "#9a3412" }}>
                     This content has already been uploaded. Using the original lecture note.
                   </Typography>
                )}

                <Box
                  sx={{
                    p: 2,
                    background: "white",
                    borderRadius: 2,
                    mb: 2,
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
                    <Typography variant="body1" sx={{ fontWeight: 600 }}>
                      Lecture Note ID:
                    </Typography>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                      <Chip
                        label={successData.note_id}
                        color="primary"
                        sx={{ fontWeight: 700, fontSize: "16px" }}
                      />
                      <Tooltip title="Copy Note ID">
                        <IconButton onClick={copyNoteId} size="small">
                          <ContentCopyIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>

                  <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                    <TopicIcon sx={{ color: "#667eea" }} />
                    <Typography variant="body1">
                      Extracted Topics: <strong>{successData.topics?.length || 0}</strong>
                    </Typography>
                  </Box>
                </Box>

                <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
                  <Button
                    variant="contained"
                    fullWidth
                    startIcon={<QuizIcon />}
                    onClick={() => setQuizDialogOpen(true)}
                    sx={{
                      height: "48px",
                    }}
                  >
                    Start Quiz
                  </Button>
                  <Button
                    variant="outlined"
                    fullWidth
                    onClick={() => {
                      setSuccessData(null);
                      setTitle("");
                      setContent("");
                      setPdfFile(null);
                    }}
                    sx={{
                      height: "48px",
                      fontWeight: 600,
                    }}
                  >
                    Upload Another
                  </Button>
                </Stack>
              </Paper>
            </Box>
          )}

          {/* Helper Text */}
          <Box
            sx={{
              mt: 3,
              p: 2,
              background: "rgba(102, 126, 234, 0.1)",
              borderRadius: 2,
              borderLeft: "4px solid #667eea",
            }}
          >
            <Typography variant="body2" color="text.secondary">
              💡 <strong>Tip:</strong> Upload your lecture notes as text or PDF. Our AI will automatically extract topics and generate questions for you!
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Quiz Configuration Dialog */}
      <Dialog open={quizDialogOpen} onClose={() => setQuizDialogOpen(false)}>
        <DialogTitle sx={{ fontWeight: 'bold' }}>🎯 Start Quiz</DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2, mt: 1 }}>
            How many questions would you like to attempt from this lecture note?
          </Typography>
          <TextField
            autoFocus
            margin="dense"
            label="Number of Questions"
            type="number"
            fullWidth
            variant="outlined"
            value={numQuestions}
            onChange={(e) => setNumQuestions(e.target.value)}
            InputProps={{ inputProps: { min: 1, max: 50 } }}
          />
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setQuizDialogOpen(false)} color="inherit">
            Cancel
          </Button>
          <Button 
            onClick={() => {
              setQuizDialogOpen(false);
              navigate(`/quiz?noteId=${successData.note_id}&n=${numQuestions}`);
            }} 
            variant="contained"
            disabled={!numQuestions || numQuestions < 1}
          >
            Start Quiz
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
