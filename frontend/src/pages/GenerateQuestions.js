import React, { useState } from "react";
import {
  Card,
  Typography,
  Box,
  Container,
  TextField,
  Button,
  CircularProgress,
  Grid,
  Chip,
  InputAdornment,
  Alert,
  IconButton,
  Tooltip,
  Stack,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  Paper,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import API from "../api/api";
import AutoStoriesIcon from "@mui/icons-material/AutoStories";
import LightbulbIcon from "@mui/icons-material/Lightbulb";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import ClearIcon from "@mui/icons-material/Clear";
import SendIcon from "@mui/icons-material/Send";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import EditIcon from "@mui/icons-material/Edit";
import QuizIcon from "@mui/icons-material/Quiz";
import InfoIcon from "@mui/icons-material/Info";
import SaveIcon from "@mui/icons-material/Save";
import CloseIcon from "@mui/icons-material/Close";
import LectureSelect from "../components/LectureSelect";

const cleanOption = (text) => {
  if (!text) return "";
  // Remove any leading A), B), C), D), A., B., (A), (B) etc, even if repeated
  return text.replace(/^([A-D][\\.\\)]\\s*|\\([A-D]\\)\\s*)+/gi, "").trim();
};

// Unified control dimensions and spacing
const CONTROL_HEIGHT = 56;
const controlSx = {
  "& .MuiInputBase-root": {
    height: CONTROL_HEIGHT,
    background: "white",
    borderRadius: 2,
    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
    transition: "all 0.3s ease",
    "&:hover": {
      boxShadow: "0 4px 12px rgba(0,0,0,0.12)",
    },
  },
  "& .MuiInputBase-input": {
    py: 1.5,
    px: 1.5,
  },
};

const buttonSx = {
  minHeight: CONTROL_HEIGHT,
  px: 3,
  borderRadius: 2,
};

function GenerateQuestions() {
  const [noteId, setNoteId] = useState("");
  const [maxGenerate, setMaxGenerate] = useState("");
  const [quizCount, setQuizCount] = useState("");
  const [questions, setQuestions] = useState([]);
  const [showQuestions, setShowQuestions] = useState(false);
  const [expandedQuestion, setExpandedQuestion] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  // Edit dialog state
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingQuestion, setEditingQuestion] = useState(null);
  const [editForm, setEditForm] = useState({
    question_text: "",
    option_a: "",
    option_b: "",
    option_c: "",
    option_d: "",
    correct_option: "A",
    explanation: "",
  });
  const [saving, setSaving] = useState(false);

  // Quiz Dialog State
  const [quizDialogOpen, setQuizDialogOpen] = useState(false);
  const [numQuestions, setNumQuestions] = useState(10);

  const generate = async () => {
    setError("");
    setSuccess(false);

    if (!noteId || String(noteId).trim() === "") {
      setError("Please enter a Note ID.");
      return;
    }

    if (!maxGenerate || String(maxGenerate).trim() === "" || Number(maxGenerate) <= 0) {
      setError("Please enter a valid number for max question generation.");
      return;
    }

    try {
      setLoading(true);

      const res = await API.post("generate-mcqs/", {
        note_id: noteId,
        count: maxGenerate,
      });

      setQuestions(res.data.questions || res.data.mcqs || []);
      setSuccess(true);
    } catch (err) {
      console.error(err);
      setError(
        (err?.response?.data && JSON.stringify(err.response.data)) ||
          err.message ||
          "Failed to generate questions"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      generate();
    }
  };

  const clearInput = () => {
    setNoteId("");
    setMaxGenerate("");
    setError("");
    setSuccess(false);
  };

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpandedQuestion(isExpanded ? panel : false);
  };

  const handleEditClick = (question, index) => {
    setEditingQuestion({ ...question, index });
    setEditForm({
      question_text: question.question || question.question_text || "",
      option_a: question.option_a || "",
      option_b: question.option_b || "",
      option_c: question.option_c || "",
      option_d: question.option_d || "",
      correct_option:
        question.correct_option || question.answer || question.correct || "A",
      explanation: question.explanation || "",
    });
    setEditDialogOpen(true);
  };

  const handleEditFormChange = (field, value) => {
    setEditForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSaveEdit = async () => {
    if (!editingQuestion) return;

    setSaving(true);
    try {
      const token = localStorage.getItem("access_token");
      const response = await API.put(
        `questions/${editingQuestion.id}/update/`,
        editForm,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Update the question in the local state
      const updatedQuestions = [...questions];
      updatedQuestions[editingQuestion.index] = response.data.question;
      setQuestions(updatedQuestions);

      setEditDialogOpen(false);
      setSuccess(true);
      setError("");

      // Show success message
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      console.error("Failed to update question:", err);
      setError("Failed to update question. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 5, mb: 5 }}>
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
        💡 Generate Questions
      </Typography>

      {/* INPUT CARD */}
      <Card
        sx={{
          p: 4,
          background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
          borderRadius: 3,
          mb: 4,
        }}
      >
        <CardContent sx={{ p: 0 }}>
          <Typography
            variant="h6"
            sx={{ mb: 3, fontWeight: 600, color: "#333" }}
          >
            📝 Enter Details
          </Typography>

          <Stack direction="column" spacing={3} alignItems="stretch">
            {/* NOTE ID */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{ mb: 1, color: "#4a5568", fontWeight: 700 }}
              >
                Lecture Note
              </Typography>
              <LectureSelect value={noteId} onChange={(v) => setNoteId(v)} />
            </Box>

            {/* MAX GENERATE INPUT */}
            <Box sx={{ width: "100%" }}>
              <Typography sx={{ fontWeight: 700, color: "#4a5568", mb: 1 }}>
                Max Questions to Generate
              </Typography>
              <TextField
                placeholder="e.g., 20, 30, 50"
                variant="outlined"
                value={maxGenerate}
                onChange={(e) => setMaxGenerate(e.target.value)}
                onKeyPress={handleKeyPress}
                fullWidth
                sx={controlSx}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <LightbulbIcon sx={{ color: "#667eea", ml: 1 }} />
                    </InputAdornment>
                  ),
                }}
              />
            </Box>

            {/* GENERATE BUTTON */}
            <Button
              variant="contained"
              color="primary"
              onClick={generate}
              startIcon={loading ? null : <SendIcon />}
              disabled={loading || !noteId || !maxGenerate}
              sx={{
                ...buttonSx,
              }}
            >
              {loading ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                "Generate Questions"
              )}
            </Button>
          </Stack>

          {/* ERROR */}
          {error ? (
            <Box sx={{ mt: 3 }}>
              <Alert severity="error">{error}</Alert>
            </Box>
          ) : null}

          {/* SUCCESS */}
          {success ? (
            <Box sx={{ mt: 3 }}>
              <Alert severity="success" icon={<CheckCircleIcon />}>
                {questions.length} questions generated successfully!
              </Alert>

              {/* Action Buttons */}
              <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={2}
                sx={{ mt: 3 }}
              >
                <Button
                  variant="outlined"
                  color="primary"
                  onClick={() => setShowQuestions(!showQuestions)}
                  sx={{ ...buttonSx, flex: 1 }}
                  startIcon={<QuizIcon />}
                >
                  {showQuestions ? "Hide Questions" : "Display Questions"}
                </Button>

                <Button
                  variant="contained"
                  onClick={() => setQuizDialogOpen(true)}
                  sx={{
                    ...buttonSx,
                    flex: 1,
                  }}
                >
                  Go to Quiz
                </Button>
              </Stack>
            </Box>
          ) : null}

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
              💡 <strong>Tip:</strong> Enter the Note ID and specify how many
              questions to generate. Our AI will create MCQs from your lecture
              notes!
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Show Generated MCQs with Improved UI */}
      {showQuestions && (
        <Box>
          <Box
            sx={{
              mb: 3,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              Generated Questions ({questions.length})
            </Typography>
            <Chip
              label={`${questions.length} Questions`}
              color="primary"
              icon={<QuizIcon />}
            />
          </Box>

          {questions.map((q, i) => (
            <Accordion
              key={i}
              expanded={expandedQuestion === `panel${i}`}
              onChange={handleAccordionChange(`panel${i}`)}
              sx={{
                mb: 2,
                borderRadius: "12px !important",
                overflow: "hidden",
                "&:before": { display: "none" },
                boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                "&.Mui-expanded": {
                  boxShadow: "0 8px 24px rgba(102, 126, 234, 0.15)",
                },
              }}
            >
              <AccordionSummary
                expandIcon={<ExpandMoreIcon />}
                sx={{
                  background:
                    "linear-gradient(135deg, #667eea15 0%, #764ba215 100%)",
                  "&:hover": {
                    background:
                      "linear-gradient(135deg, #667eea25 0%, #764ba225 100%)",
                  },
                  borderRadius: "12px",
                  minHeight: "72px",
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 2,
                    width: "100%",
                    pr: 2,
                  }}
                >
                  <Box
                    sx={{
                      width: 40,
                      height: 40,
                      borderRadius: "10px",
                      background:
                        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                      color: "white",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontWeight: 700,
                      fontSize: "16px",
                    }}
                  >
                    {i + 1}
                  </Box>
                  <Box sx={{ flex: 1 }}>
                    <Typography
                      variant="body1"
                      sx={{ fontWeight: 600, color: "#1a202c" }}
                    >
                      {q.question || q.question_text}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.5,
                        mt: 0.5,
                      }}
                    >
                      <InfoIcon sx={{ fontSize: 14 }} />
                      Topic: {q.topic || "General"}
                    </Typography>
                  </Box>
                </Box>
              </AccordionSummary>

              <AccordionDetails sx={{ p: 3, bgcolor: "#fafbfc" }}>
                <Box>
                  {/* Question Text */}
                  <Typography
                    variant="h6"
                    sx={{ mb: 3, fontWeight: 600, color: "#2d3748" }}
                  >
                    {q.question || q.question_text}
                  </Typography>

                  <Divider sx={{ mb: 3 }} />

                  {/* Display Options */}
                  <Typography
                    variant="subtitle2"
                    sx={{ mb: 2, color: "#64748b", fontWeight: 600 }}
                  >
                    Answer Options:
                  </Typography>
                  <Stack spacing={1.5}>
                    {["A", "B", "C", "D"].map((letter) => {
                      const optionKey = `option_${letter.toLowerCase()}`;
                      const optionText =
                        q[optionKey] ||
                        (q.options && q.options[letter.charCodeAt(0) - 65]);

                      if (!optionText) return null;

                      const isCorrect =
                        (q.correct_option || q.answer || q.correct) === letter;

                      return (
                        <Paper
                          key={letter}
                          elevation={0}
                          sx={{
                            p: 2,
                            borderRadius: "12px",
                            border: "2px solid",
                            borderColor: isCorrect ? "#10b981" : "#e2e8f0",
                            bgcolor: isCorrect ? "#d1fae5" : "white",
                            transition: "all 0.2s",
                            "&:hover": {
                              borderColor: isCorrect ? "#059669" : "#cbd5e0",
                              transform: "translateX(4px)",
                            },
                          }}
                        >
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 2,
                                flex: 1,
                              }}
                            >
                              <Box
                                sx={{
                                  width: 32,
                                  height: 32,
                                  borderRadius: "8px",
                                  background: isCorrect
                                    ? "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                                    : "linear-gradient(135deg, #94a3b8 0%, #64748b 100%)",
                                  color: "white",
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "center",
                                  fontWeight: 700,
                                  fontSize: "14px",
                                }}
                              >
                                {letter}
                              </Box>
                              <Typography
                                variant="body1"
                                sx={{
                                  color: isCorrect ? "#065f46" : "#1e293b",
                                  fontWeight: isCorrect ? 600 : 400,
                                }}
                              >
                                {cleanOption(optionText)}
                              </Typography>
                            </Box>
                            {isCorrect && (
                              <Chip
                                label="Correct Answer"
                                size="small"
                                sx={{
                                  bgcolor: "#10b981",
                                  color: "white",
                                  fontWeight: 600,
                                  height: 28,
                                }}
                                icon={
                                  <CheckCircleIcon
                                    sx={{ color: "white !important" }}
                                  />
                                }
                              />
                            )}
                          </Box>
                        </Paper>
                      );
                    })}
                  </Stack>

                  {/* Explanation if available */}
                  {q.explanation && (
                    <Box sx={{ mt: 3 }}>
                      <Divider sx={{ mb: 2 }} />
                      <Paper
                        elevation={0}
                        sx={{
                          p: 2.5,
                          bgcolor: "#eff6ff",
                          borderRadius: "12px",
                          borderLeft: "4px solid #3b82f6",
                        }}
                      >
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 1,
                            mb: 1,
                          }}
                        >
                          <InfoIcon sx={{ color: "#3b82f6", fontSize: 20 }} />
                          <Typography
                            variant="subtitle2"
                            sx={{ fontWeight: 700, color: "#1e40af" }}
                          >
                            Explanation
                          </Typography>
                        </Box>
                        <Typography
                          variant="body2"
                          sx={{ color: "#1e3a8a", lineHeight: 1.7 }}
                        >
                          {q.explanation}
                        </Typography>
                      </Paper>
                    </Box>
                  )}

                  {/* Edit Button */}
                  <Box
                    sx={{ mt: 3, display: "flex", justifyContent: "flex-end" }}
                  >
                    <Button
                      size="small"
                      variant="outlined"
                      startIcon={<EditIcon />}
                      onClick={() => handleEditClick(q, i)}
                      sx={{
                        borderRadius: "8px",
                        borderColor: "#667eea",
                        color: "#667eea",
                        "&:hover": {
                          borderColor: "#764ba2",
                          bgcolor: "rgba(102, 126, 234, 0.05)",
                        },
                      }}
                    >
                      Edit Question
                    </Button>
                  </Box>
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {/* Edit Question Dialog */}
      <Dialog
        open={editDialogOpen}
        onClose={() => setEditDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle
          sx={{
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            color: "white",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <EditIcon />
            <Typography variant="h6" fontWeight={600}>
              Edit Question
            </Typography>
          </Box>
          <IconButton
            onClick={() => setEditDialogOpen(false)}
            sx={{ color: "white" }}
          >
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ mt: 3 }}>
          <Stack spacing={3}>
            <TextField
              label="Question Text"
              fullWidth
              multiline
              rows={3}
              value={editForm.question_text}
              onChange={(e) =>
                handleEditFormChange("question_text", e.target.value)
              }
              variant="outlined"
            />

            <Divider />

            <Typography
              variant="subtitle2"
              color="text.secondary"
              fontWeight={600}
            >
              Answer Options
            </Typography>

            <TextField
              label="Option A"
              fullWidth
              value={editForm.option_a}
              onChange={(e) => handleEditFormChange("option_a", e.target.value)}
            />

            <TextField
              label="Option B"
              fullWidth
              value={editForm.option_b}
              onChange={(e) => handleEditFormChange("option_b", e.target.value)}
            />

            <TextField
              label="Option C"
              fullWidth
              value={editForm.option_c}
              onChange={(e) => handleEditFormChange("option_c", e.target.value)}
            />

            <TextField
              label="Option D"
              fullWidth
              value={editForm.option_d}
              onChange={(e) => handleEditFormChange("option_d", e.target.value)}
            />

            <FormControl fullWidth>
              <InputLabel>Correct Answer</InputLabel>
              <Select
                value={editForm.correct_option}
                label="Correct Answer"
                onChange={(e) =>
                  handleEditFormChange("correct_option", e.target.value)
                }
              >
                <MenuItem value="A">A</MenuItem>
                <MenuItem value="B">B</MenuItem>
                <MenuItem value="C">C</MenuItem>
                <MenuItem value="D">D</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Explanation (Optional)"
              fullWidth
              multiline
              rows={3}
              value={editForm.explanation}
              onChange={(e) =>
                handleEditFormChange("explanation", e.target.value)
              }
            />
          </Stack>
        </DialogContent>

        <DialogActions sx={{ p: 3, gap: 1 }}>
          <Button
            onClick={() => setEditDialogOpen(false)}
            variant="outlined"
            disabled={saving}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSaveEdit}
            variant="contained"
            startIcon={
              saving ? (
                <CircularProgress size={20} color="inherit" />
              ) : (
                <SaveIcon />
              )
            }
            disabled={saving}
            sx={{ fontWeight: 600 }}
          >
            {saving ? "Saving..." : "Save Changes"}
          </Button>
        </DialogActions>
      </Dialog>
      {/* Quiz Configuration Dialog */}
      <Dialog open={quizDialogOpen} onClose={() => setQuizDialogOpen(false)}>
        <DialogTitle sx={{ fontWeight: "bold" }}>🎯 Start Quiz</DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2, mt: 1 }}>
            How many questions would you like to attempt?
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
              window.location.href = `/quiz?noteId=${noteId}&n=${numQuestions}`;
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

export default GenerateQuestions;
