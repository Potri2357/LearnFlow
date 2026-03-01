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
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import API from "../api/api";
import AutoStoriesIcon from "@mui/icons-material/AutoStories";
import LightbulbIcon from "@mui/icons-material/Lightbulb";
import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import SendIcon from "@mui/icons-material/Send";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import EditIcon from "@mui/icons-material/Edit";
import QuizIcon from "@mui/icons-material/Quiz";
import InfoIcon from "@mui/icons-material/Info";
import SaveIcon from "@mui/icons-material/Save";
import CloseIcon from "@mui/icons-material/Close";
import FilterListIcon from "@mui/icons-material/FilterList";
import LectureSelect from "../components/LectureSelect";

// Markdown renderer component with consistent styling
const MarkdownText = ({ children, sx = {} }) => (
  <Box
    sx={{
      "& p": { margin: 0, lineHeight: 1.7 },
      "& strong": { fontWeight: 700 },
      "& em": { fontStyle: "italic" },
      "& code": {
        fontFamily: "monospace",
        background: (theme) =>
          theme.palette.mode === "dark"
            ? "rgba(255,255,255,0.1)"
            : "rgba(0,0,0,0.06)",
        px: 0.5,
        py: 0.25,
        borderRadius: "4px",
        fontSize: "0.85em",
      },
      "& pre": {
        background: (theme) =>
          theme.palette.mode === "dark"
            ? "rgba(255,255,255,0.05)"
            : "rgba(0,0,0,0.04)",
        p: 1,
        borderRadius: 1,
        overflowX: "auto",
      },
      ...sx,
    }}
  >
    <ReactMarkdown remarkPlugins={[remarkGfm]}>{children || ""}</ReactMarkdown>
  </Box>
);

const CONTROL_HEIGHT = 56;
const buttonSx = { minHeight: CONTROL_HEIGHT, px: 3, borderRadius: 2 };

function GenerateQuestions() {
  const [noteId, setNoteId] = useState("");
  const [maxGenerate, setMaxGenerate] = useState("");
  const [questions, setQuestions] = useState([]);
  const [showQuestions, setShowQuestions] = useState(false);
  const [expandedQuestion, setExpandedQuestion] = useState(false);
  const [filterTopic, setFilterTopic] = useState("All");

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
      setError("Please select a Lecture Note.");
      return;
    }
    if (!maxGenerate || Number(maxGenerate) <= 0) {
      setError("Please enter a valid number of questions to generate.");
      return;
    }

    try {
      setLoading(true);
      const res = await API.post("generate-mcqs/", {
        note_id: noteId,
        count: Number(maxGenerate),
      });

      const qs = res.data.questions || res.data.mcqs || [];
      setQuestions(qs);
      setSuccess(true);
      setShowQuestions(true);
      setFilterTopic("All");
    } catch (err) {
      console.error("Generation error:", err);
      const errData = err?.response?.data;
      setError(
        typeof errData === "string"
          ? errData
          : errData?.error || errData?.details || err.message || "Failed to generate questions"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpandedQuestion(isExpanded ? panel : false);
  };

  const handleEditClick = (question, index) => {
    setEditingQuestion({ ...question, index });
    setEditForm({
      question_text: question.question_text || "",
      option_a: question.option_a || "",
      option_b: question.option_b || "",
      option_c: question.option_c || "",
      option_d: question.option_d || "",
      correct_option: question.correct_option || "A",
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
      const updatedQuestions = [...questions];
      updatedQuestions[editingQuestion.index] = response.data.question;
      setQuestions(updatedQuestions);
      setEditDialogOpen(false);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      console.error("Failed to update question:", err);
      setError("Failed to update question. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  // Get unique topics for filter
  const topics = ["All", ...new Set(questions.map((q) => q.topic).filter(Boolean))];

  const filteredQuestions =
    filterTopic === "All"
      ? questions
      : questions.filter((q) => q.topic === filterTopic);

  return (
    <Container maxWidth="lg" sx={{ mt: 5, mb: 5 }}>
      <Typography
        variant="h3"
        gutterBottom
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(135deg, #038C7F 0%, #027373 100%)",
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
          background: (theme) =>
            theme.palette.mode === "dark"
              ? "background.paper"
              : "linear-gradient(135deg, #f0fdf4 0%, #ccfbf1 100%)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
          borderRadius: 3,
          mb: 4,
          border: "1px solid",
          borderColor: "divider",
        }}
      >
        <CardContent sx={{ p: 0 }}>
          <Typography variant="h6" sx={{ mb: 3, fontWeight: 600, color: "text.primary" }}>
            📝 Enter Details
          </Typography>

          <Stack direction="column" spacing={3} alignItems="stretch">
            {/* NOTE SELECT */}
            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1, color: "text.secondary", fontWeight: 700 }}>
                Lecture Note
              </Typography>
              <LectureSelect value={noteId} onChange={(v) => setNoteId(v)} />
            </Box>

            {/* MAX COUNT */}
            <Box sx={{ width: "100%" }}>
              <Typography sx={{ fontWeight: 700, color: "text.secondary", mb: 1 }}>
                Max Questions to Generate
              </Typography>
              <TextField
                placeholder="e.g., 20, 30, 50"
                variant="outlined"
                value={maxGenerate}
                type="number"
                onChange={(e) => setMaxGenerate(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && generate()}
                fullWidth
                sx={{
                  "& .MuiOutlinedInput-root": {
                    height: CONTROL_HEIGHT,
                    bgcolor: "background.paper",
                    borderRadius: 2,
                    boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
                    transition: "all 0.3s ease",
                    "&:hover": { boxShadow: "0 4px 12px rgba(0,0,0,0.12)" },
                  },
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <LightbulbIcon sx={{ color: "primary.main", ml: 1 }} />
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
              sx={{ ...buttonSx, bgcolor: "primary.main", "&:hover": { bgcolor: "primary.dark" } }}
            >
              {loading ? <CircularProgress size={20} color="inherit" /> : "Generate Questions"}
            </Button>
          </Stack>

          {/* ERROR */}
          {error && (
            <Box sx={{ mt: 3 }}>
              <Alert severity="error" onClose={() => setError("")}>{error}</Alert>
            </Box>
          )}

          {/* SUCCESS Actions */}
          {success && (
            <Box sx={{ mt: 3 }}>
              <Alert severity="success" icon={<CheckCircleIcon />}>
                {questions.length} questions generated successfully!
              </Alert>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={2} sx={{ mt: 3 }}>
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
                    bgcolor: "secondary.main",
                    "&:hover": { bgcolor: "secondary.dark" },
                  }}
                >
                  Go to Quiz
                </Button>
              </Stack>
            </Box>
          )}

          {/* Tip */}
          <Box
            sx={{
              mt: 3,
              p: 2,
              background: (theme) =>
                theme.palette.mode === "dark"
                  ? "rgba(3, 140, 127, 0.1)"
                  : "rgba(102, 126, 234, 0.1)",
              borderRadius: 2,
              borderLeft: "4px solid",
              borderColor: "primary.main",
            }}
          >
            <Typography variant="body2" color="text.secondary">
              💡 <strong>Tip:</strong> Select a lecture note and specify how many questions to
              generate. Our AI will create MCQs from your lecture notes with markdown formatting!
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* QUESTIONS LIST */}
      {showQuestions && questions.length > 0 && (
        <Box>
          {/* Header + Filter */}
          <Box
            sx={{
              mb: 3,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 2,
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                Generated Questions
              </Typography>
              <Chip label={`${filteredQuestions.length} / ${questions.length}`} color="primary" icon={<QuizIcon />} />
            </Box>

            {topics.length > 1 && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                <FilterListIcon sx={{ color: "text.secondary", fontSize: 18 }} />
                {topics.map((t) => (
                  <Chip
                    key={t}
                    label={t}
                    size="small"
                    variant={filterTopic === t ? "filled" : "outlined"}
                    color={filterTopic === t ? "primary" : "default"}
                    onClick={() => setFilterTopic(t)}
                    sx={{ cursor: "pointer" }}
                  />
                ))}
              </Box>
            )}
          </Box>

          {filteredQuestions.map((q, i) => {
            const globalIdx = questions.indexOf(q);
            return (
              <Accordion
                key={q.id || i}
                expanded={expandedQuestion === `panel${i}`}
                onChange={handleAccordionChange(`panel${i}`)}
                sx={{
                  mb: 2,
                  borderRadius: "12px !important",
                  overflow: "hidden",
                  "&:before": { display: "none" },
                  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                  bgcolor: "background.paper",
                  "&.Mui-expanded": { boxShadow: "0 8px 24px rgba(3, 140, 127, 0.15)" },
                }}
              >
                <AccordionSummary
                  expandIcon={<ExpandMoreIcon />}
                  sx={{
                    background: (theme) =>
                      theme.palette.mode === "dark"
                        ? "rgba(255, 255, 255, 0.05)"
                        : "linear-gradient(135deg, rgba(3, 140, 127, 0.05) 0%, rgba(2, 115, 115, 0.05) 100%)",
                    "&:hover": {
                      background: (theme) =>
                        theme.palette.mode === "dark"
                          ? "rgba(255, 255, 255, 0.1)"
                          : "linear-gradient(135deg, rgba(3, 140, 127, 0.1) 0%, rgba(2, 115, 115, 0.1) 100%)",
                    },
                    borderRadius: "12px",
                    minHeight: "72px",
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center", gap: 2, width: "100%", pr: 2 }}>
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        minWidth: 40,
                        borderRadius: "10px",
                        background: "linear-gradient(135deg, #038C7F 0%, #027373 100%)",
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
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Box
                        sx={{
                          fontWeight: 600,
                          color: "text.primary",
                          fontSize: "0.95rem",
                          lineHeight: 1.5,
                          "& p": { margin: 0 },
                          "& strong": { fontWeight: 800 },
                        }}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {q.question_text || ""}
                        </ReactMarkdown>
                      </Box>
                      {q.topic && (
                        <Chip
                          label={q.topic}
                          size="small"
                          sx={{ mt: 0.5, height: 20, fontSize: "0.7rem" }}
                          variant="outlined"
                          color="primary"
                        />
                      )}
                    </Box>
                  </Box>
                </AccordionSummary>

                <AccordionDetails
                  sx={{
                    p: 3,
                    bgcolor: (theme) =>
                      theme.palette.mode === "dark" ? "background.default" : "#fafbfc",
                  }}
                >
                  <Box>
                    {/* Full Question with markdown */}
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2.5,
                        mb: 3,
                        borderRadius: "10px",
                        bgcolor: (theme) =>
                          theme.palette.mode === "dark"
                            ? "rgba(255,255,255,0.03)"
                            : "rgba(3, 140, 127, 0.04)",
                        border: "1px solid",
                        borderColor: "divider",
                      }}
                    >
                      <Typography variant="caption" color="primary.main" sx={{ fontWeight: 700, mb: 1, display: "block" }}>
                        QUESTION
                      </Typography>
                      <MarkdownText sx={{ fontSize: "1rem", fontWeight: 600, color: "text.primary" }}>
                        {q.question_text}
                      </MarkdownText>
                    </Paper>

                    <Typography variant="subtitle2" sx={{ mb: 2, color: "text.secondary", fontWeight: 600 }}>
                      Answer Options:
                    </Typography>

                    <Stack spacing={1.5}>
                      {["A", "B", "C", "D"].map((letter) => {
                        const optText = q[`option_${letter.toLowerCase()}`];
                        if (!optText) return null;
                        const isCorrect = (q.correct_option || "").toUpperCase() === letter;

                        return (
                          <Paper
                            key={letter}
                            elevation={0}
                            sx={{
                              p: 2,
                              borderRadius: "12px",
                              border: "2px solid",
                              borderColor: isCorrect ? "success.main" : "divider",
                              bgcolor: isCorrect
                                ? (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "rgba(16, 185, 129, 0.2)"
                                      : "#d1fae5"
                                : "background.paper",
                              transition: "all 0.2s",
                              "&:hover": {
                                borderColor: isCorrect ? "success.dark" : "text.disabled",
                                transform: "translateX(4px)",
                              },
                            }}
                          >
                            <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
                              <Box sx={{ display: "flex", alignItems: "flex-start", gap: 2, flex: 1 }}>
                                <Box
                                  sx={{
                                    width: 32,
                                    height: 32,
                                    minWidth: 32,
                                    borderRadius: "8px",
                                    background: isCorrect
                                      ? "linear-gradient(135deg, #10b981 0%, #059669 100%)"
                                      : (theme) =>
                                          theme.palette.mode === "dark"
                                            ? "rgba(255,255,255,0.1)"
                                            : "linear-gradient(135deg, #94a3b8 0%, #64748b 100%)",
                                    color: "white",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    fontWeight: 700,
                                    fontSize: "14px",
                                    mt: 0.25,
                                  }}
                                >
                                  {letter}
                                </Box>
                                <Box sx={{ flex: 1 }}>
                                  <MarkdownText
                                    sx={{
                                      color: isCorrect ? "success.dark" : "text.primary",
                                      fontWeight: isCorrect ? 600 : 400,
                                      "& p": { margin: 0 },
                                    }}
                                  >
                                    {optText}
                                  </MarkdownText>
                                </Box>
                              </Box>
                              {isCorrect && (
                                <Chip
                                  label="Correct"
                                  size="small"
                                  sx={{
                                    bgcolor: "success.main",
                                    color: "white",
                                    fontWeight: 600,
                                    height: 24,
                                    ml: 1,
                                    flexShrink: 0,
                                  }}
                                  icon={<CheckCircleIcon sx={{ color: "white !important", fontSize: "14px !important" }} />}
                                />
                              )}
                            </Box>
                          </Paper>
                        );
                      })}
                    </Stack>

                    {/* Explanation */}
                    {q.explanation && (
                      <Box sx={{ mt: 3 }}>
                        <Divider sx={{ mb: 2 }} />
                        <Paper
                          elevation={0}
                          sx={{
                            p: 2.5,
                            bgcolor: (theme) =>
                              theme.palette.mode === "dark" ? "rgba(59, 130, 246, 0.1)" : "#eff6ff",
                            borderRadius: "12px",
                            borderLeft: "4px solid",
                            borderColor: "info.main",
                          }}
                        >
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1 }}>
                            <InfoIcon sx={{ color: "info.main", fontSize: 20 }} />
                            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "info.dark" }}>
                              Explanation
                            </Typography>
                          </Box>
                          <MarkdownText sx={{ color: "info.dark", "& p": { margin: 0, lineHeight: 1.7 } }}>
                            {q.explanation}
                          </MarkdownText>
                        </Paper>
                      </Box>
                    )}

                    {/* Edit button */}
                    <Box sx={{ mt: 3, display: "flex", justifyContent: "flex-end" }}>
                      <Button
                        size="small"
                        variant="outlined"
                        startIcon={<EditIcon />}
                        onClick={() => handleEditClick(q, globalIdx)}
                        sx={{
                          borderRadius: "8px",
                          borderColor: "primary.main",
                          color: "primary.main",
                          "&:hover": { borderColor: "primary.dark", bgcolor: "action.hover" },
                        }}
                      >
                        Edit Question
                      </Button>
                    </Box>
                  </Box>
                </AccordionDetails>
              </Accordion>
            );
          })}
        </Box>
      )}

      {/* EDIT DIALOG */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle
          sx={{
            background: "linear-gradient(135deg, #038C7F 0%, #027373 100%)",
            color: "white",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <EditIcon />
            <Typography variant="h6" fontWeight={600}>Edit Question</Typography>
          </Box>
          <IconButton onClick={() => setEditDialogOpen(false)} sx={{ color: "white" }}>
            <CloseIcon />
          </IconButton>
        </DialogTitle>

        <DialogContent sx={{ mt: 3 }}>
          <Stack spacing={3}>
            <TextField
              label="Question Text (Markdown supported)"
              fullWidth
              multiline
              rows={3}
              value={editForm.question_text}
              onChange={(e) => handleEditFormChange("question_text", e.target.value)}
              variant="outlined"
            />
            <Divider />
            <Typography variant="subtitle2" color="text.secondary" fontWeight={600}>
              Answer Options (Markdown supported)
            </Typography>
            {["a", "b", "c", "d"].map((letter) => (
              <TextField
                key={letter}
                label={`Option ${letter.toUpperCase()}`}
                fullWidth
                value={editForm[`option_${letter}`]}
                onChange={(e) => handleEditFormChange(`option_${letter}`, e.target.value)}
              />
            ))}
            <FormControl fullWidth>
              <InputLabel>Correct Answer</InputLabel>
              <Select
                value={editForm.correct_option}
                label="Correct Answer"
                onChange={(e) => handleEditFormChange("correct_option", e.target.value)}
              >
                {["A", "B", "C", "D"].map((l) => (
                  <MenuItem key={l} value={l}>{l}</MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label="Explanation (Optional — Markdown supported)"
              fullWidth
              multiline
              rows={3}
              value={editForm.explanation}
              onChange={(e) => handleEditFormChange("explanation", e.target.value)}
            />
          </Stack>
        </DialogContent>

        <DialogActions sx={{ p: 3, gap: 1 }}>
          <Button onClick={() => setEditDialogOpen(false)} variant="outlined" disabled={saving}>
            Cancel
          </Button>
          <Button
            onClick={handleSaveEdit}
            variant="contained"
            startIcon={saving ? <CircularProgress size={20} color="inherit" /> : <SaveIcon />}
            disabled={saving}
            sx={{ fontWeight: 600 }}
          >
            {saving ? "Saving..." : "Save Changes"}
          </Button>
        </DialogActions>
      </Dialog>

      {/* QUIZ CONFIG DIALOG */}
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
          <Button onClick={() => setQuizDialogOpen(false)} color="inherit">Cancel</Button>
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
