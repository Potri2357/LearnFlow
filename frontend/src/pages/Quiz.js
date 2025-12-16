// src/pages/Quiz.js
import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import './QuizComplete.css';
import {
  Container,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  TextField,
  Box,
  CircularProgress,
  Stack,
  Paper,
  Chip,
  IconButton,
} from "@mui/material";
import TimerIcon from '@mui/icons-material/Timer';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

const cleanOption = (text) => {
  if (!text) return "";
  // Remove any leading A), B), C), D), A., B., (A), (B) etc, even if repeated
  return text.replace(/^([A-D][\.\)]\s*|\([A-D]\)\s*)+/gi, "").trim();
};

function Quiz() {
  const { api: API } = useAuth();
  const [questions, setQuestions] = useState([]);
  const [idx, setIdx] = useState(0);
  const [selected, setSelected] = useState("");
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [noteId, setNoteId] = useState(null);
  const location = useLocation();
  const navigate = useNavigate();
  const [timer, setTimer] = useState(30);
  const [numQuestions, setNumQuestions] = useState(10);
  const [generating, setGenerating] = useState(false);
  const [userAnswers, setUserAnswers] = useState([]); // Track all answers for review

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError("");
      const searchParams = new URLSearchParams(location.search);
      const noteIdParam = searchParams.get("noteId");
      const n = parseInt(searchParams.get("n") || "10", 10);
      
      if (!noteIdParam) {
        setError("No lecture note ID provided. Please select a lecture note first.");
        setLoading(false);
        return;
      }
      
      setNoteId(noteIdParam);
      
      try {
        // 1. Try to fetch existing questions
        let res = await API.get(`quiz/${noteIdParam}/?n=${n}`);
        let fetchedQuestions = res.data.questions || [];
        
        // 2. If we don't have enough questions, generate more
        if (fetchedQuestions.length < n) {
          const needed = n - fetchedQuestions.length;
          // Only auto-generate if we need a reasonable amount (e.g. don't auto-gen 100)
          if (needed > 0 && needed <= 20) {
            setGenerating(true);
            try {
              await API.post("generate-mcqs/", {
                note_id: noteIdParam,
                count: needed,
              });
              // 3. Re-fetch after generation
              res = await API.get(`quiz/${noteIdParam}/?n=${n}`);
              fetchedQuestions = res.data.questions || [];
            } catch (genErr) {
              console.error("Auto-generation failed:", genErr);
              // Continue with what we have, or show error?
              // We'll just show what we have for now.
            } finally {
              setGenerating(false);
            }
          }
        }

        setQuestions(fetchedQuestions);
      } catch (err) {
        console.error("Failed to load questions:", err);
        setError("Failed to load questions. Please try again.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [location.search]);

  // Confetti effect for high scores - MUST be before conditional returns
  useEffect(() => {
    if (finished && questions.length > 0 && score / questions.length >= 0.75) {
      const colors = ['#667eea', '#764ba2', '#f59e0b', '#10b981', '#ef4444'];
      const confettiCount = 50;
      
      for (let i = 0; i < confettiCount; i++) {
        setTimeout(() => {
          const confetti = document.createElement('div');
          confetti.className = 'confetti-piece';
          confetti.style.left = Math.random() * 100 + '%';
          confetti.style.background = colors[Math.floor(Math.random() * colors.length)];
          confetti.style.animationDelay = Math.random() * 0.5 + 's';
          document.body.appendChild(confetti);
          
          setTimeout(() => confetti.remove(), 3000);
        }, i * 30);
      }
    }
  }, [finished, score, questions.length]);

  // Timer logic - MUST be before conditional returns
  useEffect(() => {
    if (finished) return;
    setTimer(30); // Reset timer on question change
  }, [idx, finished]);

  useEffect(() => {
    if (finished || !questions.length) return;
    if (timer === 0) {
      // Auto-submit when timer reaches 0
      const autoSubmit = async () => {
        if (idx >= questions.length) return;
        const q = questions[idx];
        const timeTaken = 30;
        
        try {
          const res = await API.post("submit-mcq/", {
            question_id: q.id,
            selected_option: selected || "TIMEOUT",
            time_taken: timeTaken,
          });
          if (res.data.correct) setScore((s) => s + 1);
          
          if (idx + 1 < questions.length) {
            setIdx(idx + 1);
            setSelected("");
          } else {
            setFinished(true);
            try {
              await API.post("quiz-completed/", {
                note_id: noteId,
                score: res.data.correct ? score + 1 : score,
                total: questions.length
              });
              // Set flags for dashboard refresh
              localStorage.setItem('dashboardNeedsRefresh', 'true');
              localStorage.setItem('lastQuizNoteId', noteId);
              window.dispatchEvent(new CustomEvent('refreshNotifications'));
            } catch (error) {
              console.error("Failed to create notification:", error);
            }
          }
        } catch (error) {
          console.error("Auto-submit failed:", error);
        }
      };
      autoSubmit();
      return;
    }
    const interval = setInterval(() => {
      setTimer((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, [timer, finished, questions, idx, selected, noteId, score, API]);

  // Strict Navigation Blocking
  useEffect(() => {
    if (finished) return;

    // Push state to trap back button
    window.history.pushState(null, "", window.location.href);

    const handlePopState = (e) => {
      // Prevent navigation
      window.history.pushState(null, "", window.location.href);
      alert("⚠️ You cannot leave the quiz . Please use the 'Cancel Quiz' button if you wish to exit.");
    };

    const handleBeforeUnload = (e) => {
      e.preventDefault();
      e.returnValue = "Are you sure you want to leave? Your progress will be lost.";
      return e.returnValue;
    };

    window.addEventListener("popstate", handlePopState);
    window.addEventListener("beforeunload", handleBeforeUnload);

    return () => {
      window.removeEventListener("popstate", handlePopState);
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [finished]);

  const handleCancel = () => {
    if (window.confirm("Are you sure you want to cancel the quiz? Your progress will be lost.")) {
      navigate("/quiz-entry");
    }
  };

  const handleGenerateQuestions = async () => {
    setGenerating(true);
    try {
      // Generate questions
      await API.post("generate-mcqs/", {
        note_id: noteId,
        count: numQuestions,
      });
      
      // Reload the page to fetch the newly generated questions
      window.location.reload();
    } catch (err) {
      console.error("Failed to generate questions:", err);
      setError("Failed to generate questions. Please try again.");
      setGenerating(false);
    }
  };

  if (generating) return <Container className="p-6">Generating questions...</Container>;
  if (loading) return <Container className="p-6">Loading Quiz...</Container>;
  
  if (error) {
    return (
      <Container maxWidth="md" style={{ marginTop: 40 }}>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button
          variant="contained"
          onClick={() => navigate('/upload')}
          sx={{ mt: 2 }}
        >
          Go to Upload Lecture
        </Button>
      </Container>
    );
  }
  
  if (!questions.length) {
    // No questions available - show generation interface
    return (
      <Container maxWidth="sm" style={{ marginTop: 40 }}>
        <Card sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h5" sx={{ mb: 2, fontWeight: 600 }}>
            No Questions Available
          </Typography>
          <Typography variant="body1" sx={{ mb: 3, color: 'text.secondary' }}>
            This lecture note doesn't have any questions yet. Let's generate some!
          </Typography>
          
          <TextField
            label="Number of Questions"
            type="number"
            value={numQuestions}
            onChange={(e) => setNumQuestions(Math.max(1, parseInt(e.target.value) || 10))}
            fullWidth
            sx={{ mb: 3 }}
            inputProps={{ min: 1, max: 50 }}
          />
          
          <Button
            variant="contained"
            onClick={handleGenerateQuestions}
            disabled={generating}
            fullWidth
            sx={{
              height: "56px",
              background: "linear-gradient(135deg, #4f46e5, #6366f1)",
              borderRadius: "12px",
              fontWeight: 700,
              fontSize: "16px",
            }}
          >
            {generating ? "Generating Questions..." : `Generate ${numQuestions} Questions`}
          </Button>
        </Card>
      </Container>
    );
  }

  const q = questions[idx];
  const opts = [
    { key: "A", text: cleanOption(q.option_a) },
    { key: "B", text: cleanOption(q.option_b) },
    { key: "C", text: cleanOption(q.option_c) },
    { key: "D", text: cleanOption(q.option_d) },
  ];

  const submitAndNext = async (autoSubmit = false) => {
    if (!selected && !autoSubmit) return;

    const timeTaken = 30 - timer;

    const res = await API.post("submit-mcq/", {
      question_id: q.id,
      selected_option: selected || "TIMEOUT",
      time_taken: timeTaken,
    });
    const isCorrect = res.data.correct;
    if (isCorrect) setScore((s) => s + 1);

    // Track this answer for review
    const answerRecord = {
      questionId: q.id,
      question: q.question_text,
      options: {
        A: cleanOption(q.option_a),
        B: cleanOption(q.option_b),
        C: cleanOption(q.option_c),
        D: cleanOption(q.option_d),
      },
      userAnswer: selected || "TIMEOUT",
      correctAnswer: res.data.correct_option,
      isCorrect: isCorrect,
      explanation: q.explanation || "",
    };
    setUserAnswers(prev => [...prev, answerRecord]);

    // move to next or finish
    if (idx + 1 < questions.length) {
      setIdx(idx + 1);
      setSelected("");
    } else {
      setFinished(true);
      const finalScore = isCorrect ? score + 1 : score;
      const allAnswers = [...userAnswers, answerRecord];
      
      // Trigger notification creation
      try {
        await API.post("quiz-completed/", {
          note_id: noteId,
          score: finalScore,
          total: questions.length
        });
        
        // Set flags for dashboard refresh
        localStorage.setItem('dashboardNeedsRefresh', 'true');
        localStorage.setItem('lastQuizNoteId', noteId);
        
        // Trigger a custom event to refresh notifications
        window.dispatchEvent(new CustomEvent('refreshNotifications'));
        
        // Navigate to result page with answers
        navigate('/quiz-result', { 
          state: { 
            score: finalScore, 
            total: questions.length, 
            noteId: noteId,
            answers: allAnswers
          } 
        });
      } catch (error) {
        console.error("Failed to create notification:", error);
        // Navigate anyway
        navigate('/quiz-result', { 
          state: { 
            score: finalScore, 
            total: questions.length, 
            noteId: noteId,
            answers: allAnswers
          } 
        });
      }
    }
  };

  // Removed inline result rendering as we now navigate to QuizResult page

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        py: 4,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Container maxWidth="md">
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip 
              label={`Lecture ID: ${noteId}`} 
              sx={{ 
                fontWeight: 600,
                bgcolor: 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.3)'
              }}
            />
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            {/* Circular Timer */}
            <Box sx={{ position: 'relative', display: 'inline-flex' }}>
              <CircularProgress
                variant="determinate"
                value={(timer / 30) * 100}
                size={80}
                thickness={4}
                sx={{
                  color: timer > 20 ? '#10b981' : timer > 10 ? '#fbbf24' : '#ef4444',
                  filter: timer < 10 ? 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.6))' : 'none',
                  animation: timer < 10 ? 'pulse 1s infinite' : 'none',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                  },
                }}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: 'absolute',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexDirection: 'column',
                }}
              >
                <Typography 
                  variant="h5" 
                  component="div" 
                  sx={{ 
                    fontWeight: 700,
                    color: 'white',
                    textShadow: '0 2px 4px rgba(0,0,0,0.2)'
                  }}
                >
                  {timer}
                </Typography>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: 'rgba(255,255,255,0.9)',
                    fontWeight: 600,
                    fontSize: '0.65rem'
                  }}
                >
                  sec
                </Typography>
              </Box>
            </Box>
            
            <Button 
              variant="contained" 
              color="error" 
              onClick={handleCancel}
              startIcon={<CancelIcon />}
              sx={{ 
                borderRadius: 2,
                textTransform: 'none',
                fontWeight: 600,
                bgcolor: 'rgba(239, 68, 68, 0.2)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                boxShadow: 'none',
                '&:hover': {
                  bgcolor: 'rgba(239, 68, 68, 0.4)',
                  boxShadow: 'none',
                }
              }}
            >
              Cancel Quiz
            </Button>
          </Box>
        </Box>

        {/* Progress Bar */}
        <Box sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 600 }}>
              Question {idx + 1} of {questions.length}
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 600 }}>
              {Math.round(((idx) / questions.length) * 100)}% Completed
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={((idx) / questions.length) * 100}
            sx={{ 
              height: 10, 
              borderRadius: 5,
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              '& .MuiLinearProgress-bar': {
                background: 'white',
                borderRadius: 5,
              }
            }}
          />
        </Box>

        {/* Question Card */}
        <Card
          sx={{
            background: "rgba(255, 255, 255, 0.95)",
            backdropFilter: "blur(20px)",
            boxShadow: "0 20px 60px rgba(0,0,0,0.3)",
            borderRadius: 4,
            border: "1px solid rgba(255,255,255,0.8)",
            overflow: 'visible',
            position: 'relative'
          }}
        >
        <CardContent sx={{ p: { xs: 3, md: 5 } }}>
          <Typography 
            variant="h5" 
            sx={{ 
              mb: 4, 
              fontWeight: 700, 
              color: '#1e293b',
              lineHeight: 1.6 
            }}
          >
            {q.question_text}
          </Typography>

          <Stack spacing={2}>
            {opts.map((o) => {
              const isSelected = selected === o.key;
              return (
                <Paper
                  key={o.key}
                  elevation={0}
                  onClick={() => setSelected(o.key)}
                  sx={{
                    p: 2.5,
                    borderRadius: 3,
                    border: '2px solid',
                    borderColor: isSelected ? '#667eea' : '#e2e8f0',
                    background: isSelected ? '#f5f7ff' : 'white',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 2,
                    '&:hover': {
                      borderColor: isSelected ? '#667eea' : '#cbd5e0',
                      transform: 'translateY(-2px)',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
                    }
                  }}
                >
                  <Box
                    sx={{
                      width: 36,
                      height: 36,
                      borderRadius: '50%',
                      background: isSelected 
                        ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                        : '#f1f5f9',
                      color: isSelected ? 'white' : '#64748b',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontWeight: 700,
                      fontSize: '16px',
                      flexShrink: 0
                    }}
                  >
                    {o.key}
                  </Box>
                  <Typography 
                    variant="body1" 
                    sx={{ 
                      fontWeight: isSelected ? 600 : 500,
                      color: isSelected ? '#1e293b' : '#475569',
                      flex: 1
                    }}
                  >
                    {o.text}
                  </Typography>
                  {isSelected && (
                    <CheckCircleIcon sx={{ color: '#667eea' }} />
                  )}
                </Paper>
              );
            })}
          </Stack>

          <Box sx={{ mt: 5, display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              onClick={() => submitAndNext()}
              disabled={!selected}
              endIcon={loading ? <CircularProgress size={20} color="inherit" /> : null}
              sx={{
                height: "56px",
                minWidth: 160,
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                borderRadius: 3,
                px: 4,
                fontWeight: 700,
                fontSize: "16px",
                boxShadow: "0 6px 20px rgba(102, 126, 234, 0.4)",
                transition: "all 0.3s ease",
                '&:hover': {
                  boxShadow: "0 8px 25px rgba(102, 126, 234, 0.6)",
                  transform: "translateY(-2px)",
                },
                '&:disabled': {
                  background: '#cbd5e0',
                  color: '#94a3b8'
                }
              }}
            >
              {idx + 1 === questions.length ? "Finish Quiz" : "Next Question"}
            </Button>
          </Box>
        </CardContent>
      </Card>
      </Container>
    </Box>
  );
}

export default Quiz;
