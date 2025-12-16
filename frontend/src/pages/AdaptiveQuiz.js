// src/pages/AdaptiveQuiz.js
import React, { useEffect, useState } from "react";
import API from "../api/api";
import {
  Container,
  Card,
  CardContent,
  Typography,
  RadioGroup,
  FormControlLabel,
  Radio,
  Button,
  LinearProgress,
  Chip,
  Box,
  CircularProgress,
} from "@mui/material";
import { useParams, useNavigate } from "react-router-dom";

const cleanOption = (text) => {
  if (!text) return "";
  // Remove any leading A), B), C), D), A., B., (A), (B) etc, even if repeated
  return text.replace(/^([A-D][\.\)]\s*|\([A-D]\)\s*)+/gi, "").trim();
};

export default function AdaptiveQuizPage() {
  const { noteId } = useParams();
  const navigate = useNavigate();
  const [questions, setQuestions] = useState([]);
  const [idx, setIdx] = useState(0);
  const [selected, setSelected] = useState("");
  const [score, setScore] = useState(0);
  const [finished, setFinished] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [timer, setTimer] = useState(30);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      setError("");
      try {
        const res = await API.post("adaptive/quiz/start/", {
          note_id: noteId,
        });
        setQuestions(res.data.questions || []);
      } catch (err) {
        console.error("Failed to start adaptive quiz:", err);
        setError("Failed to start adaptive quiz.");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [noteId]);

  if (loading)
    return <Container className="p-6">Loading Adaptive Quiz...</Container>;
  if (!questions.length)
    return (
      <Container className="p-6">No adaptive questions available.</Container>
    );

  const q = questions[idx];
  const opts = [
    { key: "A", text: q.option_a },
    { key: "B", text: q.option_b },
    { key: "C", text: q.option_c },
    { key: "D", text: q.option_d },
  ];

  // Timer logic
  useEffect(() => {
    if (finished) return;
    setTimer(30); // Reset timer on question change
  }, [idx, finished]);

  useEffect(() => {
    if (finished) return;
    if (timer === 0) {
      submitAndNext(true); // Auto-submit on timeout
      return;
    }
    const interval = setInterval(() => {
      setTimer((prev) => (prev > 0 ? prev - 1 : 0));
    }, 1000);
    return () => clearInterval(interval);
  }, [timer, finished]);

  const submitAndNext = async (autoSubmit = false) => {
    if (!selected && !autoSubmit) return;

    const timeTaken = 30 - timer;

    const res = await API.post("submit-mcq/", {
      question_id: q.id,
      selected_option: selected || "TIMEOUT",
      time_taken: timeTaken,
    });
    if (res.data.correct) setScore((s) => s + 1);
    
    // Check if this is the last question
    if (idx + 1 < questions.length) {
      setIdx(idx + 1);
      setSelected("");
    } else {
      // Quiz finished - notify backend and set refresh flag
      const finalScore = res.data.correct ? score + 1 : score;
      try {
        await API.post("quiz-completed/", {
          note_id: noteId,
          score: finalScore,
          total: questions.length
        });
        // Set flag for dashboard to refresh
        localStorage.setItem('dashboardNeedsRefresh', 'true');
        localStorage.setItem('lastQuizNoteId', noteId);
      } catch (err) {
        console.error("Failed to notify quiz completion:", err);
      }
      setFinished(true);
    }
  };

  if (finished) {
    const percentage = Math.round((score / questions.length) * 100);
    const previousScore = localStorage.getItem(`adaptive_quiz_${noteId}_score`);
    const previousPercentage = previousScore ? parseInt(previousScore) : null;
    
    // Store current score
    localStorage.setItem(`adaptive_quiz_${noteId}_score`, percentage.toString());
    
    // Determine message based on performance
    let message = '';
    let emoji = '';
    
    if (percentage >= 90) {
      message = "Phenomenal! You've mastered this! 🏆";
      emoji = "🎉";
    } else if (percentage >= 75) {
      message = "Fantastic! You're doing great! 🌟";
      emoji = "✨";
    } else if (percentage >= 60) {
      message = "Well done! Keep up the momentum! 📚";
      emoji = "👏";
    } else if (percentage >= 40) {
      message = "Good effort! Practice makes perfect! 💡";
      emoji = "🌱";
    } else {
      message = "Stay strong! You're learning and growing! 🚀";
      emoji = "💪";
    }
    
    // Compare with previous score
    let comparisonMessage = '';
    let isImprovement = false;
    if (previousPercentage !== null) {
      if (percentage > previousPercentage) {
        comparisonMessage = `🎊 Amazing! +${percentage - previousPercentage}% improvement!`;
        isImprovement = true;
      } else if (percentage < previousPercentage) {
        comparisonMessage = `Previous best: ${previousPercentage}%. You can do it!`;
        isImprovement = false;
      } else {
        comparisonMessage = `Consistent! Try to beat ${previousPercentage}%!`;
      }
    }

    return (
      <Container maxWidth="sm" style={{ marginTop: 40 }}>
        <Card sx={{ 
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          borderRadius: '20px',
          boxShadow: '0 20px 60px rgba(102, 126, 234, 0.3)'
        }}>
          <CardContent sx={{ padding: 4, textAlign: 'center' }}>
            <div style={{ fontSize: '80px', marginBottom: '16px' }}>{emoji}</div>
            
            <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
              Adaptive Quiz Complete!
            </Typography>
            
            <Typography variant="h6" sx={{ mb: 3, opacity: 0.9 }}>
              {message}
            </Typography>
            
            <div style={{
              background: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '24px'
            }}>
              <Typography variant="h2" sx={{ fontWeight: 800, mb: 1 }}>
                {score} / {questions.length}
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 600 }}>
                {percentage}% Score
              </Typography>
              
              {comparisonMessage && (
                <Typography 
                  variant="body1" 
                  sx={{ 
                    mt: 2, 
                    padding: '12px 20px',
                    background: isImprovement 
                      ? 'rgba(16, 185, 129, 0.3)' 
                      : 'rgba(245, 158, 11, 0.3)',
                    borderRadius: '12px',
                    fontWeight: 600
                  }}
                >
                  {comparisonMessage}
                </Typography>
              )}
            </div>

            <Button
              variant="contained"
              onClick={() => {
                setIdx(0);
                setSelected("");
                setScore(0);
                setFinished(false);
              }}
              sx={{
                background: "white",
                color: "#667eea",
                borderRadius: "10px",
                py: 1.5,
                px: 5,
                fontWeight: "bold",
                fontSize: "15px",
                boxShadow: "0 6px 20px rgba(255,255,255,0.3)",
                '&:hover': {
                  background: 'rgba(255, 255, 255, 0.9)',
                  transform: 'translateY(-2px)',
                }
              }}
            >
              Retry Adaptive Quiz 🔄
            </Button>
            
            <Button
              variant="contained"
              onClick={() => navigate(`/study-plan?noteId=${noteId}`, { state: { noteId } })}
              sx={{
                ml: 2,
                background: "linear-gradient(135deg, #10b981, #059669)",
                color: "white",
                borderRadius: "10px",
                py: 1.5,
                px: 5,
                fontWeight: "bold",
                fontSize: "15px",
                boxShadow: "0 6px 20px rgba(16,185,129,0.3)",
                '&:hover': {
                  background: 'linear-gradient(135deg, #059669, #047857)',
                  transform: 'translateY(-2px)',
                }
              }}
            >
              Study Plan 📊
            </Button>
          </CardContent>
        </Card>
      </Container>
    );
  }

  // difficulty display
  const diff = q.difficulty || 0.5;
  const diffLabel = diff < 0.4 ? "Easy" : diff < 0.7 ? "Medium" : "Hard";

  return (
    <Container maxWidth="md" style={{ marginTop: 40 }}>
      <Typography variant="subtitle1" sx={{ mb: 2, color: 'text.secondary', fontWeight: 'bold' }}>
        Lecture ID: {noteId}
      </Typography>

      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Typography variant="h6" color={timer < 10 ? "error" : "primary"} sx={{ fontWeight: 'bold' }}>
          Time Left: {timer}s
        </Typography>
        <CircularProgress 
          variant="determinate" 
          value={(timer / 30) * 100} 
          color={timer < 10 ? "error" : "primary"}
          size={40}
        />
      </Box>

      <LinearProgress
        variant="determinate"
        value={(idx / questions.length) * 100}
        style={{ marginBottom: 16 }}
      />
      <Card>
        <CardContent>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <Typography variant="subtitle2">
              Question {idx + 1} / {questions.length}
            </Typography>
            <Chip label={diffLabel} />
          </div>

          <Typography variant="h6" style={{ marginTop: 12 }}>
            {q.question_text}
          </Typography>

          <RadioGroup
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
          >
            {opts.map((o) => (
              <FormControlLabel
                key={o.key}
                value={o.key}
                control={<Radio />}
                label={`${o.key}) ${o.text}`}
              />
            ))}
          </RadioGroup>

          <div style={{ marginTop: 16 }}>
            <Button
              variant="contained"
              onClick={submitAndNext}
              disabled={!selected}
              sx={{
                background: "linear-gradient(135deg, #4f46e5, #6366f1)",
                borderRadius: "10px",
                py: 1.2,
                px: 4,
                fontWeight: "bold",
                fontSize: "15px",
              }}
            >
              {idx + 1 === questions.length ? "Finish" : "Next"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </Container>
  );
}
