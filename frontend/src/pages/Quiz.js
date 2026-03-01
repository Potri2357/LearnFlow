// src/pages/Quiz.js
import React, { useEffect, useState, useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  Box,
  Typography,
  Button,
  LinearProgress,
  Container,
  Paper,
  IconButton,
  CircularProgress,
  Dialog,
  Grid,
  Chip,
  useTheme,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  useMediaQuery,
  Alert
} from "@mui/material";
import {
  ArrowBack as ArrowBackIcon,
  Timer as TimerIcon,
  Settings as SettingsIcon,
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
  PlayCircle as PlayCircleIcon,
  Flag as FlagIcon,
  ArrowForward as ArrowForwardIcon,
  AutoAwesome as AutoAwesomeIcon,
  Close as CloseIcon,
  Check as CheckIcon,
  Cancel as CancelIcon
} from "@mui/icons-material";

const cleanOption = (text) => {
  if (!text) return "";
  return text.replace(/^([A-D][\.\)]\s*|\([A-D]\)\s*)+/gi, "").trim();
};

const Quiz = () => {
    const { api: API } = useAuth();
    const theme = useTheme();
    const isLgUp = useMediaQuery(theme.breakpoints.up("lg"));
    const location = useLocation();
    const navigate = useNavigate();

    // State
    const [questions, setQuestions] = useState([]);
    const [idx, setIdx] = useState(0);
    const [selected, setSelected] = useState("");
    const [score, setScore] = useState(0);
    const [finished, setFinished] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [noteId, setNoteId] = useState(null);
    const [timer, setTimer] = useState(30);
    const [numQuestions, setNumQuestions] = useState(10);
    const [generating, setGenerating] = useState(false);
    const [userAnswers, setUserAnswers] = useState([]);
    const [quizStartTime] = useState(() => Date.now());

    // Derived State
    const currentQuestion = questions[idx];
    const progress = questions.length > 0 ? ((idx + 1) / questions.length) * 100 : 0;
    
    // Initial Load
    useEffect(() => {
        const load = async () => {
          setLoading(true);
          setError("");
          const searchParams = new URLSearchParams(location.search);
          
          // Support both single noteId and multiple noteIds
          const noteIdParam = searchParams.get("noteId");
          const noteIdsParam = searchParams.get("noteIds");
          const n = parseInt(searchParams.get("n") || "10", 10);
          
          // Check navigation state for weak topics practice
          const { weakTopics, questionCount, isPracticeAll } = location.state || {};
          
          let noteIdsToFetch = [];
          
          if (noteIdsParam) {
              // Multiple noteIds from URL
              noteIdsToFetch = noteIdsParam.split(',').filter(id => id);
          } else if (noteIdParam) {
              // Single noteId (backward compatibility) or comma-separated
              if (noteIdParam.includes(',')) {
                  noteIdsToFetch = noteIdParam.split(',').filter(id => id);
              } else {
                  noteIdsToFetch = [noteIdParam];
              }
          } else if (isPracticeAll && weakTopics && weakTopics.length > 0) {
              // For Practice All: fetch lectures that contain these topics (Fallback if no IDs in URL)
              try {
                  const topicsQuery = weakTopics.join(',');
                  const res = await API.get(`lectures/by-topics/?topics=${topicsQuery}`);
                  noteIdsToFetch = res.data.note_ids || [];
              } catch (err) {
                  console.error("Failed to fetch lectures by topics:", err);
                  setError("Failed to find lectures for selected topics.");
                  setLoading(false);
                  return;
              }
          }
          
          if (noteIdsToFetch.length === 0) {
            setError("No lecture notes provided. Please select at least one lecture.");
            setLoading(false);
            return;
          }
          
          // Store for quiz completion
          setNoteId(noteIdsToFetch.length === 1 ? noteIdsToFetch[0] : noteIdsToFetch.join(','));
          
          try {
            const questionsPerLecture = Math.ceil((questionCount || n) / noteIdsToFetch.length);
            let allQuestions = [];
            
            // Fetch questions from each lecture
            for (const id of noteIdsToFetch) {
                try {
                    let res = await API.get(`quiz/${id}/?n=${questionsPerLecture}`);
                    let fetchedQuestions = res.data.questions || [];
                    
                    // If not enough questions, try to generate more
                    if (fetchedQuestions.length < questionsPerLecture) {
                      const needed = questionsPerLecture - fetchedQuestions.length;
                      if (needed > 0 && needed <= 20) {
                        setGenerating(true);
                        try {
                          await API.post("generate-mcqs/", {
                            note_id: id,
                            count: needed,
                          });
                          res = await API.get(`quiz/${id}/?n=${questionsPerLecture}`);
                          fetchedQuestions = res.data.questions || [];
                        } catch (genErr) {
                          console.error(`Auto-generation failed for lecture ${id}:`, genErr);
                        } finally {
                          setGenerating(false);
                        }
                      }
                    }
                    
                    allQuestions.push(...fetchedQuestions);
                } catch (err) {
                    console.error(`Failed to load questions from lecture ${id}:`, err);
                }
            }
            
            // Shuffle questions for variety
            allQuestions = allQuestions.sort(() => Math.random() - 0.5);
            
            // Limit to requested count
            const finalQuestions = allQuestions.slice(0, questionCount || n);
            
            if (finalQuestions.length === 0) {
                setError("No questions available. Please try generating questions first.");
            }
            
            setQuestions(finalQuestions);
          } catch (err) {
            console.error("Failed to load questions:", err);
            setError("Failed to load questions. Please try again.");
          } finally {
            setLoading(false);
          }
        };
        load();
    }, [location.search, location.state, API]);

    // Timer Logic
    useEffect(() => {
        if (finished) return;
        setTimer(30); 
    }, [idx, finished]);
    
    useEffect(() => {
        if (finished || !questions.length) return;
        if (timer === 0) {
           submitAndNext(true); // Auto-submit
           return;
        }
        const interval = setInterval(() => {
          setTimer((prev) => (prev > 0 ? prev - 1 : 0));
        }, 1000);
        return () => clearInterval(interval);
    }, [timer, finished, questions.length]); // Added dependency on timer state for correct countdown

    // Navigation Block
    useEffect(() => {
        if (finished) return;
    
        window.history.pushState(null, "", window.location.href);
    
        const handlePopState = (e) => {
          window.history.pushState(null, "", window.location.href);
          alert("⚠️ You cannot leave the quiz. Please use the 'Exit Quiz' button if you wish to exit.");
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

    const handleExit = () => {
        if (window.confirm("Are you sure you want to exit? Your progress will be lost.")) {
            navigate("/quiz-entry"); // Or dashboard?
        }
    };

    const submitAndNext = async (autoSubmit = false) => {
        if (!selected && !autoSubmit && !finished) return; // Allow skip via autoSubmit? actually autoSubmit means timeout
        
        // If simply skipping (manual), selected is empty. 
        // Logic: if manual click "Skip", call this with selected=""?
        // Or handle skip separately. "Skip" usually just moves next without scoring, maybe marks as skipped.
        // Current backend expects an option or timeout. Let's assume Skip = Timeout or Wrong for now, 
        // provided implementation sends "TIMEOUT" if not selected.
        
        const q = questions[idx];
        const timeTaken = 30 - timer;
        const answerToSend = selected || "TIMEOUT";
    
        try {
            const res = await API.post("submit-mcq/", {
                question_id: q.id,
                selected_option: answerToSend,
                time_taken: timeTaken,
            });
            const isCorrect = res.data.correct;
            
            // Optimistic update for score? Wait for confirmation.
            // Actually API returns correct status.
            let newScore = score;
            if (isCorrect) {
                setScore((s) => s + 1);
                newScore = score + 1;
            }
        
            const answerRecord = {
                questionId: q.id,
                question: q.question_text,
                options: {
                    A: cleanOption(q.option_a),
                    B: cleanOption(q.option_b),
                    C: cleanOption(q.option_c),
                    D: cleanOption(q.option_d),
                },
                userAnswer: answerToSend,
                correctAnswer: res.data.correct_option,
                isCorrect: isCorrect,
                explanation: q.explanation || "",
            };
            
            const nextAnswers = [...userAnswers, answerRecord];
            setUserAnswers(nextAnswers);
        
            if (idx + 1 < questions.length) {
                setIdx(idx + 1);
                setSelected("");
            } else {
                setFinished(true);
                // Finish Quiz
                const totalTimeTaken = Math.round((Date.now() - quizStartTime) / 1000); // in seconds
                try {
                     await API.post("quiz-completed/", {
                        note_id: noteId,
                        score: newScore,
                        total: questions.length
                    });
                    localStorage.setItem('dashboardNeedsRefresh', 'true');
                    localStorage.setItem('lastQuizNoteId', noteId);
                    window.dispatchEvent(new CustomEvent('refreshNotifications'));
                    navigate('/quiz-result', { state: { score: newScore, total: questions.length, noteId: noteId, answers: nextAnswers, totalTimeTaken } });
                } catch (err) {
                    console.error("Failed to complete quiz:", err);
                    // Navigate anyway
                    navigate('/quiz-result', { state: { score: newScore, total: questions.length, noteId: noteId, answers: nextAnswers, totalTimeTaken } });
                }
            }
        } catch (error) {
            console.error("Submit failed", error);
        }
    };

    const handleSkip = () => {
        // Treat as unanswered/timeout for now, or just move next without checking API?
        // Usually better to record it as skipped/wrong.
        submitAndNext(true); 
    };

    // --- RENDER HELPERS ---
    
    const options = useMemo(() => {
        if (!currentQuestion) return [];
        return [
            { key: "A", text: cleanOption(currentQuestion.option_a) },
            { key: "B", text: cleanOption(currentQuestion.option_b) },
            { key: "C", text: cleanOption(currentQuestion.option_c) },
            { key: "D", text: cleanOption(currentQuestion.option_d) },
        ];
    }, [currentQuestion]);

    if (generating) return (
        <Box sx={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 2 }}>
            <CircularProgress />
            <Typography>Generating adaptive questions...</Typography>
        </Box>
    );

    if (loading) return (
        <Box sx={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <CircularProgress />
        </Box>
    );

    if (error) return (
         <Container maxWidth="md" sx={{ mt: 8 }}>
            <Alert severity="error" action={
                <Button color="inherit" size="small" onClick={() => navigate('/lectures')}>
                    Go Back
                </Button>
            }>
                {error}
            </Alert>
        </Container>
    );

    if (!questions.length) return (
        <Container maxWidth="sm" sx={{ mt: 8, textAlign: 'center' }}>
             <Typography variant="h5" gutterBottom>No questions generated yet.</Typography>
             <Button variant="contained" onClick={() => navigate('/lectures')}>Back to Library</Button>
        </Container>
    );

    return (
        <Box sx={{ 
            height: '100vh', 
            display: 'flex', 
            flexDirection: 'column',
            bgcolor: 'background.default',
            overflow: 'hidden'
        }}>
            {/* --- HEADER --- */}
            <Box sx={{ 
                height: 72, 
                px: { xs: 2, md: 4 }, 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                borderBottom: '1px solid',
                borderColor: 'divider',
                bgcolor: 'background.paper',
                zIndex: 10
            }}>
                {/* Left: Exit */}
                <Box sx={{ width: { xs: 'auto', md: '25%' }, display: 'flex', alignItems: 'center' }}>
                     <Button 
                        startIcon={<ArrowBackIcon />} 
                        onClick={handleExit}
                        sx={{ 
                            color: 'text.secondary', 
                            '&:hover': { color: 'text.primary', bgcolor: 'transparent' },
                            textTransform: 'none',
                            fontWeight: 700
                        }}
                    >
                        Exit Quiz
                    </Button>
                </Box>

                {/* Center: Progress & Title */}
                <Box sx={{ 
                    flex: 1, 
                    maxWidth: 600, 
                    display: 'flex', 
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 1
                }}>
                    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', px: 1 }}>
                        <Typography variant="body2" fontWeight={700} sx={{ display: { xs: 'none', sm: 'block' } }}>
                            Quiz Session
                        </Typography>
                        <Typography variant="caption" fontWeight={600} color="text.secondary">
                            Question {idx + 1} / {questions.length}
                        </Typography>
                    </Box>
                    <LinearProgress 
                        variant="determinate" 
                        value={progress} 
                        sx={{ 
                            width: '100%', 
                            height: 6, 
                            borderRadius: 4, 
                            bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)',
                            '& .MuiLinearProgress-bar': { borderRadius: 4 }
                        }}
                    />
                </Box>

                {/* Right: Timer */}
                <Box sx={{ width: { xs: 'auto', md: '25%' }, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                    <Paper sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: 1.5, 
                        px: 2, 
                        py: 1, 
                        borderRadius: '12px',
                        border: '1px solid',
                        borderColor: 'primary.main',
                        bgcolor: theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.1)' : 'rgba(19, 127, 236, 0.05)',
                        boxShadow: '0 4px 20px -4px rgba(19,127,236,0.3)'
                    }}>
                        <Box sx={{ 
                            width: 32, 
                            height: 32, 
                            borderRadius: '50%', 
                            bgcolor: 'rgba(19, 127, 236, 0.1)', 
                            color: 'primary.main',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <TimerIcon sx={{ fontSize: 18 }} />
                        </Box>
                        <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                             <Typography variant="caption" sx={{ fontSize: '0.65rem', fontWeight: 700, color: 'text.secondary', textTransform: 'uppercase', lineHeight: 1 }}>
                                Time Left
                             </Typography>
                             <Typography variant="body1" sx={{ fontWeight: 700, lineHeight: 1, fontFamily: 'monospace' }}>
                                00:{timer < 10 ? `0${timer}` : timer}
                             </Typography>
                        </Box>
                    </Paper>
                </Box>
            </Box>

            {/* --- MAIN CONTENT AREA --- */}
            <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                
                {/* --- SIDEBAR (Desktop) --- */}
                {isLgUp && (
                     <Drawer
                        variant="permanent"
                        sx={{
                            width: 300,
                            flexShrink: 0,
                            '& .MuiDrawer-paper': {
                                width: 300,
                                position: 'relative',
                                borderRight: '1px solid',
                                borderColor: 'divider',
                                boxSizing: 'border-box',
                                bgcolor: 'background.paper' // Match header
                            },
                        }}
                    >
                        <Box sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 4, height: '100%', overflowY: 'auto' }}>
                            <Box>
                                <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    Question Map
                                </Typography>
                                <List dense sx={{ mt: 1 }}>
                                    {questions.map((q, i) => {
                                        let status = 'future';
                                        if (i < idx) status = 'completed';
                                        if (i === idx) status = 'current';

                                        return (
                                            <ListItem 
                                                key={q.id} 
                                                disablePadding 
                                                sx={{ 
                                                    mb: 1,
                                                    borderRadius: '8px',
                                                    border: status === 'current' ? '1px solid' : 'none',
                                                    borderColor: 'primary.main',
                                                    bgcolor: status === 'current' ? (theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.1)' : '#eff6ff') : 'transparent',
                                                    position: 'relative',
                                                    overflow: 'hidden'
                                                }}
                                            >
                                                {status === 'current' && (
                                                    <Box sx={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, bgcolor: 'primary.main' }} />
                                                )}
                                                <ListItemButton disabled={status === 'future'} dense sx={{ borderRadius: '8px', py: 1.5 }}>
                                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                                        {status === 'completed' && <CheckCircleIcon color="disabled" fontSize="small" />}
                                                        {status === 'current' && <PlayCircleIcon color="primary" fontSize="small" />}
                                                        {status === 'future' && <RadioButtonUncheckedIcon color="disabled" fontSize="small" />}
                                                    </ListItemIcon>
                                                    <ListItemText 
                                                        primary={`Question ${i + 1}`} 
                                                        primaryTypographyProps={{ 
                                                            variant: 'body2', 
                                                            fontWeight: status === 'current' ? 700 : 500,
                                                            color: status === 'current' ? 'text.primary' : 'text.disabled'
                                                        }}
                                                    />
                                                </ListItemButton>
                                            </ListItem>
                                        );
                                    })}
                                </List>
                            </Box>

                            <Box sx={{ mt: 'auto' }}>
                                <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                                    AI Context
                                </Typography>
                                <Paper sx={{ 
                                    mt: 1.5, 
                                    p: 2, 
                                    borderRadius: '12px', 
                                    background: theme.palette.mode === 'dark' 
                                        ? 'linear-gradient(135deg, rgba(19, 127, 236, 0.1) 0%, rgba(17, 26, 34, 0.5) 100%)' 
                                        : 'linear-gradient(135deg, rgba(19, 127, 236, 0.05) 0%, #ffffff 100%)',
                                    border: '1px solid',
                                    borderColor: 'rgba(19, 127, 236, 0.2)'
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, color: 'primary.main' }}>
                                        <AutoAwesomeIcon fontSize="small" />
                                        <Typography variant="caption" fontWeight={700}>Adaptive Difficulty</Typography>
                                    </Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                        Questions are adapted based on your previous answers to optimize your learning curve.
                                    </Typography>
                                </Paper>
                            </Box>
                        </Box>
                    </Drawer>
                )}

                {/* --- CENTER CONTENT --- */}
                <Box sx={{ 
                    flex: 1, 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center', 
                    overflowY: 'auto',
                    position: 'relative',
                    p: { xs: 2, md: 6 }
                }}>
                    
                     {/* Background Grid Pattern */}
                    <Box sx={{ 
                        position: 'absolute', 
                        inset: 0, 
                        zIndex: 0,
                        opacity: 0.05,
                        backgroundImage: `radial-gradient(${theme.palette.text.primary} 1px, transparent 1px)`,
                        backgroundSize: '32px 32px',
                        pointerEvents: 'none'
                    }} />

                    <Container maxWidth="md" sx={{ zIndex: 1, width: '100%', display: 'flex', flexDirection: 'column', gap: 4, mb: 10 }}>
                        
                        {/* Question Text */}
                        <Box sx={{ animation: 'fadeInUp 0.5s ease-out' }}>
                            <Typography variant="caption" sx={{ color: 'primary.main', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', mb: 1, display: 'block' }}>
                                Multiple Choice
                            </Typography>
                            <Typography variant="h4" component="h1" fontWeight={700} sx={{ lineHeight: 1.3 }}>
                                {currentQuestion.question_text}
                            </Typography>
                        </Box>

                        {/* Options */}
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            {options.map((opt) => {
                                const isSelected = selected === opt.key;
                                return (
                                    <Paper
                                        key={opt.key}
                                        onClick={() => setSelected(opt.key)}
                                        sx={{
                                            p: 2,
                                            borderRadius: '16px',
                                            cursor: 'pointer',
                                            border: '1px solid',
                                            borderColor: isSelected ? 'primary.main' : 'divider',
                                            bgcolor: isSelected 
                                                ? (theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.1)' : '#eff6ff') 
                                                : 'background.paper',
                                            boxShadow: isSelected ? '0 0 0 1px #137fec' : 'none',
                                            transition: 'all 0.2s ease',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: 2,
                                            '&:hover': {
                                                borderColor: isSelected ? 'primary.main' : 'text.secondary',
                                                transform: 'translateY(-2px)'
                                            }
                                        }}
                                    >
                                        <Box sx={{ 
                                            width: 40, 
                                            height: 40, 
                                            borderRadius: '10px', 
                                            bgcolor: isSelected ? 'primary.main' : (theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)'),
                                            color: isSelected ? 'white' : 'text.secondary',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            fontWeight: 700,
                                            border: '1px solid',
                                            borderColor: isSelected ? 'primary.main' : 'divider',
                                            flexShrink: 0
                                        }}>
                                            {opt.key}
                                        </Box>
                                        <Typography variant="body1" fontWeight={500} sx={{ flex: 1 }}>
                                            {opt.text}
                                        </Typography>
                                    </Paper>
                                );
                            })}
                        </Box>

                        {/* Footer / Actions */}
                        <Box sx={{ 
                            mt: 2, 
                            pt: 3, 
                            borderTop: '1px solid', 
                            borderColor: 'divider',
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center'
                        }}>
                             <Button 
                                startIcon={<FlagIcon />} 
                                sx={{ color: 'text.secondary', textTransform: 'none', fontWeight: 600 }}
                             >
                                Report Issue
                             </Button>

                             <Box sx={{ display: 'flex', gap: 2 }}>
                                 <Button 
                                    variant="outlined" 
                                    onClick={handleSkip}
                                    sx={{ 
                                        fontWeight: 700, 
                                        color: 'text.primary', 
                                        borderColor: 'divider',
                                        px: 3 
                                    }}
                                 >
                                    Skip
                                 </Button>
                                 <Button 
                                    variant="contained" 
                                    onClick={() => submitAndNext()}
                                    disabled={!selected}
                                    endIcon={<ArrowForwardIcon />}
                                    sx={{ 
                                        fontWeight: 700, 
                                        px: 4, 
                                        boxShadow: '0 4px 14px 0 rgba(19, 127, 236, 0.4)'
                                    }}
                                 >
                                    {idx + 1 === questions.length ? 'Finish Quiz' : 'Submit Answer'}
                                 </Button>
                             </Box>
                        </Box>

                    </Container>
                </Box>
            </Box>
            
            {/* Global Keyframes for animations */}
            <style>
                {`
                    @keyframes fadeInUp {
                        from { opacity: 0; transform: translateY(20px); }
                        to { opacity: 1; transform: translateY(0); }
                    }
                `}
            </style>
        </Box>
    );
};

export default Quiz;
