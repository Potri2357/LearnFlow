// src/pages/QuizResult.js
import React, { useEffect, useState, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Paper,
  Divider,
  Chip,
  useTheme,
  IconButton,
  LinearProgress,
  Collapse,
  Avatar
} from '@mui/material';
import {
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  EmojiEvents as TrophyIcon,
  Timer as TimerIcon,
  TrendingUp as TrendingUpIcon,
  LocalFireDepartment as FireIcon,
  ArrowForward as ArrowForwardIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Schedule as ScheduleIcon,
  ThumbUp as ThumbUpIcon,
  Lightbulb as LightbulbIcon,
  AutoAwesome as AutoAwesomeIcon,
  Check as CheckIcon,
  Close as CloseIcon,
  Speed as SpeedIcon
} from '@mui/icons-material';
import confetti from 'canvas-confetti';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import API from '../api/api';

function QuizResult() {
  const theme = useTheme();
  const location = useLocation();
  const navigate = useNavigate();
  const { score, total, noteId, answers = [], totalTimeTaken } = location.state || { score: 0, total: 0, answers: [], totalTimeTaken: 0 };
  
  const [showQuestions, setShowQuestions] = useState(false);
  const [streak, setStreak] = useState(0);
  
  const percentage = total ? Math.round((score / total) * 100) : 0;
  
  // Format time taken
  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null) return '--';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins === 0) return `${secs}s`;
    return `${mins}m ${secs}s`;
  };
  
  const avgTimePerQuestion = total > 0 && totalTimeTaken ? Math.round(totalTimeTaken / total) : null;
  
  // Build real performance chart from answers
  const chartData = useMemo(() => {
    if (!answers || answers.length === 0) return [];
    
    // Show cumulative score progression
    let cumCorrect = 0;
    return answers.map((ans, idx) => {
      if (ans.isCorrect) cumCorrect++;
      const score = Math.round((cumCorrect / (idx + 1)) * 100);
      return {
        name: `Q${idx + 1}`,
        score: score,
      };
    });
  }, [answers]);

  // Real topic breakdown from answers
  const topicBreakdown = useMemo(() => {
    if (!answers || answers.length === 0) return [];
    
    const topicMap = {};
    answers.forEach(ans => {
      // Use first word of question as pseudo-topic, or just group by correctness
      const key = ans.isCorrect ? 'Correct' : 'Incorrect';
      if (!topicMap[key]) topicMap[key] = { correct: 0, total: 0 };
      topicMap[key].total++;
      if (ans.isCorrect) topicMap[key].correct++;
    });
    
    // Build a meaningful breakdown by segmenting questions into groups
    const groupSize = Math.ceil(answers.length / 3);
    const groups = [
      { label: 'First Phase', items: answers.slice(0, groupSize) },
      { label: 'Middle Phase', items: answers.slice(groupSize, groupSize * 2) },
      { label: 'Final Phase', items: answers.slice(groupSize * 2) }
    ].filter(g => g.items.length > 0);
    
    return groups.map((g, i) => {
      const correct = g.items.filter(a => a.isCorrect).length;
      const pct = Math.round((correct / g.items.length) * 100);
      const colors = ['success', 'primary', 'warning'];
      return { label: g.label, percentage: pct, color: colors[i % 3] };
    });
  }, [answers]);
  
  // Celebration effect
  useEffect(() => {
    if (percentage >= 70) {
        const duration = 3000;
        const end = Date.now() + duration;

        (function frame() {
            confetti({
                particleCount: 3,
                angle: 60,
                spread: 55,
                origin: { x: 0 },
                colors: ['#137fec', '#2563eb', '#60a5fa']
            });
            confetti({
                particleCount: 3,
                angle: 120,
                spread: 55,
                origin: { x: 1 },
                colors: ['#137fec', '#2563eb', '#60a5fa']
            });
            
            if (Date.now() < end) {
                requestAnimationFrame(frame);
            }
        }());
    }
    
    // Fetch streak from API
    API.get('dashboard/stats/').then(res => {
        setStreak(res.data.streak || 0);
    }).catch(() => {});
  }, [percentage]);

  const getGrade = () => {
    if (percentage >= 90) return { grade: 'A+', color: 'success.main', msg: 'Excellent!' };
    if (percentage >= 80) return { grade: 'A', color: 'success.main', msg: 'Great job!' };
    if (percentage >= 70) return { grade: 'B', color: 'primary.main', msg: 'Good work!' };
    if (percentage >= 60) return { grade: 'C', color: 'warning.main', msg: 'Keep practicing' };
    return { grade: 'D', color: 'error.main', msg: 'Needs improvement' };
  };
  
  const { grade, color: gradeColor, msg: gradeMsg } = getGrade();

  return (
    <Box sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default',
        color: 'text.primary',
        pb: 8
    }}>
        {/* Header */}
        <Box sx={{ 
            bgcolor: 'background.paper', 
            borderBottom: '1px solid', 
            borderColor: 'divider',
            pt: 4,
            pb: 4,
            px: { xs: 2, md: 6 }
        }}>
            <Container maxWidth="xl">
                <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { md: 'flex-end' }, gap: 3 }}>
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                            <Chip 
                                label="Completed" 
                                size="small" 
                                sx={{ bgcolor: 'rgba(19, 127, 236, 0.1)', color: 'primary.main', fontWeight: 700, textTransform: 'uppercase', fontSize: '0.7rem' }} 
                            />
                            <Typography variant="caption" color="text.secondary">•</Typography>
                            <Typography variant="caption" color="text.secondary" fontWeight={500}>
                                {total} Questions
                            </Typography>
                        </Box>
                        <Typography variant="h3" fontWeight={900} sx={{ mb: 1, letterSpacing: '-0.02em', lineHeight: 1.2 }}>
                            Quiz Results
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, color: 'text.secondary' }}>
                            <ScheduleIcon sx={{ fontSize: 18 }} />
                            <Typography variant="body2" fontWeight={500}>
                                Completed on {new Date().toLocaleDateString()} • {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </Typography>
                        </Box>
                    </Box>
                    
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button 
                            variant="outlined" 
                            onClick={() => setShowQuestions(!showQuestions)}
                            endIcon={showQuestions ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                            sx={{ fontWeight: 700, borderColor: 'divider', color: 'text.primary', '&:hover': { borderColor: 'text.primary', bgcolor: 'transparent' } }}
                        >
                            {showQuestions ? 'Hide' : 'Review'} Questions
                        </Button>
                        <Button 
                            variant="contained" 
                            onClick={() => navigate('/dashboard')}
                            sx={{ fontWeight: 700, px: 4, boxShadow: '0 8px 16px -4px rgba(19, 127, 236, 0.3)' }}
                        >
                            Back to Dashboard
                        </Button>
                    </Box>
                </Box>
            </Container>
        </Box>

        <Container maxWidth="xl" sx={{ mt: 4 }}>
            {/* KPI Grid */}
            <Grid container spacing={3} sx={{ mb: 6 }}>
                {/* Score */}
                <Grid item xs={12} sm={6} lg={3}>
                    <Paper sx={{ p: 3, borderRadius: '16px', position: 'relative', overflow: 'hidden', height: '100%', border: '1px solid', borderColor: 'divider' }}>
                         <Box sx={{ position: 'absolute', top: 0, right: 0, p: 3, opacity: 0.05 }}>
                            <TrophyIcon sx={{ fontSize: 80, color: '#137fec' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary', mb: 2 }}>
                            <TrendingUpIcon sx={{ fontSize: 20 }} />
                            <Typography variant="caption" fontWeight={700} sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>Overall Score</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 1 }}>
                            <Typography variant="h3" fontWeight={800}>{percentage}%</Typography>
                            <Typography variant="body2" sx={{ color: gradeColor, fontWeight: 700 }}>
                                {grade} · {gradeMsg}
                            </Typography>
                        </Box>
                        <LinearProgress 
                            variant="determinate" 
                            value={percentage} 
                            sx={{ height: 6, borderRadius: 3, bgcolor: 'action.hover', '& .MuiLinearProgress-bar': { borderRadius: 3 } }} 
                        />
                    </Paper>
                </Grid>

                {/* Accuracy */}
                <Grid item xs={12} sm={6} lg={3}>
                    <Paper sx={{ p: 3, borderRadius: '16px', position: 'relative', overflow: 'hidden', height: '100%', border: '1px solid', borderColor: 'divider' }}>
                         <Box sx={{ position: 'absolute', top: 0, right: 0, p: 3, opacity: 0.05 }}>
                            <CheckCircleIcon sx={{ fontSize: 80, color: '#10B981' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary', mb: 2 }}>
                            <CheckCircleIcon sx={{ fontSize: 20 }} />
                            <Typography variant="caption" fontWeight={700} sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>Accuracy Rate</Typography>
                        </Box>
                        <Typography variant="h3" fontWeight={800}>{percentage}%</Typography>
                        <Typography variant="body2" color="text.secondary" fontWeight={500}>
                            {score} Correct / {total - score} Incorrect
                        </Typography>
                    </Paper>
                </Grid>

                {/* Time Taken */}
                <Grid item xs={12} sm={6} lg={3}>
                     <Paper sx={{ p: 3, borderRadius: '16px', position: 'relative', overflow: 'hidden', height: '100%', border: '1px solid', borderColor: 'divider' }}>
                         <Box sx={{ position: 'absolute', top: 0, right: 0, p: 3, opacity: 0.05 }}>
                            <TimerIcon sx={{ fontSize: 80, color: '#F59E0B' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary', mb: 2 }}>
                            <TimerIcon sx={{ fontSize: 20 }} />
                            <Typography variant="caption" fontWeight={700} sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>Time Taken</Typography>
                        </Box>
                        <Typography variant="h3" fontWeight={800}>
                            {totalTimeTaken !== undefined && totalTimeTaken !== null ? formatTime(totalTimeTaken) : '--'}
                        </Typography>
                         <Typography variant="body2" color="text.secondary" fontWeight={500}>
                            {avgTimePerQuestion ? `~${avgTimePerQuestion}s avg per question` : 'Avg time per question'}
                        </Typography>
                    </Paper>
                </Grid>

                {/* Streak */}
                <Grid item xs={12} sm={6} lg={3}>
                     <Paper sx={{ p: 3, borderRadius: '16px', position: 'relative', overflow: 'hidden', height: '100%', border: '1px solid', borderColor: 'divider' }}>
                         <Box sx={{ position: 'absolute', top: 0, right: 0, p: 3, opacity: 0.05 }}>
                            <FireIcon sx={{ fontSize: 80, color: '#EF4444' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'text.secondary', mb: 2 }}>
                            <FireIcon sx={{ fontSize: 20 }} />
                            <Typography variant="caption" fontWeight={700} sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>Study Streak</Typography>
                        </Box>
                        <Typography variant="h3" fontWeight={800}>{streak} <span style={{ fontSize: '1.5rem', fontWeight: 500 }}>Day{streak !== 1 ? 's' : ''}</span></Typography>
                         <Typography variant="body2" color="text.secondary" fontWeight={500}>
                            {streak > 0 ? 'Keep it up! 🔥' : 'Start your streak today!'}
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>
            
            <Grid container spacing={4}>
                {/* Main Content: Chart & Questions */}
                <Grid item xs={12} lg={8}>
                     {/* Performance Analysis Chart */}
                    {chartData.length > 0 && (
                        <Paper sx={{ p: 4, borderRadius: '16px', border: '1px solid', borderColor: 'divider', mb: 4 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
                                <Box>
                                    <Typography variant="h6" fontWeight={700}>Performance Analysis</Typography>
                                    <Typography variant="body2" color="text.secondary">Score progression during quiz</Typography>
                                </Box>
                                <Chip 
                                    label={`Final: ${percentage}%`}
                                    size="small"
                                    sx={{ 
                                        bgcolor: percentage >= 70 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                                        color: percentage >= 70 ? 'success.main' : 'error.main',
                                        fontWeight: 700
                                    }}
                                />
                            </Box>
                            <Box sx={{ height: 250, width: '100%' }}>
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={chartData}>
                                        <defs>
                                            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.2}/>
                                                <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0}/>
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} vertical={false} />
                                        <XAxis dataKey="name" stroke={theme.palette.text.secondary} tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                        <YAxis domain={[0, 100]} stroke={theme.palette.text.secondary} tick={{ fontSize: 12 }} axisLine={false} tickLine={false} />
                                        <Tooltip 
                                            contentStyle={{ backgroundColor: theme.palette.background.paper, borderRadius: '8px', border: `1px solid ${theme.palette.divider}` }}
                                            itemStyle={{ color: theme.palette.text.primary }}
                                            formatter={(value) => [`${value}%`, 'Cumulative Score']}
                                        />
                                        <Area type="monotone" dataKey="score" stroke={theme.palette.primary.main} strokeWidth={3} fillOpacity={1} fill="url(#colorScore)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </Box>
                        </Paper>
                    )}

                    {/* Question Review List */}
                    <Collapse in={showQuestions}>
                        <Box id="question-review" sx={{ mb: 4 }}>
                            <Typography variant="h5" fontWeight={700} sx={{ mb: 2 }}>Question Review</Typography>
                            <Paper sx={{ borderRadius: '16px', border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
                                {/* Table Header */}
                                <Box sx={{ 
                                    display: 'grid', 
                                    gridTemplateColumns: '50px 1fr 100px', 
                                    gap: 2, 
                                    p: 2, 
                                    bgcolor: 'action.hover',
                                    borderBottom: '1px solid',
                                    borderColor: 'divider'
                                }}>
                                    <Typography variant="caption" fontWeight={700} align="center" color="text.secondary">#</Typography>
                                    <Typography variant="caption" fontWeight={700} color="text.secondary">QUESTION</Typography>
                                    <Typography variant="caption" fontWeight={700} align="center" color="text.secondary">STATUS</Typography>
                                </Box>
                                
                                {/* Rows */}
                                {answers.map((ans, idx) => (
                                    <Box 
                                        key={idx}
                                        sx={{ 
                                            display: 'grid', 
                                            gridTemplateColumns: { xs: '1fr', sm: '50px 1fr 100px' },
                                            gap: { xs: 1, sm: 2 }, 
                                            p: 2, 
                                            borderBottom: '1px solid',
                                            borderColor: 'divider',
                                            alignItems: 'center',
                                            bgcolor: 'background.paper',
                                            transition: 'bgcolor 0.2s',
                                            '&:hover': { bgcolor: 'action.hover' },
                                            '&:last-child': { borderBottom: 'none' }
                                        }}
                                    >
                                        <Typography variant="body2" fontWeight={700} color="text.secondary" align="center" sx={{ display: { xs: 'none', sm: 'block' } }}>
                                            {String(idx + 1).padStart(2, '0')}
                                        </Typography>

                                        <Box>
                                            <Typography variant="body2" fontWeight={600} sx={{ mb: 0.5 }}>
                                                {ans.question}
                                            </Typography>
                                            <Box sx={{ display: 'flex', gap: 2, fontSize: '0.75rem', flexWrap: 'wrap' }}>
                                                <Typography variant="inherit" color={ans.isCorrect ? 'success.main' : 'error.main'} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                    {ans.isCorrect ? <CheckIcon sx={{ fontSize: 12 }} /> : <CloseIcon sx={{ fontSize: 12 }} />}
                                                    You: {ans.userAnswer === 'TIMEOUT' ? 'Timed Out' : ans.userAnswer}
                                                </Typography>
                                                {!ans.isCorrect && (
                                                    <Typography variant="inherit" color="text.secondary">
                                                        Correct: {ans.correctAnswer}
                                                    </Typography>
                                                )}
                                            </Box>
                                            {ans.explanation && !ans.isCorrect && (
                                                <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block', fontStyle: 'italic' }}>
                                                    {ans.explanation}
                                                </Typography>
                                            )}
                                        </Box>

                                        <Box sx={{ display: 'flex', justifyContent: 'center' }}>
                                            <Chip 
                                                icon={ans.isCorrect ? <CheckIcon sx={{ fontSize: '14px !important' }} /> : <CloseIcon sx={{ fontSize: '14px !important' }} />}
                                                label={ans.isCorrect ? 'Correct' : 'Wrong'}
                                                size="small"
                                                color={ans.isCorrect ? 'success' : 'error'}
                                                variant="outlined"
                                                sx={{ fontWeight: 700, borderRadius: '6px' }}
                                            />
                                        </Box>
                                    </Box>
                                ))}
                                {answers.length === 0 && (
                                    <Box sx={{ p: 4, textAlign: 'center' }}>
                                        <Typography variant="body2" color="text.secondary">No detailed answer data available.</Typography>
                                    </Box>
                                )}
                            </Paper>
                        </Box>
                    </Collapse>
                    
                    {/* Show review button if questions hidden */}
                    {!showQuestions && (
                        <Box sx={{ textAlign: 'center', mb: 4 }}>
                            <Button 
                                variant="outlined"
                                onClick={() => setShowQuestions(true)}
                                endIcon={<ExpandMoreIcon />}
                                sx={{ fontWeight: 600, borderColor: 'divider', color: 'text.secondary' }}
                            >
                                Show Question Review ({answers.length} questions)
                            </Button>
                        </Box>
                    )}
                </Grid>

                {/* Sidebar: AI Insights & Topics */}
                <Grid item xs={12} lg={4}>
                    <Box sx={{ mb: 4 }}>
                         <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                            <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'secondary.main', color: 'secondary.contrastText', display: 'flex' }}>
                                <AutoAwesomeIcon fontSize="small" />
                            </Box>
                            <Typography variant="h5" fontWeight={700}>AI Insights</Typography>
                         </Box>

                         <Grid container spacing={2}>
                             <Grid item xs={12}>
                                 <Paper sx={{ p: 3, borderRadius: '12px', bgcolor: percentage >= 70 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', border: '1px solid', borderColor: percentage >= 70 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)' }}>
                                     <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: percentage >= 70 ? '#10B981' : '#EF4444', mb: 1 }}>
                                         <ThumbUpIcon fontSize="small" />
                                         <Typography variant="subtitle2" fontWeight={800}>
                                            {percentage >= 70 ? 'Strong Performance' : 'Room to Improve'}
                                         </Typography>
                                     </Box>
                                     <Typography variant="body2" color="text.primary" sx={{ opacity: 0.9, lineHeight: 1.6 }}>
                                         {percentage >= 90 ? 'Outstanding! You have mastered this material. Focus on maintaining this level.' :
                                          percentage >= 70 ? `Great job! You answered ${score} out of ${total} correctly. Review the ${total - score} missed questions.` :
                                          `You answered ${score} out of ${total} correctly. A focused study session will help you improve.`}
                                     </Typography>
                                 </Paper>
                             </Grid>
                             {avgTimePerQuestion && (
                                 <Grid item xs={12}>
                                     <Paper sx={{ p: 3, borderRadius: '12px', bgcolor: 'rgba(19, 127, 236, 0.05)', border: '1px solid', borderColor: 'rgba(19, 127, 236, 0.2)' }}>
                                         <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: 'primary.main', mb: 1 }}>
                                             <SpeedIcon fontSize="small" />
                                             <Typography variant="subtitle2" fontWeight={800}>Speed Analysis</Typography>
                                         </Box>
                                         <Typography variant="body2" color="text.primary" sx={{ opacity: 0.9, lineHeight: 1.6 }}>
                                             {avgTimePerQuestion <= 15 ? 'Excellent pace! You answered questions quickly and confidently.' :
                                              avgTimePerQuestion <= 25 ? `Average ${avgTimePerQuestion}s per question. Good balance of speed and accuracy.` :
                                              `Average ${avgTimePerQuestion}s per question. Taking more time is okay — accuracy matters more than speed.`}
                                         </Typography>
                                     </Paper>
                                 </Grid>
                             )}
                             <Grid item xs={12}>
                                 <Paper sx={{ p: 3, borderRadius: '12px', bgcolor: 'rgba(245, 158, 11, 0.1)', border: '1px solid', borderColor: 'rgba(245, 158, 11, 0.2)' }}>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, color: '#F59E0B', mb: 1 }}>
                                         <LightbulbIcon fontSize="small" />
                                         <Typography variant="subtitle2" fontWeight={800}>Recommended Focus</Typography>
                                     </Box>
                                     <Typography variant="body2" color="text.primary" sx={{ opacity: 0.9, lineHeight: 1.6 }}>
                                         {total - score > 0 
                                            ? `Review the ${total - score} questions you missed. Consider generating a study plan to strengthen weak areas.`
                                            : 'Excellent work! Try a harder quiz or explore new topics to keep challenging yourself.'}
                                     </Typography>
                                     {total - score > 0 && (
                                         <Button size="small" sx={{ mt: 1, color: '#F59E0B', fontWeight: 700 }} onClick={() => navigate(`/study-plan?noteId=${noteId}`)}>
                                             Go to Study Plan →
                                         </Button>
                                     )}
                                 </Paper>
                             </Grid>
                         </Grid>
                    </Box>

                    {/* Topic Breakdown  */}
                    <Paper sx={{ p: 3, borderRadius: '16px', border: '1px solid', borderColor: 'divider' }}>
                         <Typography variant="h6" fontWeight={700} sx={{ mb: 3 }}>Performance Breakdown</Typography>
                         
                         <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                             {topicBreakdown.length > 0 ? (
                                 topicBreakdown.map((t, i) => (
                                     <Box key={t.label}>
                                         <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                             <Typography variant="body2" fontWeight={600} color="text.secondary">{t.label}</Typography>
                                             <Typography variant="body2" fontWeight={700}>{t.percentage}%</Typography>
                                         </Box>
                                         <LinearProgress 
                                            variant="determinate" 
                                            value={t.percentage} 
                                            color={t.color}
                                            sx={{ 
                                                height: 8, 
                                                borderRadius: 4, 
                                                bgcolor: 'action.hover',
                                                '& .MuiLinearProgress-bar': { borderRadius: 4 } 
                                            }} 
                                        />
                                     </Box>
                                 ))
                             ) : (
                                 <Box sx={{ mb: 2 }}>
                                     <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                         <Typography variant="body2" fontWeight={600} color="text.secondary">Your Score</Typography>
                                         <Typography variant="body2" fontWeight={700}>{percentage}%</Typography>
                                     </Box>
                                     <LinearProgress 
                                        variant="determinate" 
                                        value={percentage} 
                                        sx={{ 
                                            height: 8, 
                                            borderRadius: 4, 
                                            bgcolor: 'action.hover',
                                            '& .MuiLinearProgress-bar': { borderRadius: 4 } 
                                        }} 
                                    />
                                 </Box>
                             )}
                         </Box>
                         
                         <Divider sx={{ my: 3 }} />
                         
                         <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                             <Box sx={{ textAlign: 'center' }}>
                                 <Typography variant="h5" fontWeight={800} color="success.main">{score}</Typography>
                                 <Typography variant="caption" color="text.secondary">Correct</Typography>
                             </Box>
                             <Box sx={{ textAlign: 'center' }}>
                                 <Typography variant="h5" fontWeight={800} color="error.main">{total - score}</Typography>
                                 <Typography variant="caption" color="text.secondary">Incorrect</Typography>
                             </Box>
                             <Box sx={{ textAlign: 'center' }}>
                                 <Typography variant="h5" fontWeight={800} color="primary.main">{total}</Typography>
                                 <Typography variant="caption" color="text.secondary">Total</Typography>
                             </Box>
                         </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    </Box>
  );
}

export default QuizResult;
