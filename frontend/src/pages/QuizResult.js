import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Card,
  CardContent,
  Typography,
  Button,
  Collapse,
  Divider,
  Chip,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import './QuizComplete.css';
import VideoGenerator from '../components/VideoGenerator';

function QuizResult() {
  const location = useLocation();
  const navigate = useNavigate();
  const { score, total, noteId, answers } = location.state || {};
  const [showAnswers, setShowAnswers] = React.useState(false);

  useEffect(() => {
    if (score === undefined || !total || !noteId) {
      navigate('/dashboard');
    }
  }, [score, total, noteId, navigate]);

  // Confetti effect
  useEffect(() => {
    if (score !== undefined && total > 0 && score / total >= 0.75) {
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
  }, [score, total]);

  const percentage = total ? Math.round((score / total) * 100) : 0;

  // Store current score
  useEffect(() => {
    if (noteId && percentage !== undefined) {
      localStorage.setItem(`quiz_${noteId}_score`, percentage.toString());
    }
  }, [noteId, percentage]);

  if (score === undefined || !total || !noteId) return null;

  const previousScore = localStorage.getItem(`quiz_${noteId}_score`);
  const previousPercentage = previousScore ? parseInt(previousScore) : null;

  // Determine message based on performance
  let message = '';
  let emoji = '';
  let messageColor = '';
  
  if (percentage >= 90) {
    message = "Outstanding! You're a master! 🌟";
    emoji = "🎉";
    messageColor = "#10b981";
  } else if (percentage >= 75) {
    message = "Excellent work! Keep it up! 💪";
    emoji = "✨";
    messageColor = "#3b82f6";
  } else if (percentage >= 60) {
    message = "Good job! You're making progress! 📈";
    emoji = "👍";
    messageColor = "#8b5cf6";
  } else if (percentage >= 40) {
    message = "Keep practicing! You'll get better! 💡";
    emoji = "🌱";
    messageColor = "#f59e0b";
  } else {
    message = "Don't give up! Every attempt makes you stronger! 💪";
    emoji = "🚀";
    messageColor = "#ef4444";
  }
  
  // Compare with previous score
  let comparisonMessage = '';
  if (previousPercentage !== null) {
    if (percentage > previousPercentage) {
      comparisonMessage = `🎊 Improved by ${percentage - previousPercentage}% from last time!`;
      messageColor = "#10b981";
    } else if (percentage < previousPercentage) {
      comparisonMessage = `Keep trying! Previous best: ${previousPercentage}%`;
      messageColor = "#f59e0b";
    } else {
      comparisonMessage = `Same as before. Try to beat your score!`;
    }
  }

  return (
    <Box
      sx={{
        minHeight: "80vh", // Adjusted since it's inside sidebar layout
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        py: 4
      }}
    >
      <Container maxWidth="sm">
        <Card className="quiz-complete-card" sx={{ 
          background: "rgba(255, 255, 255, 0.95)",
          backdropFilter: "blur(20px)",
          color: '#1e293b',
          borderRadius: '24px',
          boxShadow: '0 20px 60px rgba(0,0,0,0.1)',
          border: "1px solid rgba(255,255,255,0.8)",
          overflow: 'visible'
        }}>
          <CardContent sx={{ padding: { xs: 4, md: 6 }, textAlign: 'center' }}>
            <div className="quiz-emoji" style={{ fontSize: '80px', marginBottom: '24px', filter: 'drop-shadow(0 10px 20px rgba(0,0,0,0.15))' }}>{emoji}</div>
            
            <Typography variant="h3" sx={{ fontWeight: 800, mb: 2, background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", backgroundClip: "text", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
              Quiz Complete!
            </Typography>
            
            <Typography variant="h6" sx={{ mb: 4, color: '#64748b', fontWeight: 600 }}>
              {message}
            </Typography>
            
            <div style={{
              background: 'linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%)',
              borderRadius: '20px',
              padding: '32px',
              marginBottom: '32px',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.05)',
              border: '1px solid #e2e8f0'
            }}>
              <Typography variant="h2" sx={{ fontWeight: 800, mb: 1, color: '#1e293b' }}>
                {score} / {total}
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 700, color: messageColor }}>
                {percentage}% Score
              </Typography>
              
              {comparisonMessage && (
                <Typography 
                  variant="body1" 
                  sx={{ 
                    mt: 2, 
                    padding: '12px 20px',
                    background: 'white',
                    borderRadius: '12px',
                    fontWeight: 600,
                    color: '#475569',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                    display: 'inline-block'
                  }}
                >
                  {comparisonMessage}
                </Typography>
              )}
            </div>

            {/* Statistics Breakdown */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '24px',
              marginBottom: '32px',
              textAlign: 'left',
              boxShadow: '0 10px 30px rgba(0,0,0,0.05)',
              border: '1px solid #e2e8f0'
            }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 3, textAlign: 'center', color: '#64748b', textTransform: 'uppercase', letterSpacing: '1px', fontSize: '0.875rem' }}>
                Performance Breakdown
              </Typography>
              
              <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '20px' }}>
                <div style={{ textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 800, color: '#10b981', mb: 0.5 }}>
                    {score}
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 600 }}>Correct</Typography>
                </div>
                
                <div style={{ textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 800, color: '#ef4444', mb: 0.5 }}>
                    {total - score}
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 600 }}>Incorrect</Typography>
                </div>
                
                <div style={{ textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 800, color: '#f59e0b', mb: 0.5 }}>
                    {percentage}%
                  </Typography>
                  <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 600 }}>Accuracy</Typography>
                </div>
              </div>
              
              {/* Progress bar */}
              <div style={{
                width: '100%',
                height: '12px',
                background: '#f1f5f9',
                borderRadius: '6px',
                overflow: 'hidden',
                marginTop: '16px'
              }}>
                <div style={{
                  width: `${percentage}%`,
                  height: '100%',
                  background: `linear-gradient(90deg, ${messageColor} 0%, ${messageColor}dd 100%)`,
                  transition: 'width 1s ease-out',
                  borderRadius: '6px'
                }} />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                onClick={() => navigate(`/quiz?noteId=${noteId}&n=${total}`)}
                sx={{
                  height: "56px",
                  minWidth: 160,
                  borderColor: "#e2e8f0",
                  color: "#64748b",
                  borderRadius: "16px",
                  px: 4,
                  fontWeight: 700,
                  fontSize: "16px",
                  borderWidth: '2px',
                  '&:hover': {
                    borderColor: "#cbd5e0",
                    background: '#f8fafc',
                    borderWidth: '2px',
                    transform: 'translateY(-2px)',
                  }
                }}
              >
                Try Again 🔄
              </Button>
              
              {answers && answers.length > 0 && (
                <Button
                  variant="outlined"
                  onClick={() => setShowAnswers(!showAnswers)}
                  startIcon={showAnswers ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  sx={{
                    height: "56px",
                    minWidth: 180,
                    borderColor: "#667eea",
                    color: "#667eea",
                    borderRadius: "16px",
                    px: 4,
                    fontWeight: 700,
                    fontSize: "16px",
                    borderWidth: '2px',
                    '&:hover': {
                      borderColor: "#764ba2",
                      background: '#f5f7ff',
                      borderWidth: '2px',
                      transform: 'translateY(-2px)',
                    }
                  }}
                >
                  {showAnswers ? 'Hide' : 'View'} Answers 📝
                </Button>
              )}
              
              <Button
                variant="contained"
                onClick={() => {
                  navigate(`/study-plan?noteId=${noteId}`);
                }}
                sx={{
                  height: "56px",
                  minWidth: 180,
                  background: "linear-gradient(135deg, #10b981 0%, #059669 100%)",
                  color: "white",
                  borderRadius: "16px",
                  px: 4,
                  fontWeight: 700,
                  fontSize: "16px",
                  boxShadow: "0 10px 25px rgba(16,185,129,0.3)",
                  '&:hover': {
                    background: 'linear-gradient(135deg, #059669 0%, #047857 100%)',
                    transform: 'translateY(-2px)',
                    boxShadow: "0 15px 35px rgba(16,185,129,0.4)",
                  }
                }}
              >
                Study Plan 📊
              </Button>
            </div>

            {/* Answer Review Section */}
            {answers && answers.length > 0 && (
              <Collapse in={showAnswers} sx={{ mt: 4 }}>
                <Divider sx={{ mb: 3 }} />
                <Typography variant="h5" sx={{ fontWeight: 700, mb: 3, color: '#1e293b', textAlign: 'center' }}>
                  📝 Answer Review
                </Typography>
                {answers.map((ans, index) => (
                  <Box
                    key={index}
                    sx={{
                      mb: 3,
                      p: 3,
                      background: ans.isCorrect ? '#f0fdf4' : '#fef2f2',
                      borderRadius: '16px',
                      border: `2px solid ${ans.isCorrect ? '#10b981' : '#ef4444'}`,
                      boxShadow: '0 4px 12px rgba(0,0,0,0.05)'
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                      {ans.isCorrect ? (
                        <CheckCircleIcon sx={{ color: '#10b981', fontSize: 28 }} />
                      ) : (
                        <CancelIcon sx={{ color: '#ef4444', fontSize: 28 }} />
                      )}
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#64748b' }}>
                        Question {index + 1}
                      </Typography>
                      <Chip 
                        label={ans.isCorrect ? 'Correct' : 'Incorrect'}
                        size="small"
                        sx={{
                          bgcolor: ans.isCorrect ? '#10b981' : '#ef4444',
                          color: 'white',
                          fontWeight: 600
                        }}
                      />
                    </Box>
                    
                    <Typography variant="body1" sx={{ fontWeight: 600, mb: 2, color: '#1e293b' }}>
                      {ans.question}
                    </Typography>
                    
                    <Box sx={{ pl: 2 }}>
                      {Object.entries(ans.options).map(([key, value]) => {
                        const isUserAnswer = key === ans.userAnswer;
                        const isCorrectAnswer = key === ans.correctAnswer;
                        
                        return (
                          <Box
                            key={key}
                            sx={{
                              p: 1.5,
                              mb: 1,
                              borderRadius: '8px',
                              background: isCorrectAnswer 
                                ? '#d1fae5' 
                                : isUserAnswer 
                                ? '#fee2e2' 
                                : 'white',
                              border: `2px solid ${
                                isCorrectAnswer 
                                  ? '#10b981' 
                                  : isUserAnswer 
                                  ? '#ef4444' 
                                  : '#e2e8f0'
                              }`,
                              display: 'flex',
                              alignItems: 'center',
                              gap: 1
                            }}
                          >
                            <Typography 
                              variant="body2" 
                              sx={{ 
                                fontWeight: 600, 
                                color: isCorrectAnswer || isUserAnswer ? '#1e293b' : '#64748b',
                                flex: 1
                              }}
                            >
                              {key}. {value}
                            </Typography>
                            {isCorrectAnswer && (
                              <Chip 
                                label="Correct Answer" 
                                size="small" 
                                sx={{ bgcolor: '#10b981', color: 'white', fontWeight: 600 }}
                              />
                            )}
                            {isUserAnswer && !isCorrectAnswer && (
                              <Chip 
                                label="Your Answer" 
                                size="small" 
                                sx={{ bgcolor: '#ef4444', color: 'white', fontWeight: 600 }}
                              />
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                    
                    {ans.explanation && (
                      <Box sx={{ mt: 2, p: 2, background: '#f8fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                        <Typography variant="caption" sx={{ fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                          Explanation:
                        </Typography>
                        <Typography variant="body2" sx={{ mt: 0.5, color: '#475569' }}>
                          {ans.explanation}
                        </Typography>
                        
                        {/* AI Video Generator */}
                        <VideoGenerator questionId={ans.questionId} text={ans.question} />
                      </Box>
                    )}
                  </Box>
                ))}
              </Collapse>
            )}
          </CardContent>
        </Card>
      </Container>
    </Box>
  );
}

export default QuizResult;
