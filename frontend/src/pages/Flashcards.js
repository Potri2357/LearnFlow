import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Card, Box, CircularProgress, Alert, Button, 
  MenuItem, Select, FormControl, InputLabel, Chip, Stack, IconButton,
  Paper, LinearProgress
} from '@mui/material';
import { 
  ArrowForward as NextIcon,
  ArrowBack as PrevIcon,
  CheckCircle as CheckIcon,
  Shuffle as ShuffleIcon,
  Refresh as RefreshIcon,
  School as SchoolIcon,
  AutoAwesome as SparkleIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import API from '../api/api';
import { useAuth } from '../context/AuthContext';

export default function Flashcards() {
  const { user } = useAuth();
  
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState('');
  const [flashcards, setFlashcards] = useState([]);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [learnedCards, setLearnedCards] = useState(new Set());
  const [direction, setDirection] = useState(0);

  useEffect(() => {
    fetchLectures();
  }, []);

  const fetchLectures = async () => {
    try {
      const response = await API.get('lectures/');
      setLectures(response.data);
    } catch (err) {
      console.error('Failed to fetch lectures', err);
      setError('Failed to load lectures. Please refresh.');
    }
  };

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (flashcards.length === 0) return;
      if (e.key === 'ArrowLeft') handlePrev();
      else if (e.key === 'ArrowRight') handleNext();
      else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        setIsFlipped(!isFlipped);
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [flashcards, isFlipped, currentIndex]);

  const handleGenerate = async () => {
    if (!selectedLecture) return;
    
    setGenerating(true);
    setError('');
    setFlashcards([]);
    setCurrentIndex(0);
    setIsFlipped(false);
    setLearnedCards(new Set());
    setDirection(0);

    try {
      const response = await API.post('flashcards/generate/', { 
        note_id: selectedLecture, 
        count: 15 
      });
      setFlashcards(response.data.flashcards);
    } catch (err) {
      console.error("Generation error:", err);
      setError(err.response?.data?.error || 'Failed to generate flashcards. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const handleNext = () => {
    setDirection(1);
    setIsFlipped(false);
    setTimeout(() => {
      setCurrentIndex((prev) => (prev + 1) % flashcards.length);
    }, 200);
  };

  const handlePrev = () => {
    setDirection(-1);
    setIsFlipped(false);
    setTimeout(() => {
      setCurrentIndex((prev) => (prev - 1 + flashcards.length) % flashcards.length);
    }, 200);
  };

  const handleShuffle = () => {
    const shuffled = [...flashcards].sort(() => Math.random() - 0.5);
    setFlashcards(shuffled);
    setCurrentIndex(0);
    setIsFlipped(false);
    setDirection(0);
  };

  const toggleLearned = () => {
    const newLearned = new Set(learnedCards);
    if (newLearned.has(currentIndex)) {
      newLearned.delete(currentIndex);
    } else {
      newLearned.add(currentIndex);
    }
    setLearnedCards(newLearned);
  };

  const progress = flashcards.length > 0 ? ((currentIndex + 1) / flashcards.length) * 100 : 0;

  // Animation variants
  const variants = {
    enter: (direction) => ({
      x: direction > 0 ? 300 : -300,
      opacity: 0,
      scale: 0.8,
      rotateY: 0,
    }),
    center: {
      zIndex: 1,
      x: 0,
      opacity: 1,
      scale: 1,
      rotateY: 0,
      transition: {
        duration: 0.3,
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    },
    exit: (direction) => ({
      zIndex: 0,
      x: direction < 0 ? 300 : -300,
      opacity: 0,
      scale: 0.8,
      rotateY: 0,
      transition: { duration: 0.3 }
    })
  };

  return (
    <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
      {/* Header matching UploadNote.js */}
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
        🎴 Smart Flashcards
      </Typography>

      {/* Control Panel */}
      <Card
        sx={{
          p: 4,
          mb: 4,
          background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
          boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
          borderRadius: 3,
        }}
      >
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
          <FormControl fullWidth sx={{ 
            '& .MuiOutlinedInput-root': { bgcolor: 'white' } 
          }}>
            <InputLabel>Select Lecture Note</InputLabel>
            <Select
              value={selectedLecture}
              label="Select Lecture Note"
              onChange={(e) => setSelectedLecture(e.target.value)}
            >
              {lectures.map((l) => (
                <MenuItem key={l.id} value={l.id}>{l.title}</MenuItem>
              ))}
            </Select>
          </FormControl>
          
          <Button 
            variant="contained" 
            onClick={handleGenerate}
            disabled={!selectedLecture || generating}
            sx={{ 
              height: 56,
              px: 4,
              whiteSpace: 'nowrap',
            }}
          >
            {generating ? 'Creating...' : 'Generate'}
          </Button>
        </Stack>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>{error}</Alert>
      )}

      {flashcards.length > 0 && (
        <Box>
          {/* Progress Info */}
          <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography variant="subtitle1" fontWeight={600} color="text.secondary">
              Card {currentIndex + 1} of {flashcards.length}
            </Typography>
            <Chip 
              icon={<CheckIcon />} 
              label={`${learnedCards.size} Mastered`} 
              color="success" 
              variant={learnedCards.size > 0 ? "filled" : "outlined"}
            />
          </Box>

          <LinearProgress 
            variant="determinate" 
            value={progress} 
            sx={{ 
              mb: 4, 
              height: 8, 
              borderRadius: 4,
              bgcolor: '#e2e8f0',
              '& .MuiLinearProgress-bar': {
                background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                borderRadius: 4,
              }
            }} 
          />

          {/* Card Area */}
          <Box 
            sx={{ 
              position: 'relative', 
              height: 400, 
              width: '100%',
              perspective: 1000,
              mb: 4
            }}
          >
            <AnimatePresence initial={false} custom={direction} mode="wait">
              <motion.div
                key={currentIndex}
                custom={direction}
                variants={variants}
                initial="enter"
                animate="center"
                exit="exit"
                style={{
                  position: 'absolute',
                  width: '100%',
                  height: '100%',
                }}
              >
                <Box
                  onClick={() => setIsFlipped(!isFlipped)}
                  sx={{
                    width: '100%',
                    height: '100%',
                    position: 'relative',
                    cursor: 'pointer',
                    transformStyle: 'preserve-3d',
                    transition: 'transform 0.6s cubic-bezier(0.4, 0.0, 0.2, 1)',
                    transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
                  }}
                >
                  {/* Front */}
                  <Card
                    elevation={4}
                    sx={{
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      backfaceVisibility: 'hidden',
                      borderRadius: 4,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      p: 4,
                      background: 'linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)',
                      border: '1px solid rgba(102, 126, 234, 0.1)',
                      boxShadow: '0 8px 32px rgba(102, 126, 234, 0.1)',
                    }}
                  >
                    <Box sx={{ position: 'absolute', top: 20, right: 20, opacity: 0.1 }}>
                      <SchoolIcon sx={{ fontSize: 100, color: '#667eea' }} />
                    </Box>
                    <Typography 
                      variant="overline" 
                      sx={{ 
                        color: '#667eea', 
                        fontWeight: 700, 
                        letterSpacing: 2,
                        mb: 3,
                        borderBottom: '2px solid #667eea',
                        pb: 0.5
                      }}
                    >
                      Question
                    </Typography>
                    <Typography 
                      variant="h5" 
                      align="center" 
                      sx={{ 
                        fontWeight: 600, 
                        color: '#2d3748',
                        lineHeight: 1.6,
                        maxWidth: '90%',
                        zIndex: 1
                      }}
                    >
                      {flashcards[currentIndex].front}
                    </Typography>
                    <Typography variant="caption" sx={{ position: 'absolute', bottom: 24, color: '#999' }}>
                      Click to flip
                    </Typography>
                  </Card>

                  {/* Back */}
                  <Card
                    elevation={4}
                    sx={{
                      position: 'absolute',
                      width: '100%',
                      height: '100%',
                      backfaceVisibility: 'hidden',
                      transform: 'rotateY(180deg)',
                      borderRadius: 4,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      p: 4,
                      background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                      color: 'white',
                    }}
                  >
                    <Typography 
                      variant="overline" 
                      sx={{ 
                        color: 'rgba(255,255,255,0.8)', 
                        fontWeight: 700, 
                        letterSpacing: 1.5,
                        mb: 2
                      }}
                    >
                      Answer
                    </Typography>
                    <Typography 
                      variant="h5" 
                      align="center" 
                      sx={{ 
                        fontWeight: 500, 
                        lineHeight: 1.6
                      }}
                    >
                      {flashcards[currentIndex].back}
                    </Typography>
                  </Card>
                </Box>
              </motion.div>
            </AnimatePresence>
          </Box>

          {/* Controls */}
          <Stack 
            direction="row" 
            justifyContent="center" 
            alignItems="center" 
            spacing={3} 
          >
            <IconButton 
              onClick={handlePrev}
              sx={{ 
                bgcolor: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                '&:hover': { bgcolor: '#f8fafc' }
              }}
            >
              <PrevIcon />
            </IconButton>

            <Button
              variant={learnedCards.has(currentIndex) ? "contained" : "outlined"}
              onClick={toggleLearned}
              startIcon={<CheckIcon />}
              sx={{
                px: 4,
                py: 1.5,
                borderRadius: 2,
                ...(learnedCards.has(currentIndex) ? {
                  bgcolor: '#10b981',
                  '&:hover': { bgcolor: '#059669' }
                } : {
                  // Outlined button will use theme defaults for color/border
                  // We just need to override color if we want green specifically for "Mark as Learned"
                  color: '#10b981',
                  borderColor: '#10b981',
                  '&:hover': { bgcolor: 'rgba(16, 185, 129, 0.05)', borderColor: '#059669' }
                })
              }}
            >
              {learnedCards.has(currentIndex) ? 'Mastered' : 'Mark as Learned'}
            </Button>

            <IconButton 
              onClick={handleNext}
              sx={{ 
                bgcolor: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                '&:hover': { bgcolor: '#f8fafc' }
              }}
            >
              <NextIcon />
            </IconButton>
          </Stack>
          
          <Box sx={{ textAlign: 'center', mt: 3 }}>
            <Button 
              startIcon={<ShuffleIcon />} 
              onClick={handleShuffle}
              size="small"
              sx={{ color: '#667eea' }}
            >
              Shuffle
            </Button>
          </Box>
        </Box>
      )}
    </Container>
  );
}
