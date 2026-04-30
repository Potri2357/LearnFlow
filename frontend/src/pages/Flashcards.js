import React, { useState, useEffect, useCallback } from 'react';
import { 
  Container, Typography, Box, CircularProgress, Alert, Button, 
  MenuItem, Select, FormControl, InputLabel, Chip, IconButton,
  Paper, LinearProgress, Grid, Divider, useTheme, Tooltip,
  Tab, Tabs, Table, TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import { 
  ArrowForward as NextIcon,
  ArrowBack as PrevIcon,
  CheckCircle as CheckCircleIcon,
  Shuffle as ShuffleIcon,
  AutoAwesome as SparkleIcon,
  KeyboardArrowLeft,
  KeyboardArrowRight,
  BookmarkBorder as BookmarkBorderIcon,
  Bookmark as BookmarkIcon,
  RestartAlt as RestartIcon,
  Psychology as PsychologyIcon,
  EmojiEvents as TrophyIcon,
  TrendingUp as TrendingUpIcon,
  Close as CloseIcon,
  Check as CheckIcon
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import API from '../api/api';

export default function Flashcards() {
  const theme = useTheme();
  
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState('');
  const [flashcards, setFlashcards] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isFlipped, setIsFlipped] = useState(false);
  const [bookmarked, setBookmarked] = useState(new Set());
  const [ratings, setRatings] = useState({}); // { cardIndex: 'again' | 'hard' | 'good' | 'easy' }
  const [direction, setDirection] = useState(0);
  const [sessionComplete, setSessionComplete] = useState(false);

  useEffect(() => {
    let cancelled = false;
    API.get('lectures/').then(r => {
      if (!cancelled) setLectures(r.data || []);
    }).catch(() => {
      if (!cancelled) setError('Failed to load lectures. Please refresh.');
    }).finally(() => {
      if (!cancelled) setLoading(false);
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedLecture) {
      setFlashcards([]);
      return;
    }
    setLoading(true);
    API.get(`flashcards/?note_id=${selectedLecture}`)
       .then(res => {
           setFlashcards(res.data || []);
           setCurrentIndex(0);
           setIsFlipped(false);
           setSessionComplete(false);
           setRatings({});
       })
       .catch(err => setError("Failed to load cards."))
       .finally(() => setLoading(false));
  }, [selectedLecture]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (flashcards.length === 0 || sessionComplete) return;
      if (e.key === 'ArrowLeft') { e.preventDefault(); handlePrev(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); handleNext(); }
      else if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        setIsFlipped(f => !f);
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [flashcards.length, sessionComplete, currentIndex]);

  const handleGenerate = async () => {
    if (!selectedLecture) return;
    setGenerating(true);
    setError('');
    setFlashcards([]);
    setCurrentIndex(0);
    setIsFlipped(false);
    setBookmarked(new Set());
    setRatings({});
    setDirection(0);
    setSessionComplete(false);

    try {
      await API.post('flashcards/generate/', { note_id: selectedLecture, count: 15 });
      const res = await API.get(`flashcards/?note_id=${selectedLecture}`);
      setFlashcards(res.data || []);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate flashcards. Please try again.');
    } finally {
      setGenerating(false);
    }
  };

  const handleNext = useCallback(() => {
    if (currentIndex >= flashcards.length - 1) {
      setSessionComplete(true);
      return;
    }
    setDirection(1);
    setIsFlipped(false);
    setTimeout(() => setCurrentIndex(i => i + 1), 150);
  }, [currentIndex, flashcards.length]);

  const handlePrev = useCallback(() => {
    if (currentIndex <= 0) return;
    setDirection(-1);
    setIsFlipped(false);
    setTimeout(() => setCurrentIndex(i => i - 1), 150);
  }, [currentIndex]);

  const handleRate = async (rating) => {
    const card = flashcards[currentIndex];
    setRatings(r => ({ ...r, [currentIndex]: rating }));
    try {
        if(card.id) await API.post(`flashcards/${card.id}/review/`, { rating });
    } catch(e) { console.error(e) }
    handleNext();
  };

  const handleShuffle = () => {
    const shuffled = [...flashcards].sort(() => Math.random() - 0.5);
    setFlashcards(shuffled);
    setCurrentIndex(0);
    setIsFlipped(false);
    setDirection(0);
    setRatings({});
    setSessionComplete(false);
  };

  const handleRestart = () => {
    setCurrentIndex(0);
    setIsFlipped(false);
    setDirection(0);
    setRatings({});
    setSessionComplete(false);
  };

  const toggleBookmark = (e) => {
    e.stopPropagation();
    setBookmarked(prev => {
      const next = new Set(prev);
      if (next.has(currentIndex)) next.delete(currentIndex);
      else next.add(currentIndex);
      return next;
    });
  };

  const masteredCount = Object.values(ratings).filter(r => r === 'good' || r === 'easy').length;
  const masteryPct = flashcards.length > 0 ? Math.round((masteredCount / flashcards.length) * 100) : 0;
  const progress = flashcards.length > 0 ? ((currentIndex) / flashcards.length) * 100 : 0;
  const currentCard = flashcards[currentIndex];
  const currentRating = ratings[currentIndex];

  const ratingColors = {
    again: ['error.main', 'rgba(239,68,68,0.1)', 'rgba(239,68,68,0.2)'],
    hard: ['warning.main', 'rgba(245,158,11,0.1)', 'rgba(245,158,11,0.2)'],
    good: ['primary.main', 'rgba(19,127,236,0.1)', 'rgba(19,127,236,0.2)'],
    easy: ['success.main', 'rgba(16,185,129,0.1)', 'rgba(16,185,129,0.2)'],
  };

  if (loading) return (
    <Box sx={{ height: '60vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <CircularProgress />
    </Box>
  );

  return (
    <Box sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default',
        display: 'flex',
        flexDirection: 'column'
    }}>
      {/* === HEADER === */}
      <Box sx={{ 
          bgcolor: 'background.paper', 
          borderBottom: '1px solid', 
          borderColor: 'divider',
          px: { xs: 2, md: 4 },
          py: 3
      }}>
          <Container maxWidth="xl">
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 2 }}>
                  <Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                          <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main', display: 'flex' }}>
                              <PsychologyIcon fontSize="small" />
                          </Box>
                          <Typography variant="h4" fontWeight={900} sx={{ letterSpacing: '-0.02em' }}>
                              Flashcard Studio
                          </Typography>
                          {flashcards.length > 0 && (
                              <Chip label="Session Active" color="primary" size="small" sx={{ fontWeight: 700 }} />
                          )}
                      </Box>
                      <Typography variant="body2" color="text.secondary" fontWeight={500}>
                          AI-powered spaced repetition for accelerated learning
                      </Typography>
                  </Box>

                  {/* Lecture Selector */}
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                     <FormControl sx={{ minWidth: 220 }} size="small">
                        <InputLabel>Select Lecture</InputLabel>
                        <Select
                            value={selectedLecture}
                            label="Select Lecture"
                            onChange={(e) => setSelectedLecture(e.target.value)}
                        >
                            {lectures.map((l) => (
                                <MenuItem key={l.id} value={l.id}>{l.title}</MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    <Button 
                        variant="contained"
                        startIcon={generating ? <CircularProgress size={16} color="inherit" /> : <SparkleIcon />}
                        onClick={handleGenerate}
                        disabled={!selectedLecture || generating}
                        sx={{ fontWeight: 700, px: 3, boxShadow: '0 4px 14px 0 rgba(19, 127, 236, 0.4)' }}
                    >
                        {generating ? 'Generating...' : 'Generate Cards'}
                    </Button>
                  </Box>
              </Box>
          </Container>
      </Box>

      {error && (
          <Container maxWidth="xl" sx={{ mt: 2 }}>
              <Alert severity="error" onClose={() => setError('')}>{error}</Alert>
          </Container>
      )}

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider', px: { xs: 2, md: 4 } }}>
          <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
              <Tab label="Study Session" sx={{ fontWeight: 700 }} />
              <Tab label="Flashcard Browser" sx={{ fontWeight: 700 }} />
          </Tabs>
      </Box>

      {activeTab === 0 && (
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {/* === SIDEBAR === */}
          <Box sx={{ 
              width: { xs: 0, md: 280 }, 
              display: { xs: 'none', md: 'flex' },
              flexDirection: 'column',
              borderRight: '1px solid',
              borderColor: 'divider',
              bgcolor: 'background.paper',
              overflow: 'hidden'
          }}>
              <Box sx={{ p: 3, flex: 1, overflowY: 'auto' }}>
                  {/* Stats */}
                  {flashcards.length > 0 && (
                      <Box sx={{ mb: 3 }}>
                          <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                              Session Stats
                          </Typography>
                          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5, mt: 1.5 }}>
                              {[
                                  { label: 'Progress', value: `${currentIndex}/${flashcards.length}`, icon: <TrendingUpIcon sx={{ fontSize: 16 }} /> },
                                  { label: 'Mastered', value: `${masteryPct}%`, icon: <TrophyIcon sx={{ fontSize: 16 }} /> },
                                  { label: 'Bookmarked', value: bookmarked.size, icon: <BookmarkIcon sx={{ fontSize: 16 }} /> },
                                  { label: 'Remaining', value: flashcards.length - currentIndex, icon: <SparkleIcon sx={{ fontSize: 16 }} /> },
                              ].map(stat => (
                                  <Paper key={stat.label} sx={{ p: 1.5, borderRadius: '8px', border: '1px solid', borderColor: 'divider', textAlign: 'center' }}>
                                      <Box sx={{ color: 'primary.main', mb: 0.5 }}>{stat.icon}</Box>
                                      <Typography variant="h6" fontWeight={800} lineHeight={1}>{stat.value}</Typography>
                                      <Typography variant="caption" color="text.secondary">{stat.label}</Typography>
                                  </Paper>
                              ))}
                          </Box>
                          <LinearProgress 
                              variant="determinate" 
                              value={masteryPct}
                              sx={{ mt: 2, height: 6, borderRadius: 3, bgcolor: 'action.hover', '& .MuiLinearProgress-bar': { borderRadius: 3 } }} 
                          />
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                              {masteryPct}% Mastery
                          </Typography>
                      </Box>
                  )}

                  {/* Card Index */}
                  {flashcards.length > 0 && (
                      <Box>
                          <Typography variant="caption" fontWeight={700} color="text.secondary" sx={{ textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                              Cards
                          </Typography>
                          <Box sx={{ mt: 1.5, display: 'flex', flexDirection: 'column', gap: 1 }}>
                            {flashcards.map((card, idx) => {
                                const rating = ratings[idx];
                                const isActive = idx === currentIndex;
                                return (
                                    <Box
                                        key={idx}
                                        onClick={() => { setDirection(idx > currentIndex ? 1 : -1); setCurrentIndex(idx); setIsFlipped(false); }}
                                        sx={{ 
                                            p: 1.5, borderRadius: '8px', cursor: 'pointer',
                                            border: '1px solid',
                                            borderColor: isActive ? 'primary.main' : 'divider',
                                            bgcolor: isActive ? 'rgba(19,127,236,0.08)' : 'transparent',
                                            transition: 'all 0.15s ease',
                                            '&:hover': { bgcolor: isActive ? 'rgba(19,127,236,0.1)' : 'action.hover' },
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: 1
                                        }}
                                    >
                                        <Box sx={{ 
                                            width: 24, height: 24, borderRadius: '6px', flexShrink: 0,
                                            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem',
                                            bgcolor: rating === 'easy' || rating === 'good' ? 'rgba(16,185,129,0.2)' :
                                                     rating === 'hard' ? 'rgba(245,158,11,0.2)' :
                                                     rating === 'again' ? 'rgba(239,68,68,0.2)' :
                                                     isActive ? 'primary.main' : 'action.hover',
                                            color: isActive && !rating ? 'white' : 'text.secondary',
                                            fontWeight: 700
                                        }}>
                                            {rating === 'easy' || rating === 'good' ? <CheckIcon sx={{ fontSize: 14 }} color="success" /> :
                                             rating === 'hard' ? <span style={{ fontSize: 11, color: '#F59E0B' }}>!</span> :
                                             rating === 'again' ? <CloseIcon sx={{ fontSize: 14 }} color="error" /> :
                                             idx + 1}
                                        </Box>
                                        <Typography 
                                            variant="caption" fontWeight={isActive ? 700 : 500} 
                                            color={isActive ? 'text.primary' : 'text.secondary'}
                                            sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}
                                        >
                                            {card.front}
                                        </Typography>
                                        {bookmarked.has(idx) && <BookmarkIcon sx={{ fontSize: 12, color: 'warning.main', flexShrink: 0 }} />}
                                    </Box>
                                );
                            })}
                          </Box>
                      </Box>
                  )}

                  {flashcards.length === 0 && !generating && (
                      <Box sx={{ textAlign: 'center', py: 4, opacity: 0.5 }}>
                          <PsychologyIcon sx={{ fontSize: 48, mb: 1 }} />
                          <Typography variant="body2">No cards yet. Select a lecture and generate!</Typography>
                      </Box>
                  )}
              </Box>
          </Box>

          {/* === MAIN STUDY AREA === */}
          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: { xs: 2, md: 4 }, overflow: 'auto' }}>
              {flashcards.length === 0 ? (
                  /* Empty State */
                  <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Box sx={{ textAlign: 'center', maxWidth: 400 }}>
                          {generating ? (
                              <>
                                  <CircularProgress size={64} sx={{ mb: 3 }} />
                                  <Typography variant="h5" fontWeight={700} gutterBottom>Generating Flashcards...</Typography>
                                  <Typography variant="body2" color="text.secondary">Our AI is creating personalized flashcards from your lecture content.</Typography>
                              </>
                          ) : (
                              <>
                                  <Box sx={{ 
                                      width: 120, height: 120, borderRadius: '24px',
                                      bgcolor: 'rgba(19,127,236,0.08)', border: '2px solid rgba(19,127,236,0.15)',
                                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                                      mx: 'auto', mb: 4
                                  }}>
                                      <SparkleIcon sx={{ fontSize: 56, color: 'primary.main' }} />
                                  </Box>
                                  <Typography variant="h4" fontWeight={800} gutterBottom>
                                      Ready to Study?
                                  </Typography>
                                  <Typography variant="body1" color="text.secondary" sx={{ mb: 4, lineHeight: 1.7 }}>
                                      Select a lecture note above and click "Generate Cards" to create AI-powered flashcards tailored to your content.
                                  </Typography>
                                  <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                                      {['Smart AI generation', 'Spaced repetition', 'Keyboard shortcuts'].map(f => (
                                          <Chip key={f} label={f} size="small" sx={{ fontWeight: 600 }} />
                                      ))}
                                  </Box>
                              </>
                          )}
                      </Box>
                  </Box>
              ) : sessionComplete ? (
                  /* Session Complete */
                  <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <Paper sx={{ p: 6, borderRadius: '24px', border: '1px solid', borderColor: 'divider', maxWidth: 480, width: '100%', textAlign: 'center' }}>
                          <Box sx={{ 
                              width: 96, height: 96, borderRadius: '50%', 
                              bgcolor: 'rgba(16,185,129,0.1)', border: '2px solid rgba(16,185,129,0.3)',
                              display: 'flex', alignItems: 'center', justifyContent: 'center',
                              mx: 'auto', mb: 4
                          }}>
                              <TrophyIcon sx={{ fontSize: 48, color: '#10B981' }} />
                          </Box>
                          <Typography variant="h4" fontWeight={800} gutterBottom>Session Complete!</Typography>
                          <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                              You've reviewed all {flashcards.length} flashcards.
                          </Typography>
                          <Grid container spacing={2} sx={{ mb: 4 }}>
                              {[
                                  { label: 'Easy', count: Object.values(ratings).filter(r => r === 'easy').length, color: '#10B981' },
                                  { label: 'Good', count: Object.values(ratings).filter(r => r === 'good').length, color: '#137fec' },
                                  { label: 'Hard', count: Object.values(ratings).filter(r => r === 'hard').length, color: '#F59E0B' },
                                  { label: 'Again', count: Object.values(ratings).filter(r => r === 'again').length, color: '#EF4444' },
                              ].map(s => (
                                  <Grid item xs={3} key={s.label}>
                                      <Box sx={{ textAlign: 'center' }}>
                                          <Typography variant="h3" fontWeight={800} sx={{ color: s.color }}>{s.count}</Typography>
                                          <Typography variant="caption" color="text.secondary">{s.label}</Typography>
                                      </Box>
                                  </Grid>
                              ))}
                          </Grid>
                          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
                              <Button variant="outlined" startIcon={<RestartIcon />} onClick={handleRestart} sx={{ fontWeight: 700 }}>
                                  Study Again
                              </Button>
                              <Button variant="contained" startIcon={<ShuffleIcon />} onClick={handleShuffle} sx={{ fontWeight: 700 }}>
                                  Shuffle & Retry
                              </Button>
                          </Box>
                      </Paper>
                  </Box>
              ) : (
                  /* Active Study Mode */
                  <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', maxWidth: 800, mx: 'auto', width: '100%' }}>
                      {/* Progress Bar */}
                      <Box sx={{ width: '100%', mb: 4 }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                              <Typography variant="body2" fontWeight={600} color="text.secondary">
                                  Card {currentIndex + 1} of {flashcards.length}
                              </Typography>
                              <Box sx={{ display: 'flex', gap: 1 }}>
                                  <Tooltip title="Shuffle">
                                      <IconButton size="small" onClick={handleShuffle} sx={{ color: 'text.secondary' }}>
                                          <ShuffleIcon fontSize="small" />
                                      </IconButton>
                                  </Tooltip>
                                  <Tooltip title="Restart">
                                      <IconButton size="small" onClick={handleRestart} sx={{ color: 'text.secondary' }}>
                                          <RestartIcon fontSize="small" />
                                      </IconButton>
                                  </Tooltip>
                              </Box>
                          </Box>
                          <LinearProgress 
                              variant="determinate" 
                              value={progress}
                              sx={{ height: 8, borderRadius: 4, bgcolor: 'action.hover', '& .MuiLinearProgress-bar': { borderRadius: 4 } }} 
                          />
                          <Box sx={{ display: 'flex', gap: 2, mt: 1 }}>
                              {['easy', 'good', 'hard', 'again'].map(r => {
                                  const count = Object.values(ratings).filter(v => v === r).length;
                                  if (count === 0) return null;
                                  const colors = { easy: '#10B981', good: '#137fec', hard: '#F59E0B', again: '#EF4444' };
                                  return (
                                      <Typography key={r} variant="caption" sx={{ color: colors[r], fontWeight: 700 }}>
                                          {r}: {count}
                                      </Typography>
                                  );
                              })}
                          </Box>
                      </Box>

                      {/* THE FLASHCARD */}
                      <Box sx={{ 
                          width: '100%', flex: 1, 
                          minHeight: 320,
                          maxHeight: 420,
                          perspective: '1200px', 
                          mb: 4,
                          position: 'relative'
                      }}>
                          <AnimatePresence initial={false} custom={direction} mode="wait">
                              <motion.div
                                  key={currentIndex}
                                  custom={direction}
                                  initial={{ x: direction > 0 ? 200 : -200, opacity: 0 }}
                                  animate={{ x: 0, opacity: 1, transition: { duration: 0.3, ease: 'easeOut' } }}
                                  exit={{ x: direction > 0 ? -200 : 200, opacity: 0, transition: { duration: 0.2 } }}
                                  style={{ position: 'absolute', inset: 0 }}
                              >
                                  <Box
                                      onClick={() => setIsFlipped(f => !f)}
                                      sx={{
                                          width: '100%', height: '100%',
                                          cursor: 'pointer',
                                          transformStyle: 'preserve-3d',
                                          transition: 'transform 0.55s cubic-bezier(0.4, 0.0, 0.2, 1)',
                                          transform: isFlipped ? 'rotateY(180deg)' : 'rotateY(0deg)',
                                          position: 'relative'
                                      }}
                                  >
                                      {/* Front */}
                                      <Paper elevation={0} sx={{
                                          position: 'absolute', inset: 0, backfaceVisibility: 'hidden',
                                          borderRadius: '20px', overflow: 'hidden',
                                          border: '1px solid', borderColor: currentRating ? ratingColors[currentRating]?.[2] : 'divider',
                                          display: 'flex', flexDirection: 'column',
                                          background: theme.palette.mode === 'dark' 
                                              ? 'linear-gradient(135deg, #1a212a 0%, #111822 100%)'
                                              : 'linear-gradient(135deg, #ffffff 0%, #f8faff 100%)',
                                      }}>
                                          {/* Card header */}
                                          <Box sx={{ px: 3, py: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid', borderColor: 'divider' }}>
                                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                  <Chip 
                                                      label="CONCEPT" 
                                                      size="small" 
                                                      sx={{ fontWeight: 800, fontSize: '0.6rem', height: 20, bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main', borderRadius: '4px' }} 
                                                  />
                                                  {currentRating && (
                                                      <Chip 
                                                          label={currentRating.toUpperCase()} 
                                                          size="small" 
                                                          sx={{ fontWeight: 800, fontSize: '0.6rem', height: 20, borderRadius: '4px',
                                                              bgcolor: ratingColors[currentRating][1],
                                                              color: ratingColors[currentRating][0]
                                                          }} 
                                                      />
                                                  )}
                                              </Box>
                                              <IconButton size="small" onClick={toggleBookmark} sx={{ color: bookmarked.has(currentIndex) ? 'warning.main' : 'text.disabled' }}>
                                                  {bookmarked.has(currentIndex) ? <BookmarkIcon fontSize="small" /> : <BookmarkBorderIcon fontSize="small" />}
                                              </IconButton>
                                          </Box>
                                          
                                          {/* Card content */}
                                          <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 4, textAlign: 'center', overflowY: 'auto' }}>
                                              <Typography 
                                                  variant="h5" 
                                                  fontWeight={700}
                                                  sx={{ lineHeight: 1.5, color: 'text.primary', wordBreak: 'break-word' }}
                                              >
                                                  {currentCard?.front}
                                              </Typography>
                                          </Box>

                                          {/* Footer hint */}
                                          <Box sx={{ p: 2, textAlign: 'center', borderTop: '1px solid', borderColor: 'divider' }}>
                                              <Typography variant="caption" color="text.disabled" sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                                                  Click or press <Chip label="Space" size="small" sx={{ height: 16, fontSize: '0.6rem', fontWeight: 700 }} /> to reveal answer
                                              </Typography>
                                          </Box>
                                      </Paper>

                                      {/* Back */}
                                      <Paper elevation={0} sx={{
                                          position: 'absolute', inset: 0, backfaceVisibility: 'hidden',
                                          transform: 'rotateY(180deg)',
                                          borderRadius: '20px', overflow: 'hidden',
                                          border: '1px solid', borderColor: 'rgba(19,127,236,0.3)',
                                          display: 'flex', flexDirection: 'column',
                                          background: theme.palette.mode === 'dark'
                                              ? 'linear-gradient(135deg, rgba(19,127,236,0.12) 0%, #111822 100%)'
                                              : 'linear-gradient(135deg, rgba(19,127,236,0.05) 0%, #ffffff 100%)',
                                      }}>
                                          <Box sx={{ px: 3, py: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid', borderColor: 'divider' }}>
                                              <Chip label="ANSWER" size="small" sx={{ fontWeight: 800, fontSize: '0.6rem', height: 20, bgcolor: 'rgba(19,127,236,0.15)', color: 'primary.main', borderRadius: '4px' }} />
                                              <IconButton size="small" onClick={toggleBookmark} sx={{ color: bookmarked.has(currentIndex) ? 'warning.main' : 'text.disabled' }}>
                                                  {bookmarked.has(currentIndex) ? <BookmarkIcon fontSize="small" /> : <BookmarkBorderIcon fontSize="small" />}
                                              </IconButton>
                                          </Box>
                                          <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: 4, textAlign: 'center', overflowY: 'auto' }}>
                                              <Typography variant="h5" fontWeight={800} color="primary.main" sx={{ lineHeight: 1.5, mb: 2, wordBreak: 'break-word' }}>
                                                  {currentCard?.back}
                                              </Typography>
                                          </Box>
                                          <Box sx={{ p: 2, textAlign: 'center', borderTop: '1px solid', borderColor: 'divider' }}>
                                              <Typography variant="caption" color="text.disabled">Rate your confidence below</Typography>
                                          </Box>
                                      </Paper>
                                  </Box>
                              </motion.div>
                          </AnimatePresence>
                      </Box>

                      {/* Controls */}
                      <Box sx={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 2 }}>
                          {/* SRS Buttons (visible after flipping) */}
                          <AnimatePresence>
                              {isFlipped && (
                                  <motion.div
                                      initial={{ opacity: 0, y: 20 }}
                                      animate={{ opacity: 1, y: 0 }}
                                      exit={{ opacity: 0, y: 10 }}
                                      transition={{ duration: 0.2 }}
                                  >
                                      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 1.5, mb: 2 }}>
                                          {[
                                              { key: 'again', label: 'Again', sublabel: '< 1m', color: 'error' },
                                              { key: 'hard', label: 'Hard', sublabel: '3 days', color: 'warning' },
                                              { key: 'good', label: 'Good', sublabel: '7 days', color: 'primary' },
                                              { key: 'easy', label: 'Easy', sublabel: '14 days', color: 'success' },
                                          ].map(btn => (
                                              <Button
                                                  key={btn.key}
                                                  variant={currentRating === btn.key ? 'contained' : 'outlined'}
                                                  color={btn.color}
                                                  onClick={() => handleRate(btn.key)}
                                                  sx={{ 
                                                      flexDirection: 'column', 
                                                      py: 1.5, 
                                                      borderRadius: '12px',
                                                      fontWeight: 700,
                                                      gap: 0.25
                                                  }}
                                              >
                                                  <Typography variant="caption" fontWeight={800} sx={{ textTransform: 'uppercase', fontSize: '0.7rem' }}>{btn.label}</Typography>
                                                  <Typography variant="caption" sx={{ fontSize: '0.65rem', opacity: 0.7 }}>{btn.sublabel}</Typography>
                                              </Button>
                                          ))}
                                      </Box>
                                  </motion.div>
                              )}
                          </AnimatePresence>

                          {/* Navigation Buttons */}
                          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'space-between', alignItems: 'center' }}>
                              <Button 
                                  variant="outlined" 
                                  startIcon={<KeyboardArrowLeft />} 
                                  onClick={handlePrev}
                                  disabled={currentIndex === 0}
                                  sx={{ fontWeight: 700, borderColor: 'divider', color: 'text.primary' }}
                              >
                                  Previous
                              </Button>
                              <Button 
                                  variant="text" 
                                  onClick={() => setIsFlipped(f => !f)}
                                  sx={{ fontWeight: 700, color: 'text.secondary' }}
                              >
                                  {isFlipped ? 'Hide Answer' : 'Show Answer'}
                              </Button>
                              <Button 
                                  variant="outlined" 
                                  endIcon={<KeyboardArrowRight />} 
                                  onClick={handleNext}
                                  sx={{ fontWeight: 700, borderColor: 'divider', color: 'text.primary' }}
                              >
                                  {currentIndex >= flashcards.length - 1 ? 'Finish' : 'Next'}
                              </Button>
                          </Box>
                      </Box>
                  </Box>
              )}
          </Box>
      </Box>
      )}

      {activeTab === 1 && (
          <Box sx={{ flex: 1, p: { xs: 2, md: 4 }, overflowY: 'auto' }}>
              <TableContainer component={Paper} elevation={0} sx={{ border: '1px solid', borderColor: 'divider', borderRadius: '16px' }}>
                  <Table>
                      <TableHead sx={{ bgcolor: 'action.hover' }}>
                          <TableRow>
                              <TableCell sx={{ fontWeight: 800 }}>Front (Concept)</TableCell>
                              <TableCell sx={{ fontWeight: 800 }}>Back (Answer)</TableCell>
                              <TableCell sx={{ fontWeight: 800 }}>Ease Factor</TableCell>
                              <TableCell sx={{ fontWeight: 800 }}>Interval (Days)</TableCell>
                          </TableRow>
                      </TableHead>
                      <TableBody>
                          {flashcards.length === 0 ? (
                              <TableRow>
                                  <TableCell colSpan={4} align="center" sx={{ py: 6, color: 'text.secondary' }}>No flashcards found. Generate some first!</TableCell>
                              </TableRow>
                          ) : (
                              flashcards.map(c => (
                                  <TableRow key={c.id}>
                                      <TableCell sx={{ maxWidth: 300 }}>{c.front}</TableCell>
                                      <TableCell sx={{ maxWidth: 300 }}>{c.back}</TableCell>
                                      <TableCell>{c.ease_factor || 2.5}</TableCell>
                                      <TableCell>{c.interval || 0}</TableCell>
                                  </TableRow>
                              ))
                          )}
                      </TableBody>
                  </Table>
              </TableContainer>
          </Box>
      )}
    </Box>
  );
}
