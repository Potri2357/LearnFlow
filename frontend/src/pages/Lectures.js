import React, { useState, useEffect } from 'react';
import { 
  Container, Typography, Card, CardContent, IconButton, 
  List, ListItem, ListItemText, ListItemSecondaryAction,
  Collapse, Box, Chip, CircularProgress, Alert, Button,
  Dialog, DialogActions, DialogContent, DialogContentText, DialogTitle,
  Stack, Paper
} from '@mui/material';
import { 
  Delete as DeleteIcon, 
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  Description as NoteIcon,
  Help as QuestionIcon,
  CalendarToday as CalendarIcon,
  Fingerprint as IdIcon
} from '@mui/icons-material';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import API from '../api/api';

export default function Lectures() {
  const { user } = useAuth();
  const [lectures, setLectures] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expandedId, setExpandedId] = useState(null);
  const [lectureDetails, setLectureDetails] = useState({});
  const [loadingDetails, setLoadingDetails] = useState({});
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [lectureToDelete, setLectureToDelete] = useState(null);

  useEffect(() => {
    fetchLectures();
  }, []);

  const fetchLectures = async () => {
    try {
      const response = await API.get('lectures/');
      setLectures(response.data);
    } catch (err) {
      setError('Failed to fetch lectures');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleExpandClick = async (id) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }

    setExpandedId(id);

    // Fetch details if not already loaded
    if (!lectureDetails[id]) {
      setLoadingDetails(prev => ({ ...prev, [id]: true }));
      try {
        const response = await API.get(`lectures/${id}/`);
        setLectureDetails(prev => ({ ...prev, [id]: response.data }));
      } catch (err) {
        console.error('Failed to fetch lecture details', err);
      } finally {
        setLoadingDetails(prev => ({ ...prev, [id]: false }));
      }
    }
  };

  const handleDeleteClick = (lecture) => {
    setLectureToDelete(lecture);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!lectureToDelete) return;

    try {
      await API.delete(`lectures/${lectureToDelete.id}/`);
      
      setLectures(lectures.filter(l => l.id !== lectureToDelete.id));
      setDeleteDialogOpen(false);
      setLectureToDelete(null);
      
      // If the deleted lecture was expanded, clear it
      if (expandedId === lectureToDelete.id) {
        setExpandedId(null);
      }
    } catch (err) {
      console.error('Failed to delete lecture', err);
      alert('Failed to delete lecture');
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
      </Box>
    );
  }

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
        📚 My Lectures
      </Typography>

      {error && <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>{error}</Alert>}

      {lectures.length === 0 ? (
        <Paper 
          elevation={0}
          sx={{ 
            p: 6, 
            textAlign: 'center', 
            bgcolor: '#f8fafc',
            borderRadius: 4,
            border: '2px dashed #cbd5e0'
          }}
        >
          <NoteIcon sx={{ fontSize: 60, color: '#cbd5e0', mb: 2 }} />
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No lectures found
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 3 }}>
            Upload your first lecture note to get started with AI-powered learning!
          </Typography>
          <Button 
            variant="contained" 
            href="/upload"
          >
            Upload Note
          </Button>
        </Paper>
      ) : (
        <Stack spacing={2}>
          {lectures.map((lecture) => (
            <Card 
              key={lecture.id} 
              elevation={0}
              sx={{ 
                overflow: 'visible',
                transition: 'all 0.3s ease',
                borderRadius: 3,
                border: '1px solid rgba(0,0,0,0.05)',
                background: 'white',
                boxShadow: '0 4px 20px rgba(0,0,0,0.02)',
                '&:hover': { 
                  transform: 'translateY(-4px)',
                  boxShadow: '0 12px 30px rgba(102, 126, 234, 0.15)',
                  borderColor: 'rgba(102, 126, 234, 0.3)'
                }
              }}
            >
              <Box 
                sx={{ 
                  p: 2, 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'space-between',
                  cursor: 'pointer'
                }}
                onClick={() => handleExpandClick(lecture.id)}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1 }}>
                  <Box 
                    sx={{ 
                      width: 56,
                      height: 56,
                      borderRadius: 3,
                      background: 'linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)',
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      color: 'white',
                      boxShadow: '0 4px 12px rgba(142, 197, 252, 0.4)'
                    }}
                  >
                    <NoteIcon fontSize="medium" />
                  </Box>
                  
                  <Box>
                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#2d3748', lineHeight: 1.2, mb: 0.5 }}>
                      {lecture.title}
                    </Typography>
                    <Stack direction="row" spacing={2} alignItems="center">
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <IdIcon sx={{ fontSize: 14, color: '#a0aec0' }} />
                        <Typography variant="caption" sx={{ color: '#718096', fontWeight: 500 }}>
                          ID: {lecture.id}
                        </Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <CalendarIcon sx={{ fontSize: 14, color: '#a0aec0' }} />
                        <Typography variant="caption" sx={{ color: '#718096', fontWeight: 500 }}>
                          {new Date(lecture.created_at).toLocaleDateString()}
                        </Typography>
                      </Box>
                    </Stack>
                  </Box>
                </Box>

                <Stack direction="row" spacing={1} alignItems="center">
                  <IconButton 
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteClick(lecture);
                    }}
                    sx={{ 
                      color: '#cbd5e0',
                      '&:hover': { color: '#ef4444', bgcolor: '#fee2e2' }
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                  <IconButton 
                    sx={{ 
                      color: '#667eea',
                      transform: expandedId === lecture.id ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.3s'
                    }}
                  >
                    <ExpandMoreIcon />
                  </IconButton>
                </Stack>
              </Box>

              <Collapse in={expandedId === lecture.id} timeout="auto" unmountOnExit>
                <Box sx={{ p: 3, bgcolor: '#f8fafc', borderTop: '1px solid #edf2f7' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2, color: '#4a5568', display: 'flex', alignItems: 'center', gap: 1 }}>
                    <QuestionIcon sx={{ fontSize: 18, color: '#667eea' }} />
                    Generated Questions
                  </Typography>

                  {loadingDetails[lecture.id] ? (
                    <Box display="flex" justifyContent="center" p={2}>
                      <CircularProgress size={24} />
                    </Box>
                  ) : (
                    <Box>
                      {lectureDetails[lecture.id]?.questions && lectureDetails[lecture.id].questions.length > 0 ? (
                        <Stack spacing={2}>
                          {lectureDetails[lecture.id].questions.map((q, index) => (
                            <Paper 
                              key={q.id} 
                              elevation={0}
                              sx={{ 
                                p: 2, 
                                bgcolor: 'white', 
                                borderRadius: 2,
                                border: '1px solid #e2e8f0'
                              }}
                            >
                              <Typography variant="body2" fontWeight="600" color="#2d3748" gutterBottom>
                                {index + 1}. {q.question_text}
                              </Typography>
                              <Box sx={{ mt: 1.5, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                                <Chip 
                                  label={`Answer: ${q.correct_option}`} 
                                  size="small" 
                                  sx={{ bgcolor: '#def7ec', color: '#03543f', fontWeight: 600, height: 24 }}
                                />
                                <Chip 
                                  label={q.topic || 'General'} 
                                  size="small" 
                                  sx={{ bgcolor: '#ebf4ff', color: '#4299e1', fontWeight: 600, height: 24 }}
                                />
                                <Chip 
                                  label={q.difficulty < 0.4 ? 'Easy' : q.difficulty > 0.7 ? 'Hard' : 'Medium'} 
                                  size="small" 
                                  sx={{ 
                                    bgcolor: q.difficulty < 0.4 ? '#f0fdf4' : q.difficulty > 0.7 ? '#fef2f2' : '#fffbeb', 
                                    color: q.difficulty < 0.4 ? '#16a34a' : q.difficulty > 0.7 ? '#dc2626' : '#d97706',
                                    fontWeight: 600,
                                    height: 24
                                  }}
                                />
                              </Box>
                            </Paper>
                          ))}
                        </Stack>
                      ) : (
                        <Typography variant="body2" color="textSecondary" sx={{ fontStyle: 'italic', textAlign: 'center', py: 2 }}>
                          No questions generated for this lecture yet.
                        </Typography>
                      )}
                    </Box>
                  )}
                </Box>
              </Collapse>
            </Card>
          ))}
        </Stack>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
        PaperProps={{ sx: { borderRadius: 3, p: 1 } }}
      >
        <DialogTitle sx={{ fontWeight: 'bold' }}>Delete Lecture?</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete <strong>"{lectureToDelete?.title}"</strong>? 
            <br /><br />
            This will permanently remove the lecture note and all associated questions, flashcards, and progress.
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button onClick={() => setDeleteDialogOpen(false)} color="inherit" sx={{ fontWeight: 600 }}>Cancel</Button>
          <Button 
            onClick={confirmDelete} 
            variant="contained" 
            color="error" 
            sx={{ px: 3 }}
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
