import React, { useState, useEffect, useMemo } from 'react';
import {
  Container, Typography, Box, Paper, Chip, TextField,
  InputAdornment, Accordion, AccordionSummary, AccordionDetails,
  Button, Grid, useTheme, Divider, Select, MenuItem, FormControl, InputLabel
} from '@mui/material';
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  CheckCircle as CheckCircleIcon,
  Cancel as CancelIcon,
  LocalFireDepartment as FireIcon,
  School as SchoolIcon,
  MenuBook as BookIcon,
  Quiz as QuizIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import API from '../api/api';

export default function QuestionBank() {
  const theme = useTheme();
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [subjectFilter, setSubjectFilter] = useState('All');
  const [bloomsFilter, setBloomsFilter] = useState('All');
  const [highYieldOnly, setHighYieldOnly] = useState(false);
  
  // Quick attempt state: track selected options and correct/wrong status per question
  const [attempts, setAttempts] = useState({});

  useEffect(() => {
    API.get('questions/all/')
      .then(res => {
        setQuestions(res.data.questions || []);
      })
      .catch(err => console.error("Failed to fetch question bank:", err))
      .finally(() => setLoading(false));
  }, []);

  const subjects = useMemo(() => {
    const subs = new Set(questions.map(q => q.subject || 'General'));
    return ['All', ...Array.from(subs)];
  }, [questions]);

  const filteredQuestions = useMemo(() => {
    return questions.filter(q => {
      const matchSubject = subjectFilter === 'All' || (q.subject || 'General') === subjectFilter;
      const matchBlooms = bloomsFilter === 'All' || q.blooms_level === bloomsFilter;
      const matchHighYield = highYieldOnly ? q.is_high_yield : true;
      const matchSearch = (q.question_text || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                          (q.topic || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                          (q.lecture_title || '').toLowerCase().includes(searchQuery.toLowerCase());
      return matchSubject && matchBlooms && matchHighYield && matchSearch;
    });
  }, [questions, searchQuery, subjectFilter, bloomsFilter, highYieldOnly]);

  const groupedQuestions = useMemo(() => {
    const groups = {};
    filteredQuestions.forEach(q => {
      const sub = q.subject || 'General';
      if (!groups[sub]) groups[sub] = [];
      groups[sub].push(q);
    });
    return groups;
  }, [filteredQuestions]);

  const handleAttempt = (qId, optionLetter, correctOption) => {
    setAttempts(prev => ({
      ...prev,
      [qId]: {
        selected: optionLetter,
        isCorrect: optionLetter === (correctOption || 'A').toUpperCase()
      }
    }));
  };

  const handleExport = () => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(filteredQuestions, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute("href", dataStr);
    dlAnchorElem.setAttribute("download", "question_bank_export.json");
    dlAnchorElem.click();
  };

  if (loading) {
    return <Box sx={{ p: 4, textAlign: 'center' }}><Typography>Loading Question Bank...</Typography></Box>;
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 8 }}>
      <Box sx={{ mb: 5, textAlign: 'center' }}>
        <Typography variant="h3" fontWeight={900} gutterBottom
          sx={{ background: 'linear-gradient(135deg, #137fec 0%, #10b981 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}
        >
          Question Bank
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Browse, search, and quick-attempt all generated questions from your library.
        </Typography>
      </Box>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 4, borderRadius: '16px', display: 'flex', gap: 2, flexWrap: 'wrap', alignItems: 'center' }}>
        <TextField
          placeholder="Search questions, topics..."
          variant="outlined"
          size="small"
          fullWidth
          sx={{ flex: 1, minWidth: 250, '& .MuiOutlinedInput-root': { borderRadius: '12px' } }}
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: <InputAdornment position="start"><SearchIcon /></InputAdornment>
          }}
        />
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Subject</InputLabel>
          <Select
            value={subjectFilter}
            label="Subject"
            onChange={e => setSubjectFilter(e.target.value)}
            sx={{ borderRadius: '12px' }}
          >
            {subjects.map(s => <MenuItem key={s} value={s}>{s}</MenuItem>)}
          </Select>
        </FormControl>
        
        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel>Bloom's Level</InputLabel>
          <Select
            value={bloomsFilter}
            label="Bloom's Level"
            onChange={e => setBloomsFilter(e.target.value)}
            sx={{ borderRadius: '12px' }}
          >
            <MenuItem value="All">All Levels</MenuItem>
            <MenuItem value="remember">Remember</MenuItem>
            <MenuItem value="understand">Understand</MenuItem>
            <MenuItem value="apply">Apply</MenuItem>
            <MenuItem value="analyze">Analyze</MenuItem>
            <MenuItem value="evaluate">Evaluate</MenuItem>
            <MenuItem value="create">Create</MenuItem>
          </Select>
        </FormControl>

        <Chip 
          icon={<FireIcon sx={{ fontSize: 16 }} />} 
          label="High Yield Only" 
          onClick={() => setHighYieldOnly(!highYieldOnly)}
          color={highYieldOnly ? "error" : "default"}
          variant={highYieldOnly ? "filled" : "outlined"}
          sx={{ fontWeight: 700, borderRadius: '12px', cursor: 'pointer' }} 
        />

        <Chip label={`${filteredQuestions.length} Questions`} color="primary" sx={{ fontWeight: 700, ml: 'auto' }} />
        <Button variant="contained" startIcon={<DownloadIcon />} onClick={handleExport} sx={{ borderRadius: '12px', fontWeight: 700 }}>Export</Button>
      </Paper>

      {/* Grouped by Subject */}
      {Object.entries(groupedQuestions).map(([subject, qs]) => (
        <Box key={subject} sx={{ mb: 5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2, borderBottom: '2px solid', borderColor: 'divider', pb: 1 }}>
            <SchoolIcon color="primary" />
            <Typography variant="h5" fontWeight={800}>{subject}</Typography>
            <Chip size="small" label={qs.length} sx={{ fontWeight: 700, bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main' }} />
          </Box>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {qs.map((q, idx) => {
              const attempt = attempts[q.id];
              return (
                <Accordion key={q.id} disableGutters sx={{ borderRadius: '12px !important', border: '1px solid', borderColor: 'divider', '&:before': { display: 'none' }, boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ '& .MuiAccordionSummary-content': { my: 1.5 } }}>
                    <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-start', width: '100%' }}>
                      <Box sx={{ width: 32, height: 32, borderRadius: '8px', bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, flexShrink: 0 }}>
                        {idx + 1}
                      </Box>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography variant="body1" fontWeight={600} sx={{ mb: 1, pr: 2 }}>{q.question_text}</Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                          {q.is_high_yield && (
                            <Chip size="small" icon={<FireIcon sx={{ fontSize: 14 }}/>} label="High Yield" sx={{ bgcolor: 'rgba(245,158,11,0.1)', color: '#f59e0b', fontWeight: 800, border: '1px solid rgba(245,158,11,0.2)' }} />
                          )}
                          {q.blooms_level && (
                            <Chip size="small" label={q.blooms_level.toUpperCase()} sx={{ fontWeight: 600, fontSize: '0.65rem', bgcolor: 'rgba(124, 58, 237, 0.1)', color: '#7c3aed' }} />
                          )}
                          {q.question_type && (
                            <Chip size="small" label={q.question_type.replace('_', ' ').toUpperCase()} variant="outlined" sx={{ fontWeight: 600, fontSize: '0.65rem' }} />
                          )}
                          {q.topic && <Chip size="small" label={q.topic} variant="outlined" sx={{ fontWeight: 600, fontSize: '0.7rem' }} />}
                          <Typography variant="caption" color="text.disabled" sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 0.5 }}><BookIcon sx={{ fontSize: 14 }}/> {q.lecture_title}</Typography>
                        </Box>
                      </Box>
                      {/* Attempt Status Badge */}
                      {attempt && (
                        <Box sx={{ display: 'flex', alignItems: 'center', mr: 2, color: attempt.isCorrect ? 'success.main' : 'error.main' }}>
                          {attempt.isCorrect ? <CheckCircleIcon /> : <CancelIcon />}
                        </Box>
                      )}
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails sx={{ bgcolor: theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)', p: 3, borderTop: '1px solid', borderColor: 'divider' }}>
                    
                    <Typography variant="subtitle2" fontWeight={700} color="text.secondary" sx={{ mb: 1.5 }}>Quick Attempt</Typography>
                    
                    <Grid container spacing={2}>
                      {['A', 'B', 'C', 'D'].map(opt => {
                        const optText = q[`option_${opt.toLowerCase()}`];
                        if (!optText) return null;
                        
                        let optStyle = { borderColor: 'divider', bgcolor: 'background.paper' };
                        let icon = null;
                        
                        if (attempt) {
                           const isThisCorrect = (q.correct_option || 'A').toUpperCase() === opt;
                           const isThisSelected = attempt.selected === opt;
                           
                           if (isThisCorrect) {
                             optStyle = { borderColor: 'success.main', bgcolor: 'rgba(16,185,129,0.1)', color: 'success.dark' };
                             icon = <CheckCircleIcon color="success" fontSize="small" />;
                           } else if (isThisSelected) {
                             optStyle = { borderColor: 'error.main', bgcolor: 'rgba(239,68,68,0.1)', color: 'error.dark' };
                             icon = <CancelIcon color="error" fontSize="small" />;
                           } else {
                             optStyle = { opacity: 0.6 };
                           }
                        }

                        return (
                          <Grid item xs={12} sm={6} key={opt}>
                            <Paper
                              elevation={0}
                              onClick={() => !attempt && handleAttempt(q.id, opt, q.correct_option)}
                              sx={{
                                p: 2, borderRadius: '10px', border: '2px solid', ...optStyle,
                                cursor: attempt ? 'default' : 'pointer',
                                transition: 'all 0.2s',
                                '&:hover': attempt ? {} : { borderColor: 'primary.main', bgcolor: 'rgba(19,127,236,0.04)' }
                              }}
                            >
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Box sx={{ width: 24, height: 24, borderRadius: '6px', bgcolor: 'action.hover', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 800, fontSize: '0.8rem' }}>
                                  {opt}
                                </Box>
                                <Typography variant="body2" fontWeight={600} sx={{ flex: 1 }}>{optText}</Typography>
                                {icon}
                              </Box>
                            </Paper>
                          </Grid>
                        );
                      })}
                    </Grid>
                    
                    {attempt && q.explanation && (
                      <Box sx={{ mt: 3, p: 2, borderRadius: '8px', bgcolor: 'info.main' + '15', borderLeft: '4px solid', borderColor: 'info.main' }}>
                        <Typography variant="subtitle2" fontWeight={700} color="info.main" sx={{ mb: 0.5 }}>Explanation</Typography>
                        <Typography variant="body2">{q.explanation}</Typography>
                      </Box>
                    )}
                    
                  </AccordionDetails>
                </Accordion>
              );
            })}
          </Box>
        </Box>
      ))}
      
      {filteredQuestions.length === 0 && (
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <QuizIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">No questions found</Typography>
          <Typography variant="body2" color="text.disabled">Try adjusting your search or generate new questions.</Typography>
        </Box>
      )}
    </Container>
  );
}
