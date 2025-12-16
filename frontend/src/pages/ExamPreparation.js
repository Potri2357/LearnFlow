import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Container, Typography, Box, Card, CardContent, Button,
  TextField, Stepper, Step, StepLabel, CircularProgress,
  Alert, Grid, Switch, FormControlLabel, IconButton,
  
  Chip, Divider, Paper, Stack, Dialog, DialogTitle,
  DialogContent, DialogActions, Timeline, TimelineItem, TimelineSeparator,
  TimelineConnector, TimelineContent, TimelineDot, TimelineOppositeContent,
  Table, TableBody, TableCell, TableContainer, TableHead, TableRow
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  AutoAwesome as AIIcon,
  School as ExamIcon,
  Assignment as QuestionIcon,
  EventNote as StrategyIcon,
  Warning as WarningIcon,
  CheckCircle as CheckIcon,
  Schedule as ScheduleIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import API from '../api/api';

const steps = ['Upload Syllabus', 'Previous Papers', 'Configuration', 'Generated Exam', 'Strategy Plan'];



const TimelineSchedule = ({ strategy }) => {
  return (
    <Box sx={{ width: '100%' }}>
      {strategy.map((dayPlan, i) => (
        <Box key={i} sx={{ mb: 6 }}>
          <Typography variant="h5" sx={{ mb: 2, color: '#764ba2', fontWeight: 'bold', borderLeft: '6px solid #764ba2', pl: 2 }}>
            Day {dayPlan.day}: {dayPlan.focus}
          </Typography>
          
          <TableContainer component={Paper} elevation={3} sx={{ borderRadius: 2, overflow: 'hidden' }}>
            <Table sx={{ minWidth: 650 }} aria-label="study schedule table">
              <TableHead>
                <TableRow sx={{ bgcolor: '#764ba2' }}>
                  <TableCell align="center" width="33%" sx={{ color: 'white', fontWeight: 'bold', fontSize: '1.1rem', py: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                      <ScheduleIcon fontSize="small" /> DURATION
                    </Box>
                  </TableCell>
                  <TableCell align="center" width="33%" sx={{ color: 'white', fontWeight: 'bold', fontSize: '1.1rem', py: 2, borderLeft: '1px solid rgba(255,255,255,0.2)' }}>
                    UNIT / TOPIC
                  </TableCell>
                  <TableCell align="center" width="33%" sx={{ color: 'white', fontWeight: 'bold', fontSize: '1.1rem', py: 2, borderLeft: '1px solid rgba(255,255,255,0.2)' }}>
                    SUBTOPICS
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {dayPlan.tasks?.map((task, j) => (
                  <TableRow
                    key={j}
                    sx={{ 
                      '&:last-child td, &:last-child th': { border: 0 },
                      bgcolor: j % 2 === 0 ? '#ffffff' : '#f8f9fa',
                      '&:hover': { bgcolor: '#f3e5f5' },
                      transition: 'background-color 0.2s'
                    }}
                  >
                    {/* Duration */}
                    <TableCell align="center" sx={{ borderRight: '1px solid #e0e0e0', py: 3 }}>
                      <Typography variant="h6" fontWeight="bold" color="primary">
                        {task.duration}
                      </Typography>
                    </TableCell>

                    {/* Main Topic */}
                    <TableCell align="center" sx={{ borderRight: '1px solid #e0e0e0', py: 3 }}>
                      <Typography variant="subtitle1" fontWeight="bold" sx={{ color: '#4a148c' }}>
                        {task.main_topic}
                      </Typography>
                    </TableCell>

                    {/* Subtopics */}
                    <TableCell align="left" sx={{ py: 3, px: 4 }}>
                      {Array.isArray(task.subtopics) && task.subtopics.length > 0 ? (
                        <ul style={{ margin: 0, paddingLeft: 20, color: '#616161' }}>
                          {task.subtopics.map((sub, k) => (
                            <li key={k} style={{ marginBottom: '4px' }}>
                              <Typography variant="body2">{sub}</Typography>
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <Typography variant="body2" color="textSecondary" align="center">
                          {task.subtopics}
                        </Typography>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      ))}
    </Box>
  );
};

export default function ExamPreparation() {
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Data States
  const [syllabusId, setSyllabusId] = useState(null);
  const [syllabusText, setSyllabusText] = useState('');
  const [syllabusFile, setSyllabusFile] = useState(null);
  const [syllabusTitle, setSyllabusTitle] = useState('');
  
  const [previousPapers, setPreviousPapers] = useState([]);
  
  const [config, setConfig] = useState({
    total_marks: 100,
    num_questions: 11,
    secure_centum_mode: false,
    mark_distribution: { "6": 5, "12": 5, "10": 1 } // Default distribution
  });
  
  const [questions, setQuestions] = useState([]);
  const [patternAnalysis, setPatternAnalysis] = useState('');
  
  // Strategy State
  const [strategyConfig, setStrategyConfig] = useState({
    days_remaining: 5,
    hours_per_day: 4
  });
  const [strategy, setStrategy] = useState(null);
  const [strategyLoading, setStrategyLoading] = useState(false);

  // Edit Dialog State
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingQuestion, setEditingQuestion] = useState(null);

  // --- Step 1: Syllabus Upload ---
  const onSyllabusDrop = useCallback(acceptedFiles => {
    if (acceptedFiles?.length) {
      setSyllabusFile(acceptedFiles[0]);
      if (!syllabusTitle) setSyllabusTitle(acceptedFiles[0].name.replace('.pdf', ''));
    }
  }, [syllabusTitle]);

  const { getRootProps: getSyllabusRootProps, getInputProps: getSyllabusInputProps } = useDropzone({
    onDrop: onSyllabusDrop,
    accept: { 'application/pdf': ['.pdf'] },
    maxFiles: 1
  });

  const handleSyllabusSubmit = async () => {
    if (!syllabusTitle) {
      setError('Please enter a title for the syllabus');
      return;
    }
    if (!syllabusText && !syllabusFile) {
      setError('Please provide syllabus content or upload a PDF');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const formData = new FormData();
      formData.append('title', syllabusTitle);
      if (syllabusText) formData.append('content', syllabusText);
      if (syllabusFile) formData.append('file', syllabusFile);

      const response = await API.post('exam/syllabus/upload/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setSyllabusId(response.data.id);
      setActiveStep(1);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload syllabus');
    } finally {
      setLoading(false);
    }
  };

  // --- Step 2: Previous Papers ---
  const onPapersDrop = useCallback(acceptedFiles => {
    setPreviousPapers(prev => [...prev, ...acceptedFiles]);
  }, []);

  const { getRootProps: getPapersRootProps, getInputProps: getPapersInputProps } = useDropzone({
    onDrop: onPapersDrop,
    accept: { 'application/pdf': ['.pdf'] }
  });

  const handlePapersSubmit = async () => {
    if (previousPapers.length === 0) {
      // Skip if no papers
      setActiveStep(2);
      return;
    }

    setLoading(true);
    setError('');

    try {
      const formData = new FormData();
      previousPapers.forEach(file => {
        formData.append('files', file);
      });

      await API.post(`exam/syllabus/${syllabusId}/papers/`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      setActiveStep(2);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload previous papers');
    } finally {
      setLoading(false);
    }
  };

  // --- Step 3: Strategy Planning ---
  const handleStrategyGenerate = async () => {
    setStrategyLoading(true);
    setError('');
    
    try {
      const response = await API.post(`exam/syllabus/${syllabusId}/strategy/`, strategyConfig);
      setStrategy(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate strategy');
    } finally {
      setStrategyLoading(false);
    }
  };

  // --- Step 4: Configuration ---
  const handleConfigChange = (field, value) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handleDistributionChange = (marks, count) => {
    setConfig(prev => ({
      ...prev,
      mark_distribution: {
        ...prev.mark_distribution,
        [marks]: parseInt(count) || 0
      }
    }));
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await API.post(`exam/syllabus/${syllabusId}/generate/`, config);
      setQuestions(response.data.questions);
      setPatternAnalysis(response.data.pattern_analysis);
      setActiveStep(3);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate exam');
    } finally {
      setLoading(false);
    }
  };

  // --- Step 4: Questions Management ---
  const handleEditClick = (question) => {
    setEditingQuestion({ ...question });
    setEditDialogOpen(true);
  };

  const handleSaveEdit = async () => {
    try {
      await API.put(`exam/question/${editingQuestion.id}/update/`, editingQuestion);
      setQuestions(prev => prev.map(q => q.id === editingQuestion.id ? editingQuestion : q));
      setEditDialogOpen(false);
    } catch (err) {
      setError('Failed to update question');
    }
  };

  const handleDeleteQuestion = async (id) => {
    if (!window.confirm('Are you sure you want to delete this question?')) return;
    try {
      await API.delete(`exam/question/${id}/delete/`);
      setQuestions(prev => prev.filter(q => q.id !== id));
    } catch (err) {
      setError('Failed to delete question');
    }
  };

  const handleExport = () => {
    const content = questions.map((q, i) => 
      `Q${i+1}. ${q.question_text} (${q.marks} Marks)\nPriority: ${q.priority}\nAnswer:\n${q.answer}\n\n`
    ).join('----------------------------------------\n\n');
    
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Exam_Questions_${syllabusTitle || 'Generated'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // --- Render Helpers ---
  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Stack spacing={3}>
            <TextField
              label="Syllabus Title"
              fullWidth
              value={syllabusTitle}
              onChange={(e) => setSyllabusTitle(e.target.value)}
              placeholder="e.g., Mathematics Final Exam"
            />
            
            <Box
              {...getSyllabusRootProps()}
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: '#fafafa',
                '&:hover': { bgcolor: '#f0f0f0' }
              }}
            >
              <input {...getSyllabusInputProps()} />
              <UploadIcon sx={{ fontSize: 48, color: '#667eea', mb: 2 }} />
              <Typography>
                {syllabusFile ? syllabusFile.name : "Drag & drop syllabus PDF here, or click to select"}
              </Typography>
            </Box>

            <Divider>OR</Divider>

            <TextField
              label="Paste Syllabus Text"
              multiline
              rows={6}
              fullWidth
              value={syllabusText}
              onChange={(e) => setSyllabusText(e.target.value)}
              placeholder="Paste your syllabus content here..."
            />

            <Button
              variant="contained"
              size="large"
              onClick={handleSyllabusSubmit}
              disabled={loading}
              sx={{ alignSelf: 'flex-end', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Next: Previous Papers'}
            </Button>
          </Stack>
        );

      case 1:
        return (
          <Stack spacing={3}>
            <Typography color="textSecondary">
              Upload previous year question papers to help AI analyze patterns and important topics. (Optional)
            </Typography>

            <Box
              {...getPapersRootProps()}
              sx={{
                border: '2px dashed #ccc',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: '#fafafa',
                '&:hover': { bgcolor: '#f0f0f0' }
              }}
            >
              <input {...getPapersInputProps()} />
              <UploadIcon sx={{ fontSize: 48, color: '#667eea', mb: 2 }} />
              <Typography>Drag & drop PDF files here, or click to select multiple files</Typography>
            </Box>

            {previousPapers.length > 0 && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>Selected Files:</Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                  {previousPapers.map((file, index) => (
                    <Chip
                      key={index}
                      icon={<FileIcon />}
                      label={file.name}
                      onDelete={() => setPreviousPapers(prev => prev.filter((_, i) => i !== index))}
                    />
                  ))}
                </Stack>
              </Box>
            )}

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button onClick={() => setActiveStep(0)}>Back</Button>
              <Button
                variant="contained"
                onClick={handlePapersSubmit}
                disabled={loading}
                sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
              >
                {loading ? <CircularProgress size={24} color="inherit" /> : 'Next: Configuration'}
              </Button>
            </Box>
          </Stack>
        );

      case 2:
        return (
          <Stack spacing={3}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Total Marks"
                  type="number"
                  fullWidth
                  value={config.total_marks}
                  onChange={(e) => handleConfigChange('total_marks', e.target.value)}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Number of Questions"
                  type="number"
                  fullWidth
                  value={config.num_questions}
                  onChange={(e) => handleConfigChange('num_questions', e.target.value)}
                />
              </Grid>
            </Grid>

            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6" gutterBottom>Mark Distribution</Typography>
                <Grid container spacing={2} alignItems="center">
                  {Object.entries(config.mark_distribution).map(([marks, count]) => (
                    <React.Fragment key={marks}>
                      <Grid item xs={4}>
                        <Typography>{marks} Marks Questions:</Typography>
                      </Grid>
                      <Grid item xs={8}>
                        <TextField
                          type="number"
                          size="small"
                          value={count}
                          onChange={(e) => handleDistributionChange(marks, e.target.value)}
                        />
                      </Grid>
                    </React.Fragment>
                  ))}
                  <Grid item xs={12}>
                    <Button size="small" onClick={() => {
                      const newMarks = prompt("Enter marks value (e.g., 15):");
                      if (newMarks) handleDistributionChange(newMarks, 0);
                    }}>
                      + Add Mark Category
                    </Button>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            <FormControlLabel
              control={
                <Switch
                  checked={config.secure_centum_mode}
                  onChange={(e) => handleConfigChange('secure_centum_mode', e.target.checked)}
                  color="primary"
                />
              }
              label={
                <Box>
                  <Typography variant="subtitle1" component="span" sx={{ fontWeight: 'bold', color: '#667eea' }}>
                    Secure Centum Mode 💯
                  </Typography>
                  <Typography variant="caption" display="block" color="textSecondary">
                    Generates comprehensive, creative questions covering all topics for maximum scoring potential.
                  </Typography>
                </Box>
              }
            />

            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button onClick={() => setActiveStep(1)}>Back</Button>
              <Button
                variant="contained"
                size="large"
                onClick={handleGenerate}
                disabled={loading}
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AIIcon />}
                sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
              >
                {loading ? 'Generating Exam...' : 'Generate Exam'}
              </Button>
            </Box>
          </Stack>
        );

      case 3:
        return (
          <Stack spacing={3}>
            {patternAnalysis && (
              <Alert severity="info" icon={<AIIcon />}>
                <Typography variant="subtitle2" fontWeight="bold">AI Pattern Analysis:</Typography>
                <Box sx={{ '& p': { m: 0 }, '& ul': { pl: 2 } }}>
                  <ReactMarkdown>{patternAnalysis}</ReactMarkdown>
                </Box>
              </Alert>
            )}

            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h5" fontWeight="bold">Generated Questions</Typography>
              <Box>
                <Button 
                  variant="outlined" 
                  startIcon={<StrategyIcon />}
                  onClick={() => setActiveStep(4)}
                  sx={{ mr: 2 }}
                >
                  Create Study Strategy
                </Button>
                <Button 
                  variant="outlined" 
                  startIcon={<SaveIcon />}
                  onClick={handleExport}
                >
                  Export Exam
                </Button>
              </Box>
            </Box>

            {questions.map((q, index) => (
              <Card key={q.id} sx={{ position: 'relative', overflow: 'visible' }}>
                <Box
                  sx={{
                    position: 'absolute',
                    top: 10,
                    right: 10,
                    display: 'flex',
                    gap: 1
                  }}
                >
                  <Chip
                    label={`Priority: ${q.priority}`}
                    color={q.priority <= 3 ? "error" : "default"}
                    size="small"
                  />
                  <Chip
                    label={`${q.marks} Marks`}
                    color="primary"
                    size="small"
                  />
                  <IconButton size="small" onClick={() => handleEditClick(q)}>
                    <EditIcon fontSize="small" />
                  </IconButton>
                  <IconButton size="small" color="error" onClick={() => handleDeleteQuestion(q.id)}>
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </Box>

                <CardContent>
                  <Typography variant="subtitle1" fontWeight="bold" sx={{ pr: 15 }}>
                    Q{index + 1}. {q.question_text}
                  </Typography>
                  
                  <Divider sx={{ my: 1.5 }} />
                  
                  <Box sx={{ color: 'text.secondary' }}>
                    <Typography variant="body2" component="div" sx={{ fontWeight: 'bold', mb: 0.5 }}>
                      Answer:
                    </Typography>
                    <ReactMarkdown>{q.answer}</ReactMarkdown>
                  </Box>
                  
                  {q.topic && (
                    <Chip
                      label={q.topic}
                      size="small"
                      variant="outlined"
                      sx={{ mt: 2 }}
                    />
                  )}
                </CardContent>
              </Card>
            ))}
            
            <Button onClick={() => setActiveStep(2)} sx={{ alignSelf: 'flex-start' }}>
              Back to Configuration
            </Button>
          </Stack>
        );

      case 4:
        return (
          <Stack spacing={3}>
            <Typography variant="h6" gutterBottom>
              <StrategyIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
              Exam Strategy & Schedule
            </Typography>
            
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Days Remaining for Exam"
                  type="number"
                  fullWidth
                  value={strategyConfig.days_remaining}
                  onChange={(e) => setStrategyConfig({ ...strategyConfig, days_remaining: e.target.value })}
                  helperText="Enter 0 for today only"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Hours Available per Day"
                  type="number"
                  fullWidth
                  value={strategyConfig.hours_per_day}
                  onChange={(e) => setStrategyConfig({ ...strategyConfig, hours_per_day: e.target.value })}
                />
              </Grid>
            </Grid>
            
            <Button
              variant="contained"
              onClick={handleStrategyGenerate}
              disabled={strategyLoading}
              startIcon={strategyLoading ? <CircularProgress size={20} color="inherit" /> : <AIIcon />}
              sx={{ alignSelf: 'flex-start', background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}
            >
              {strategyLoading ? 'Generating Strategy...' : 'Generate Strategy Plan'}
            </Button>
            
            {strategy && (
              <Box sx={{ mt: 3 }}>
                {!strategy.is_relevant && (
                  <Alert severity="warning" icon={<WarningIcon />} sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" fontWeight="bold">Relevance Warning:</Typography>
                    {strategy.relevance_warning}
                  </Alert>
                )}
                
                <Typography variant="h6" gutterBottom>Prioritized Topics:</Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 3 }}>
                  {strategy.prioritized_topics?.map((topic, i) => (
                    <Chip key={i} label={topic} color="primary" variant="outlined" sx={{ mb: 1 }} />
                  ))}
                </Stack>
                
                <Typography variant="h6" gutterBottom>Detailed Schedule:</Typography>
                {strategy.strategy && <TimelineSchedule strategy={strategy.strategy} />}
              </Box>
            )}

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
              <Button onClick={() => setActiveStep(3)}>Back to Exam</Button>
            </Box>
          </Stack>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth={false} sx={{ mt: 4, mb: 8, px: { xs: 2, md: 4, lg: 8 } }}>
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography
          variant="h3"
          gutterBottom
          sx={{
            fontWeight: 'bold',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          <ExamIcon sx={{ fontSize: 40, mr: 2, verticalAlign: 'bottom', color: '#667eea' }} />
          Exam Preparation
        </Typography>
        <Typography color="textSecondary">
          AI-powered exam generation with syllabus analysis and pattern recognition
        </Typography>
      </Box>

      <Stepper activeStep={activeStep} alternativeLabel sx={{ mb: 5 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Paper elevation={0} sx={{ p: 4, borderRadius: 4, border: '1px solid #e0e0e0' }}>
        {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
        {renderStepContent(activeStep)}
      </Paper>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>Edit Question</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 1 }}>
            <TextField
              label="Question Text"
              multiline
              rows={3}
              fullWidth
              value={editingQuestion?.question_text || ''}
              onChange={(e) => setEditingQuestion({ ...editingQuestion, question_text: e.target.value })}
            />
            <TextField
              label="Answer"
              multiline
              rows={4}
              fullWidth
              value={editingQuestion?.answer || ''}
              onChange={(e) => setEditingQuestion({ ...editingQuestion, answer: e.target.value })}
            />
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  label="Marks"
                  type="number"
                  fullWidth
                  value={editingQuestion?.marks || ''}
                  onChange={(e) => setEditingQuestion({ ...editingQuestion, marks: e.target.value })}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  label="Priority"
                  type="number"
                  fullWidth
                  value={editingQuestion?.priority || ''}
                  onChange={(e) => setEditingQuestion({ ...editingQuestion, priority: e.target.value })}
                />
              </Grid>
            </Grid>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveEdit}>Save Changes</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}
