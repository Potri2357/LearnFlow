import React, { useState, useEffect } from 'react';
import {
  Container, Typography, Box, Card, CardContent, Button,
  Select, MenuItem, FormControl, InputLabel, CircularProgress,
  Alert, Accordion, AccordionSummary, AccordionDetails, Chip,
  Paper, Stack, Divider, Grid
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  AutoStories as LectureIcon,
  Lightbulb as ConceptIcon,
  MenuBook as DefinitionIcon,
  AccountTree as RelationshipIcon,
  Timeline as FlowchartIcon
} from '@mui/icons-material';
import API from '../api/api';
import mermaid from 'mermaid';
import VideoGenerator from '../components/VideoGenerator';

// Initialize Mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'default',
  securityLevel: 'loose',
  fontFamily: 'Inter, sans-serif',
});

// Safe Mermaid Component
const MermaidDiagram = ({ chart }) => {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(false);

  useEffect(() => {
    const renderChart = async () => {
      if (!chart) return;
      
      try {
        setError(false);
        // Generate a unique ID for this render
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
        
        // Attempt to render
        const { svg } = await mermaid.render(id, chart);
        setSvg(svg);
      } catch (err) {
        console.error('Mermaid rendering error:', err);
        setError(true);
      }
    };

    renderChart();
  }, [chart]);

  if (error) {
    return (
      <Box sx={{ p: 2, border: '1px dashed #ff9800', borderRadius: 2, bgcolor: '#fff3e0' }}>
        <Typography variant="body2" color="warning.main" gutterBottom fontWeight="bold">
          ⚠️ Flowchart could not be rendered (Syntax Error)
        </Typography>
        <Typography variant="caption" color="text.secondary" component="div" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap' }}>
          {chart}
        </Typography>
      </Box>
    );
  }

  return <div dangerouslySetInnerHTML={{ __html: svg }} />;
};

export default function SummarizeLectures() {
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null);
  const [flowchartKey, setFlowchartKey] = useState(0);

  useEffect(() => {
    fetchLectures();
  }, []);

  useEffect(() => {
    if (summary?.flowchart) {
      // Re-render Mermaid flowchart when summary changes
      setTimeout(() => {
        mermaid.contentLoaded();
      }, 100);
    }
  }, [summary, flowchartKey]);

  const fetchLectures = async () => {
    try {
      const response = await API.get('lectures/');
      setLectures(response.data);
    } catch (err) {
      setError('Failed to fetch lectures');
      console.error(err);
    }
  };

  const handleGenerateSummary = async () => {
    if (!selectedLecture) {
      setError('Please select a lecture');
      return;
    }

    setLoading(true);
    setError('');
    setSummary(null);

    try {
      const response = await API.post(`lectures/${selectedLecture}/summarize/`);
      setSummary(response.data.summary);
      setFlowchartKey(prev => prev + 1);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate summary');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getImportanceColor = (importance) => {
    switch (importance) {
      case 'high':
        return { bg: '#fef2f2', color: '#dc2626', border: '#fca5a5' };
      case 'medium':
        return { bg: '#fffbeb', color: '#d97706', border: '#fcd34d' };
      case 'low':
        return { bg: '#f0fdf4', color: '#16a34a', border: '#86efac' };
      default:
        return { bg: '#f8fafc', color: '#64748b', border: '#cbd5e0' };
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 6 }}>
      {/* Header */}
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
            mb: 2,
          }}
        >
          📖 Lecture Summarizer
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 600, mx: 'auto' }}>
          Get AI-powered summaries of your lectures with visual flowcharts to enhance understanding
        </Typography>
      </Box>

      {/* Lecture Selection */}
      <Paper
        elevation={0}
        sx={{
          p: 4,
          mb: 4,
          borderRadius: 4,
          background: 'linear-gradient(135deg, #667eea15 0%, #764ba215 100%)',
          border: '1px solid rgba(102, 126, 234, 0.2)',
        }}
      >
        <Stack spacing={3}>
          <FormControl fullWidth>
            <InputLabel>Select a Lecture</InputLabel>
            <Select
              value={selectedLecture}
              onChange={(e) => setSelectedLecture(e.target.value)}
              label="Select a Lecture"
              sx={{
                bgcolor: 'white',
                borderRadius: 2,
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: 'rgba(102, 126, 234, 0.3)',
                },
              }}
            >
              {lectures.map((lecture) => (
                <MenuItem key={lecture.id} value={lecture.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LectureIcon sx={{ fontSize: 20, color: '#667eea' }} />
                    {lecture.title}
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            variant="contained"
            size="large"
            onClick={handleGenerateSummary}
            disabled={loading || !selectedLecture}
            sx={{
              py: 1.5,
              borderRadius: 3,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              textTransform: 'none',
              fontSize: '16px',
              fontWeight: 600,
              boxShadow: '0 4px 20px rgba(102, 126, 234, 0.4)',
              '&:hover': {
                background: 'linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%)',
                boxShadow: '0 6px 25px rgba(102, 126, 234, 0.5)',
              },
            }}
          >
            {loading ? <CircularProgress size={24} sx={{ color: 'white' }} /> : 'Generate Summary'}
          </Button>
        </Stack>
      </Paper>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 4, borderRadius: 3 }}>
          {error}
        </Alert>
      )}

      {/* Summary Display */}
      {summary && (
        <Stack spacing={4}>
          {/* Overview Section */}
          <Card
            elevation={0}
            sx={{
              borderRadius: 4,
              border: '1px solid #e2e8f0',
              overflow: 'hidden',
              transition: 'all 0.3s',
              '&:hover': {
                boxShadow: '0 8px 30px rgba(102, 126, 234, 0.15)',
              },
            }}
          >
            <Box
              sx={{
                p: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <LectureIcon /> Overview
              </Typography>
            </Box>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="body1" sx={{ lineHeight: 1.8, color: '#2d3748', mb: 3 }}>
                {summary.overview}
              </Typography>
              
              <Divider sx={{ mb: 3 }} />
              <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700, color: '#64748b' }}>
                  AI Video Summary
              </Typography>
              <VideoGenerator text={summary.overview} />
            </CardContent>
          </Card>

          {/* Key Concepts Section */}
          <Box>
            <Typography
              variant="h5"
              sx={{
                mb: 3,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                color: '#2d3748',
              }}
            >
              <ConceptIcon sx={{ color: '#667eea' }} /> Key Concepts
            </Typography>
            <Grid container spacing={3}>
              {summary.key_concepts?.map((concept, index) => {
                const colors = getImportanceColor(concept.importance);
                return (
                  <Grid item xs={12} md={6} key={index}>
                    <Card
                      elevation={0}
                      sx={{
                        height: '100%',
                        borderRadius: 3,
                        border: `2px solid ${colors.border}`,
                        bgcolor: colors.bg,
                        transition: 'all 0.3s',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: `0 8px 20px ${colors.border}80`,
                        },
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
                          <Typography variant="h6" sx={{ fontWeight: 600, color: colors.color }}>
                            {concept.name}
                          </Typography>
                          <Chip
                            label={concept.importance}
                            size="small"
                            sx={{
                              bgcolor: colors.color,
                              color: 'white',
                              fontWeight: 600,
                              textTransform: 'uppercase',
                              fontSize: '10px',
                            }}
                          />
                        </Box>
                        <Typography variant="body2" sx={{ color: '#4a5568', lineHeight: 1.7 }}>
                          {concept.description}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </Box>

          {/* Definitions Section */}
          <Box>
            <Typography
              variant="h5"
              sx={{
                mb: 3,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                color: '#2d3748',
              }}
            >
              <DefinitionIcon sx={{ color: '#667eea' }} /> Important Definitions
            </Typography>
            <Stack spacing={2}>
              {summary.definitions?.map((def, index) => (
                <Accordion
                  key={index}
                  elevation={0}
                  sx={{
                    borderRadius: 3,
                    border: '1px solid #e2e8f0',
                    '&:before': { display: 'none' },
                    '&:hover': {
                      borderColor: '#667eea',
                    },
                  }}
                >
                  <AccordionSummary
                    expandIcon={<ExpandMoreIcon />}
                    sx={{
                      px: 3,
                      '& .MuiAccordionSummary-content': {
                        my: 2,
                      },
                    }}
                  >
                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#667eea' }}>
                      {def.term}
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails sx={{ px: 3, pb: 3 }}>
                    <Divider sx={{ mb: 2 }} />
                    <Typography variant="body1" sx={{ color: '#4a5568', lineHeight: 1.8 }}>
                      {def.definition}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Stack>
          </Box>

          {/* Relationships Section */}
          <Card
            elevation={0}
            sx={{
              borderRadius: 4,
              border: '1px solid #e2e8f0',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                p: 2,
                background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                color: 'white',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <RelationshipIcon /> Concept Relationships
              </Typography>
            </Box>
            <CardContent sx={{ p: 3 }}>
              <Typography variant="body1" sx={{ lineHeight: 1.8, color: '#2d3748' }}>
                {summary.relationships}
              </Typography>
            </CardContent>
          </Card>

          {/* Flowchart Section */}
          <Card
            elevation={0}
            sx={{
              borderRadius: 4,
              border: '1px solid #e2e8f0',
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                p: 2,
                background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                color: 'white',
              }}
            >
              <Typography variant="h6" sx={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 1 }}>
                <FlowchartIcon /> Concept Flowchart
              </Typography>
            </Box>
            <CardContent sx={{ p: 4, bgcolor: '#f8fafc' }}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  minHeight: 300,
                  bgcolor: 'white',
                  borderRadius: 3,
                  p: 3,
                  border: '1px solid #e2e8f0',
                }}
              >
                <Box sx={{ width: '100%', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
                  <MermaidDiagram chart={summary.flowchart} />
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Stack>
      )}

      {/* Empty State */}
      {!summary && !loading && !error && (
        <Paper
          elevation={0}
          sx={{
            p: 8,
            textAlign: 'center',
            bgcolor: '#f8fafc',
            borderRadius: 4,
            border: '2px dashed #cbd5e0',
          }}
        >
          <FlowchartIcon sx={{ fontSize: 80, color: '#cbd5e0', mb: 2 }} />
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No summary generated yet
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Select a lecture and click "Generate Summary" to get started
          </Typography>
        </Paper>
      )}
    </Container>
  );
}
