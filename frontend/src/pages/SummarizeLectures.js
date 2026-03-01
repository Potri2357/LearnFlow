import React, { useState, useEffect } from 'react';
import {
  Container, Typography, Box, Button,
  Select, MenuItem, FormControl, InputLabel, CircularProgress,
  Alert, Chip, Paper, Grid, Divider, useTheme, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  AutoStories as LectureIcon,
  Lightbulb as ConceptIcon,
  MenuBook as DefinitionIcon,
  AccountTree as RelationshipIcon,
  Timeline as FlowchartIcon,
  AutoAwesome as AutoAwesomeIcon,
  TrendingUp as TrendingUpIcon,
  Psychology as PsychologyIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon
} from '@mui/icons-material';
import mermaid from 'mermaid';
import API from '../api/api';

mermaid.initialize({
  startOnLoad: false,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'Inter, sans-serif',
  flowchart: { curve: 'basis' },
});

const MermaidDiagram = ({ chart }) => {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState(false);

  useEffect(() => {
    const renderChart = async () => {
      if (!chart) return;
      try {
        setError(false);
        const id = `mermaid-${Math.random().toString(36).substr(2, 9)}`;
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
      <Box sx={{ p: 3, border: '1px dashed', borderColor: 'warning.main', borderRadius: 2, bgcolor: 'rgba(245,158,11,0.05)' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1, color: 'warning.main' }}>
          <WarningIcon fontSize="small" />
          <Typography variant="body2" fontWeight={700}>Flowchart rendering error (syntax issue)</Typography>
        </Box>
        <Typography variant="caption" color="text.secondary" component="pre" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', overflowX: 'auto', display: 'block' }}>
          {chart}
        </Typography>
      </Box>
    );
  }

  if (!svg) return (
    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 4 }}>
      <CircularProgress size={32} />
    </Box>
  );

  return (
    <Box sx={{
      '& svg': { maxWidth: '100%', height: 'auto', display: 'block', mx: 'auto' },
      '& .node rect, & .node circle, & .node polygon, & .node path': { fill: 'transparent !important', stroke: 'currentColor' },
    }} dangerouslySetInnerHTML={{ __html: svg }} />
  );
};

export default function SummarizeLectures() {
  const theme = useTheme();
  const [lectures, setLectures] = useState([]);
  const [selectedLecture, setSelectedLecture] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null);
  const [lecturesLoading, setLecturesLoading] = useState(true);

  useEffect(() => {
    API.get('lectures/').then(r => {
      setLectures(r.data || []);
    }).catch(() => {
      setError('Failed to fetch lectures');
    }).finally(() => setLecturesLoading(false));
  }, []);

  const handleGenerate = async () => {
    if (!selectedLecture) { setError('Please select a lecture'); return; }
    setLoading(true);
    setError('');
    setSummary(null);
    try {
      const res = await API.post(`lectures/${selectedLecture}/summarize/`);
      setSummary(res.data.summary);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to generate summary. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const importanceMeta = {
    high: { color: 'error', label: 'High', dot: '#ef4444' },
    medium: { color: 'warning', label: 'Medium', dot: '#f59e0b' },
    low: { color: 'success', label: 'Low', dot: '#10b981' },
  };

  const selectedTitle = lectures.find(l => l.id === selectedLecture)?.title;

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', color: 'text.primary', pb: 8 }}>
      {/* === HEADER === */}
      <Box sx={{
          bgcolor: 'background.paper',
          borderBottom: '1px solid', borderColor: 'divider',
          px: { xs: 2, md: 4 }, py: 3
      }}>
          <Container maxWidth="xl">
              <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, justifyContent: 'space-between', alignItems: { md: 'center' }, gap: 2 }}>
                  <Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                          <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main', display: 'flex' }}>
                              <PsychologyIcon fontSize="small" />
                          </Box>
                          <Typography variant="h4" fontWeight={900} sx={{ letterSpacing: '-0.02em' }}>
                              Lecture Summarizer
                          </Typography>
                          {summary && <Chip label="Summary Ready" color="success" size="small" sx={{ fontWeight: 700 }} />}
                      </Box>
                      <Typography variant="body2" color="text.secondary" fontWeight={500}>
                          AI-powered summaries with key concepts, definitions, and visual flowcharts
                      </Typography>
                  </Box>

                  {/* Controls */}
                  <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
                      <FormControl sx={{ minWidth: 240 }} size="small">
                          <InputLabel>Select Lecture</InputLabel>
                          <Select
                              value={selectedLecture}
                              label="Select Lecture"
                              onChange={(e) => { setSelectedLecture(e.target.value); setSummary(null); }}
                          >
                              {lecturesLoading && <MenuItem disabled>Loading...</MenuItem>}
                              {lectures.map((l) => (
                                  <MenuItem key={l.id} value={l.id}>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                          <LectureIcon sx={{ fontSize: 16, color: 'primary.main' }} />
                                          {l.title}
                                      </Box>
                                  </MenuItem>
                              ))}
                          </Select>
                      </FormControl>
                      <Button
                          variant="contained"
                          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AutoAwesomeIcon />}
                          onClick={handleGenerate}
                          disabled={loading || !selectedLecture}
                          sx={{ fontWeight: 700, px: 3, boxShadow: '0 4px 14px 0 rgba(19, 127, 236, 0.4)', whiteSpace: 'nowrap' }}
                      >
                          {loading ? 'Summarizing...' : 'Generate Summary'}
                      </Button>
                  </Box>
              </Box>
          </Container>
      </Box>

      <Container maxWidth="xl" sx={{ mt: 4 }}>
          {error && <Alert severity="error" onClose={() => setError('')} sx={{ mb: 3, borderRadius: '12px' }}>{error}</Alert>}

          {/* Loading State */}
          {loading && (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', py: 12, gap: 3 }}>
                  <CircularProgress size={64} />
                  <Typography variant="h6" fontWeight={700}>Analysing lecture content...</Typography>
                  <Typography variant="body2" color="text.secondary">This may take 15-30 seconds while the AI processes your lecture.</Typography>
              </Box>
          )}

          {/* Empty State */}
          {!summary && !loading && (
              <Paper sx={{
                  p: { xs: 6, md: 10 }, textAlign: 'center',
                  borderRadius: '20px', border: '1px dashed', borderColor: 'divider',
                  bgcolor: 'background.paper'
              }}>
                  <Box sx={{ width: 96, height: 96, borderRadius: '20px', bgcolor: 'rgba(19,127,236,0.08)', border: '2px solid rgba(19,127,236,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', mx: 'auto', mb: 3 }}>
                      <AutoAwesomeIcon sx={{ fontSize: 48, color: 'primary.main' }} />
                  </Box>
                  <Typography variant="h4" fontWeight={800} gutterBottom>
                      AI-Powered Summaries
                  </Typography>
                  <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 480, mx: 'auto', lineHeight: 1.7 }}>
                      Select a lecture above and click "Generate Summary" to receive a structured breakdown with key concepts, definitions, relationships, and a visual flowchart.
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
                      {['Key Concepts', 'Definitions', 'Visual Flowchart', 'Concept Relationships'].map(f => (
                          <Chip key={f} label={f} size="small" sx={{ fontWeight: 600 }} icon={<CheckCircleIcon sx={{ fontSize: '14px !important' }} />} />
                      ))}
                  </Box>
              </Paper>
          )}

          {/* SUMMARY CONTENT */}
          {summary && !loading && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {/* Title Banner */}
                  <Paper sx={{ p: 4, borderRadius: '16px', border: '1px solid', borderColor: 'rgba(19,127,236,0.2)', bgcolor: 'rgba(19,127,236,0.05)', display: 'flex', alignItems: 'center', gap: 3 }}>
                      <Box sx={{ p: 1.5, borderRadius: '12px', bgcolor: 'rgba(19,127,236,0.15)', color: 'primary.main', display: 'flex' }}>
                          <LectureIcon fontSize="medium" />
                      </Box>
                      <Box sx={{ flex: 1 }}>
                          <Typography variant="overline" color="primary.main" fontWeight={700} sx={{ letterSpacing: '0.1em' }}>Summary Generated</Typography>
                          <Typography variant="h5" fontWeight={800}>{selectedTitle}</Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                              {summary.key_concepts?.length || 0} concepts • {summary.definitions?.length || 0} definitions
                          </Typography>
                      </Box>
                  </Paper>

                  {/* Overview */}
                  <Paper sx={{ p: 4, borderRadius: '16px', border: '1px solid', borderColor: 'divider' }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                          <LectureIcon color="primary" />
                          <Typography variant="h6" fontWeight={700}>Overview</Typography>
                      </Box>
                      <Typography variant="body1" sx={{ lineHeight: 1.9, color: 'text.primary', opacity: 0.9 }}>
                          {summary.overview}
                      </Typography>
                  </Paper>

                  {/* Key Concepts */}
                  {summary.key_concepts?.length > 0 && (
                      <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                              <ConceptIcon color="primary" />
                              <Typography variant="h5" fontWeight={700}>Key Concepts</Typography>
                              <Chip label={`${summary.key_concepts.length} concepts`} size="small" sx={{ fontWeight: 600 }} />
                          </Box>
                          <Grid container spacing={2}>
                              {summary.key_concepts.map((concept, idx) => {
                                  const meta = importanceMeta[concept.importance] || importanceMeta.medium;
                                  return (
                                      <Grid item xs={12} sm={6} xl={4} key={idx}>
                                          <Paper sx={{
                                              p: 3, height: '100%',
                                              borderRadius: '12px',
                                              border: '1px solid', borderColor: 'divider',
                                              transition: 'transform 0.2s, box-shadow 0.2s',
                                              '&:hover': { transform: 'translateY(-2px)', boxShadow: theme.shadows[4] }
                                          }}>
                                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                                                  <Typography variant="h6" fontWeight={700} sx={{ flex: 1, pr: 1 }}>{concept.name}</Typography>
                                                  <Chip
                                                      label={meta.label}
                                                      size="small"
                                                      color={meta.color}
                                                      variant="outlined"
                                                      sx={{ fontWeight: 700, flexShrink: 0, fontSize: '0.65rem' }}
                                                  />
                                              </Box>
                                              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                                                  {concept.description}
                                              </Typography>
                                          </Paper>
                                      </Grid>
                                  );
                              })}
                          </Grid>
                      </Box>
                  )}

                  {/* Definitions */}
                  {summary.definitions?.length > 0 && (
                      <Box>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                              <DefinitionIcon color="primary" />
                              <Typography variant="h5" fontWeight={700}>Important Definitions</Typography>
                              <Chip label={`${summary.definitions.length} terms`} size="small" sx={{ fontWeight: 600 }} />
                          </Box>
                          <Paper sx={{ borderRadius: '16px', border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
                              {summary.definitions.map((def, idx) => (
                                  <Accordion
                                      key={idx}
                                      elevation={0}
                                      disableGutters
                                      sx={{
                                          borderBottom: idx < summary.definitions.length - 1 ? '1px solid' : 'none',
                                          borderColor: 'divider',
                                          '&:before': { display: 'none' },
                                          bgcolor: 'background.paper'
                                      }}
                                  >
                                      <AccordionSummary
                                          expandIcon={<ExpandMoreIcon />}
                                          sx={{ px: 3, py: 2 }}
                                      >
                                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'primary.main', flexShrink: 0 }} />
                                              <Typography variant="subtitle1" fontWeight={700} color="primary.main">
                                                  {def.term}
                                              </Typography>
                                          </Box>
                                      </AccordionSummary>
                                      <AccordionDetails sx={{ px: 3, pb: 3 }}>
                                          <Divider sx={{ mb: 2 }} />
                                          <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.8 }}>
                                              {def.definition}
                                          </Typography>
                                      </AccordionDetails>
                                  </Accordion>
                              ))}
                          </Paper>
                      </Box>
                  )}

                  {/* Concept Relationships */}
                  {summary.relationships && (
                      <Paper sx={{ p: 4, borderRadius: '16px', border: '1px solid', borderColor: 'divider' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                              <RelationshipIcon color="primary" />
                              <Typography variant="h6" fontWeight={700}>Concept Relationships</Typography>
                          </Box>
                          <Typography variant="body1" sx={{ lineHeight: 1.9, color: 'text.primary', opacity: 0.9 }}>
                              {summary.relationships}
                          </Typography>
                      </Paper>
                  )}

                  {/* Flowchart */}
                  {summary.flowchart && (
                      <Paper sx={{ borderRadius: '16px', border: '1px solid', borderColor: 'divider', overflow: 'hidden' }}>
                          <Box sx={{ px: 4, py: 3, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', gap: 1.5 }}>
                              <FlowchartIcon color="primary" />
                              <Typography variant="h6" fontWeight={700}>Concept Flowchart</Typography>
                              <Chip label="Visual" size="small" sx={{ fontWeight: 600, ml: 'auto' }} color="primary" variant="outlined" />
                          </Box>
                          <Box sx={{
                              p: 4,
                              minHeight: 300,
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)',
                          }}>
                              <Box sx={{ width: '100%', overflowX: 'auto', display: 'flex', justifyContent: 'center' }}>
                                  <MermaidDiagram chart={summary.flowchart} />
                              </Box>
                          </Box>
                      </Paper>
                  )}
              </Box>
          )}
      </Container>
    </Box>
  );
}
