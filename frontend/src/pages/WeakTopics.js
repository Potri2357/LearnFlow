import React, { useState, useEffect, useCallback } from "react";
import API from "../api/api";
import {
  Button,
  Typography,
  Box,
  Chip,
  Stack,
  LinearProgress,
  Popover,
  Paper,
  CircularProgress,
  Alert,
} from "@mui/material";
import {
  AutoAwesome as AutoAwesomeIcon,
  PlayArrow as PlayArrowIcon,
  Psychology as PsychologyIcon,
  Warning as WarningIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
} from "@mui/icons-material";
import { useNavigate, useParams } from "react-router-dom";
import {
  ActionBar,
  AsyncContent,
  PageHeader,
  SurfaceCard,
} from "../components/ui";

import { SUBJECT_COLORS } from "../theme";
const getSubjectColor = (subject, allSubjects) => {
  const idx = allSubjects.indexOf(subject);
  return SUBJECT_COLORS[idx % SUBJECT_COLORS.length];
};

// Chip color based on accuracy/severity
const getSeverityColor = (pct) => {
  if (pct < 40) return { bg: 'rgba(239,68,68,0.12)', color: '#EF4444', label: 'Critical' };
  if (pct < 70) return { bg: 'rgba(245,158,11,0.12)', color: '#F59E0B', label: 'Needs Work' };
  return { bg: 'rgba(16,185,129,0.12)', color: '#10B981', label: 'Fair' };
};

// Topic chip with hover popover
const TopicChip = ({ topic, subject, accuracy, noteId, onNavigate }) => {
  const [anchor, setAnchor] = useState(null);
  const severity = getSeverityColor(accuracy);

  const handleExplain = () => {
    setAnchor(null);
    onNavigate('explain', topic, subject, noteId);
  };
  const handlePractice = () => {
    setAnchor(null);
    onNavigate('practice', topic, subject, noteId);
  };

  return (
    <>
      <Chip
        label={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span style={{ fontWeight: 700 }}>{topic}</span>
            <span style={{ opacity: 0.7, fontSize: '0.75em' }}>{accuracy}%</span>
          </Box>
        }
        onClick={(e) => setAnchor(e.currentTarget)}
        size="medium"
        sx={{
          bgcolor: severity.bg,
          color: severity.color,
          border: `1.5px solid ${severity.color}40`,
          fontWeight: 600,
          cursor: 'pointer',
          height: 32,
          transition: 'all 0.15s ease',
          '&:hover': {
            bgcolor: severity.bg,
            borderColor: severity.color,
            transform: 'translateY(-1px)',
            boxShadow: `0 4px 12px ${severity.color}30`,
          },
          '& .MuiChip-label': { px: 1.5 },
        }}
      />
      <Popover
        open={Boolean(anchor)}
        anchorEl={anchor}
        onClose={() => setAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        PaperProps={{ sx: { borderRadius: '16px', border: '1px solid', borderColor: 'divider', boxShadow: '0 12px 40px rgba(0,0,0,0.15)', p: 0, overflow: 'hidden', minWidth: 260 } }}
      >
        <Box sx={{ p: 2.5 }}>
          <Typography variant="subtitle2" fontWeight={800} gutterBottom>{topic}</Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 1.5 }}>
            <Chip label={severity.label} size="small"
              sx={{ bgcolor: severity.bg, color: severity.color, fontWeight: 700, fontSize: '0.7rem', height: 20 }} />
            <Typography variant="caption" color="text.secondary">{accuracy}% accuracy</Typography>
          </Box>
          <Box sx={{ mb: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">Mastery level</Typography>
              <Typography variant="caption" fontWeight={700} sx={{ color: severity.color }}>{accuracy}%</Typography>
            </Box>
            <LinearProgress variant="determinate" value={accuracy}
              sx={{ height: 5, borderRadius: 3, bgcolor: `${severity.color}18`,
                '& .MuiLinearProgress-bar': { bgcolor: severity.color, borderRadius: 3 } }} />
          </Box>
        </Box>
        <Box sx={{ px: 2.5, pb: 2, display: 'flex', gap: 1 }}>
          <Button variant="contained" size="small" startIcon={<PlayArrowIcon />}
            onClick={handlePractice}
            sx={{ flex: 1, fontWeight: 700, borderRadius: '8px', fontSize: '0.8rem',
              background: 'linear-gradient(135deg, #2563EB 0%, #4F46E5 100%)' }}>
            Practice
          </Button>
          <Button variant="outlined" size="small" startIcon={<AutoAwesomeIcon />}
            onClick={handleExplain}
            sx={{ flex: 1, fontWeight: 700, borderRadius: '8px', fontSize: '0.8rem',
              borderColor: '#7C3AED', color: '#7C3AED',
              '&:hover': { bgcolor: 'rgba(124,58,237,0.08)', borderColor: '#7C3AED' } }}>
            Explain
          </Button>
        </Box>
      </Popover>
    </>
  );
};

function WeakTopics() {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const { noteId } = useParams();
  const navigate = useNavigate();

  const fetchWeakTopics = useCallback(() => {
    setLoading(true);
    setError("");
    API.get(`weak-topics/?note_id=${noteId}`)
      .then((res) => setTopics(res.data.weak_topics || []))
      .catch((err) => {
        console.error("Error fetching weak topics:", err);
        setTopics([]);
        setError("We could not load weak topics right now.");
      })
      .finally(() => setLoading(false));
  }, [noteId]);

  useEffect(() => { fetchWeakTopics(); }, [fetchWeakTopics]);

  const handleNavigate = (action, topic, subject, nId) => {
    if (action === 'explain') {
      navigate(
        `/concept-coach?topic=${encodeURIComponent(topic)}&subject=${encodeURIComponent(subject || '')}&autoExplain=true`
      );
    } else {
      navigate(`/quiz/${nId || noteId}`);
    }
  };

  // Map topics: enrich with accuracy %, sort by severity
  const enrichedTopics = [...topics]
    .map((t) => ({
      ...t,
      score: Number(t.score || 0),
      accuracy: Math.max(0, Math.min(100, Math.round((1 - Number(t.score || 0)) * 100))),
      subject: t.subject || 'General',
    }))
    .sort((a, b) => a.accuracy - b.accuracy); // worst first

  // Group by subject
  const subjectGroups = enrichedTopics.reduce((acc, t) => {
    if (!acc[t.subject]) acc[t.subject] = [];
    acc[t.subject].push(t);
    return acc;
  }, {});

  const allSubjects = Object.keys(subjectGroups);

  // Aggregate stats per subject
  const subjectStats = allSubjects.map((subj) => {
    const group = subjectGroups[subj];
    const avgAcc = Math.round(group.reduce((s, t) => s + t.accuracy, 0) / group.length);
    return { subject: subj, avgAcc, count: group.length };
  });

  return (
    <Box sx={{ maxWidth: 1120, mx: "auto", display: "grid", gap: 2.5 }}>
      <PageHeader
        title={`Weak Topics · Lecture ${noteId}`}
        subtitle="Prioritize weak areas — click any topic chip to practice or get an instant explanation."
      />

      <SurfaceCard>
        <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ xs: 'flex-start', sm: 'center' }}
          justifyContent="space-between" spacing={1.5}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
              <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#EF4444' }} />
              <Typography variant="caption" color="text.secondary" fontWeight={600}>Critical (&lt;40%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
              <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#F59E0B' }} />
              <Typography variant="caption" color="text.secondary" fontWeight={600}>Needs Work (40–70%)</Typography>
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
              <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#10B981' }} />
              <Typography variant="caption" color="text.secondary" fontWeight={600}>Fair (&gt;70%)</Typography>
            </Box>
          </Box>
          <Button size="small" variant="outlined" onClick={fetchWeakTopics} startIcon={<RefreshIcon />}
            sx={{ borderRadius: 2.5, fontWeight: 700, borderColor: 'divider', color: 'text.secondary',
              '&:hover': { borderColor: 'primary.main', color: 'primary.main' } }}>
            Refresh
          </Button>
        </Stack>
      </SurfaceCard>

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
          <CircularProgress />
        </Box>
      )}

      {error && !loading && (
        <Alert severity="error" action={<Button size="small" onClick={fetchWeakTopics}>Retry</Button>}>
          {error}
        </Alert>
      )}

      {!loading && !error && enrichedTopics.length === 0 && (
        <Paper sx={{ p: 6, textAlign: 'center', borderRadius: '20px', border: '1px dashed', borderColor: 'divider' }}>
          <TrendingDownIcon sx={{ fontSize: 56, color: 'text.disabled', mb: 2, opacity: 0.4 }} />
          <Typography variant="h6" fontWeight={700} gutterBottom>No weak topics yet</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
            Complete a quiz for this lecture to surface weak concepts and recommendations.
          </Typography>
          <Button variant="contained" onClick={() => navigate(`/quiz/${noteId}`)} startIcon={<PlayArrowIcon />}
            sx={{ fontWeight: 700, borderRadius: 2 }}>
            Go to Quiz
          </Button>
        </Paper>
      )}

      {/* Subject-grouped inline chip sections */}
      {!loading && !error && allSubjects.length > 0 && (
        <Stack spacing={3}>
          {allSubjects.map((subject) => {
            const color = getSubjectColor(subject, allSubjects);
            const stat = subjectStats.find(s => s.subject === subject);
            const sevColor = getSeverityColor(stat.avgAcc);
            return (
              <Paper key={subject} elevation={0} sx={{
                p: 2.5, borderRadius: '16px',
                border: '1px solid', borderColor: 'divider',
                borderLeft: `4px solid ${color}`,
              }}>
                {/* Subject header row */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2, flexWrap: 'wrap' }}>
                  <Typography variant="subtitle1" fontWeight={800} sx={{ color }}>
                    {subject}
                  </Typography>
                  <Chip label={`${stat.count} topic${stat.count > 1 ? 's' : ''}`} size="small"
                    sx={{ bgcolor: `${color}15`, color, fontWeight: 700, fontSize: '0.7rem', height: 20 }} />
                  <Chip label={`Avg: ${stat.avgAcc}%`} size="small"
                    sx={{ bgcolor: sevColor.bg, color: sevColor.color, fontWeight: 700, fontSize: '0.7rem', height: 20 }} />
                  <Box sx={{ ml: 'auto', display: 'flex', gap: 1 }}>
                    <Button size="small" variant="outlined" onClick={() => navigate(`/quiz/${noteId}`)}
                      startIcon={<PlayArrowIcon sx={{ fontSize: '14px !important' }} />}
                      sx={{ fontWeight: 700, fontSize: '0.75rem', borderRadius: 2, py: 0.5,
                        borderColor: `${color}50`, color, '&:hover': { borderColor: color, bgcolor: `${color}10` } }}>
                      Practice All
                    </Button>
                  </Box>
                </Box>

                {/* Inline topic chips */}
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {subjectGroups[subject].map((t, i) => (
                    <TopicChip
                      key={`${t.topic}-${i}`}
                      topic={t.topic || `Topic ${i + 1}`}
                      subject={subject}
                      accuracy={t.accuracy}
                      noteId={t.note_id || noteId}
                      onNavigate={handleNavigate}
                    />
                  ))}
                </Box>

                {/* Bottom: mini accuracy bar for worst topic */}
                {subjectGroups[subject][0] && (
                  <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'divider' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="caption" color="text.secondary" fontWeight={600}>
                        Most critical: <strong>{subjectGroups[subject][0].topic}</strong>
                      </Typography>
                      <Typography variant="caption" fontWeight={700}
                        sx={{ color: getSeverityColor(subjectGroups[subject][0].accuracy).color }}>
                        {subjectGroups[subject][0].accuracy}%
                      </Typography>
                    </Box>
                    <LinearProgress variant="determinate" value={subjectGroups[subject][0].accuracy}
                      sx={{ height: 4, borderRadius: 2,
                        bgcolor: `${getSeverityColor(subjectGroups[subject][0].accuracy).color}18`,
                        '& .MuiLinearProgress-bar': {
                          bgcolor: getSeverityColor(subjectGroups[subject][0].accuracy).color,
                          borderRadius: 2,
                        } }} />
                  </Box>
                )}
              </Paper>
            );
          })}
        </Stack>
      )}
    </Box>
  );
}

export default WeakTopics;
