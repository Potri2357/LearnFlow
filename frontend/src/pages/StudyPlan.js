// src/pages/StudyPlan.js
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box, Container, Typography, Button, Grid, Paper, TextField,
  Slider, Switch, Chip, LinearProgress, useTheme,
  CircularProgress, Alert, Checkbox
} from '@mui/material';
import {
  School as SchoolIcon,
  Tune as TuneIcon,
  Add as AddIcon,
  Close as CloseIcon,
  AutoAwesome as AutoAwesomeIcon,
  TrendingUp as TrendingUpIcon,
  CalendarMonth as CalendarMonthIcon,
  TrendingDown as TrendingDownIcon,
  CheckCircle as CheckCircleIcon,
  PlayCircle as PlayCircleIcon,
  Article as ArticleIcon,
  LocalFireDepartment as FireIcon,
  Warning as WarningIcon,
  Psychology as PsychologyIcon,
  EmojiEvents as TrophyIcon,
  AccessTime as TimeIcon,
  QuizOutlined as QuizIcon
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import API from '../api/api';
import LectureSelect from '../components/LectureSelect';

const parseBulletPoints = (text) => {
    if (!text) return [];
    return text.split('\n')
        .map(line => line.trim())
        .filter(line => line && (line.startsWith('-') || line.startsWith('•') || line.match(/^\d+\./)))
        .map(line => line.replace(/^[-•\d.]+\s*/, '').trim())
        .filter(Boolean);
};

// ===================== GENERATOR =====================
const StudyPlanGenerator = ({ noteId, setNoteId, onGenerate, loading }) => {
    const theme = useTheme();
    const [examDate, setExamDate] = useState('');
    const [hours, setHours] = useState(2);
    const [priorityInput, setPriorityInput] = useState('');
    const [priorities, setPriorities] = useState([]);
    const [focusWeak, setFocusWeak] = useState(true);

    const handleAddPriority = () => {
        const trimmed = priorityInput.trim();
        if (trimmed && !priorities.includes(trimmed)) {
            setPriorities([...priorities, trimmed]);
            setPriorityInput('');
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') { e.preventDefault(); handleAddPriority(); }
    };

    return (
        <Box sx={{ minHeight: '100vh', background: theme.palette.mode === 'dark' ? 'linear-gradient(135deg, #101922 0%, #1a2533 100%)' : 'linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%)' }}>
            {/* Header */}
            <Box sx={{ bgcolor: 'background.paper', borderBottom: '1px solid', borderColor: 'divider', px: { xs: 2, md: 4 }, py: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                    <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'rgba(19,127,236,0.1)', color: 'primary.main', display: 'flex' }}>
                        <PsychologyIcon fontSize="small" />
                    </Box>
                    <Typography variant="h4" fontWeight={900} sx={{ letterSpacing: '-0.02em' }}>Mastery Schedule Planner</Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" fontWeight={500}>
                    AI-powered study plan personalized to your strengths and weak areas
                </Typography>
            </Box>

            <Container maxWidth="lg" sx={{ py: 6 }}>
                <Grid container spacing={4} alignItems="stretch">
                    {/* Form */}
                    <Grid item xs={12} md={5}>
                        <Paper sx={{ 
                            p: 4, borderRadius: '20px', 
                            background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)',
                            backdropFilter: 'blur(20px)',
                            border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)',
                            boxShadow: theme.palette.mode === 'dark' ? '0 16px 40px -10px rgba(0,0,0,0.4)' : '0 16px 40px -10px rgba(0,0,0,0.05)',
                            height: '100%' 
                        }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4 }}>
                                <TuneIcon color="primary" fontSize="small" />
                                <Typography variant="h6" fontWeight={700}>Plan Parameters</Typography>
                            </Box>

                            {/* Lecture Select */}
                            <Box sx={{ mb: 3 }}>
                                <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1.5 }}>
                                    Lecture Note *
                                </Typography>
                                <LectureSelect value={noteId} onChange={setNoteId} />
                                {!noteId && (
                                    <Typography variant="caption" color="text.disabled" sx={{ mt: 0.5, display: 'block' }}>
                                        Your weaknesses and strengths will be analyzed from this note
                                    </Typography>
                                )}
                            </Box>

                            {/* Exam Date */}
                            <Box sx={{ mb: 3 }}>
                                <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1.5 }}>Target Exam Date</Typography>
                                <TextField
                                    type="date"
                                    fullWidth
                                    size="small"
                                    value={examDate}
                                    onChange={(e) => setExamDate(e.target.value)}
                                    InputLabelProps={{ shrink: true }}
                                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: '10px' } }}
                                />
                            </Box>

                            {/* Daily Hours */}
                            <Box sx={{ mb: 3 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
                                    <Typography variant="subtitle2" fontWeight={700}>Daily Study Hours</Typography>
                                    <Typography variant="subtitle2" fontWeight={800} color="primary.main">{hours}h</Typography>
                                </Box>
                                <Slider
                                    value={hours}
                                    onChange={(_, v) => setHours(v)}
                                    min={0.5}
                                    max={10}
                                    step={0.5}
                                    sx={{ height: 6, '& .MuiSlider-thumb': { width: 18, height: 18 } }}
                                />
                                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="caption" color="text.disabled">0.5h</Typography>
                                    <Typography variant="caption" color="text.disabled">10h</Typography>
                                </Box>
                            </Box>

                            {/* Priority Subjects */}
                            <Box sx={{ mb: 3 }}>
                                <Typography variant="subtitle2" fontWeight={700} sx={{ mb: 1.5 }}>Priority Topics (optional)</Typography>
                                <Box sx={{ display: 'flex', gap: 1, mb: 1.5 }}>
                                    <TextField
                                        placeholder="e.g., Thermodynamics"
                                        size="small"
                                        fullWidth
                                        value={priorityInput}
                                        onChange={(e) => setPriorityInput(e.target.value)}
                                        onKeyDown={handleKeyDown}
                                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '8px' } }}
                                    />
                                    <Button variant="contained" onClick={handleAddPriority} sx={{ minWidth: 44, borderRadius: '8px', p: 0 }}>
                                        <AddIcon />
                                    </Button>
                                </Box>
                                {priorities.length > 0 && (
                                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                                        {priorities.map(p => (
                                            <Chip
                                                key={p}
                                                label={p}
                                                onDelete={() => setPriorities(priorities.filter(i => i !== p))}
                                                size="small"
                                                sx={{ fontWeight: 600, borderRadius: '6px' }}
                                            />
                                        ))}
                                    </Box>
                                )}
                            </Box>

                            {/* Focus Weak Areas Toggle */}
                            <Paper sx={{ p: 2, mb: 4, borderRadius: '12px', bgcolor: 'rgba(19,127,236,0.05)', border: '1px solid', borderColor: 'rgba(19,127,236,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                    <AutoAwesomeIcon color="primary" fontSize="small" />
                                    <Box>
                                        <Typography variant="subtitle2" fontWeight={700}>Prioritize Weak Areas</Typography>
                                        <Typography variant="caption" color="text.secondary">Extra focus on topics you struggle with</Typography>
                                    </Box>
                                </Box>
                                <Switch checked={focusWeak} onChange={(e) => setFocusWeak(e.target.checked)} />
                            </Paper>

                            <Button
                                variant="contained"
                                fullWidth
                                size="large"
                                onClick={() => onGenerate({ exam_date: examDate, hours_per_day: hours, priority_subjects: priorities, focus_weak_areas: focusWeak })}
                                disabled={loading || !noteId}
                                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <AutoAwesomeIcon />}
                                sx={{ py: 1.75, borderRadius: '12px', fontWeight: 700, fontSize: '1rem', boxShadow: '0 8px 16px -4px rgba(19, 127, 236, 0.3)' }}
                            >
                                {loading ? 'Generating Your Plan...' : 'Generate Study Plan'}
                            </Button>
                        </Paper>
                    </Grid>

                    {/* Right side: Info */}
                    <Grid item xs={12} md={7}>
                        <Paper sx={{ 
                            p: 4, borderRadius: '20px', 
                            background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)',
                            backdropFilter: 'blur(20px)',
                            border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)',
                            boxShadow: theme.palette.mode === 'dark' ? '0 16px 40px -10px rgba(0,0,0,0.4)' : '0 16px 40px -10px rgba(0,0,0,0.05)',
                            height: '100%', display: 'flex', flexDirection: 'column', gap: 3 
                        }}>
                             <Typography variant="h6" fontWeight={700} color="text.secondary">What you'll get</Typography>
                            {[
                                { icon: <TrendingUpIcon />, title: 'Strength Analysis', desc: 'Discover which topics you already excel at, so you spend less time reviewing mastered content.' },
                                { icon: <WarningIcon />, title: 'Weakness Breakdown', desc: 'Pinpoint exactly which topics need your attention based on your quiz history.' },
                                { icon: <ArticleIcon />, title: 'Resource Recommendations', desc: 'Get curated articles, videos, and study materials tailored to your weak areas.' },
                                { icon: <QuizIcon />, title: 'Practice Plan', desc: 'A structured practice schedule with Easy → Hard progression for optimal learning.' },
                                { icon: <CalendarMonthIcon />, title: 'Revision Timeline', desc: 'Organized revision schedule aligned with your exam date and daily time commitment.' },
                            ].map(item => (
                                <Box key={item.title} sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                                    <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'rgba(19,127,236,0.08)', color: 'primary.main', flexShrink: 0, display: 'flex' }}>
                                        {React.cloneElement(item.icon, { fontSize: 'small' })}
                                    </Box>
                                    <Box>
                                        <Typography variant="subtitle2" fontWeight={700}>{item.title}</Typography>
                                        <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>{item.desc}</Typography>
                                    </Box>
                                </Box>
                            ))}
                        </Paper>
                    </Grid>
                </Grid>
            </Container>
        </Box>
    );
};

// ===================== DASHBOARD =====================
const StudyPlanDashboard = ({ planData, noteId, onReset }) => {
    const theme = useTheme();
    const navigate = useNavigate();
    const { plan_sections, strengths, weak_topics } = planData;
    const [dashStats, setDashStats] = useState(null);

    const weakList = parseBulletPoints(plan_sections?.weak);
    const strongList = parseBulletPoints(plan_sections?.strengths);
    const resourceLines = parseBulletPoints(plan_sections?.resources);
    const practiceLines = parseBulletPoints(plan_sections?.practice);
    const revisionLines = parseBulletPoints(plan_sections?.revision);

    useEffect(() => {
        API.get('dashboard/stats/').then(r => setDashStats(r.data)).catch(() => {});
    }, []);

    return (
        <Box sx={{ minHeight: '100vh', background: theme.palette.mode === 'dark' ? 'linear-gradient(135deg, #101922 0%, #1a2533 100%)' : 'linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%)', pb: 8 }}>
            {/* Header */}
            <Box sx={{ bgcolor: 'background.paper', borderBottom: '1px solid', borderColor: 'divider', px: { xs: 2, md: 4 }, py: 3 }}>
                <Container maxWidth="xl">
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 0.5 }}>
                                <PsychologyIcon color="primary" />
                                <Typography variant="h4" fontWeight={900} sx={{ letterSpacing: '-0.02em' }}>Your Study Plan</Typography>
                                <Chip label="AI Generated" color="primary" size="small" sx={{ fontWeight: 700 }} />
                            </Box>
                            <Typography variant="body2" color="text.secondary">Personalized based on your quiz history and topic mastery</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <Button variant="outlined" onClick={onReset} sx={{ fontWeight: 700, borderColor: 'divider', color: 'text.primary' }}>
                                Regenerate
                            </Button>
                            {noteId && (
                                <Button
                                    variant="contained"
                                    startIcon={<PlayCircleIcon />}
                                    onClick={() => navigate(`/quiz-entry?noteId=${noteId}`)}
                                    sx={{ fontWeight: 700, boxShadow: '0 4px 14px 0 rgba(19, 127, 236, 0.4)' }}
                                >
                                    Start Practice
                                </Button>
                            )}
                        </Box>
                    </Box>
                </Container>
            </Box>

            <Container maxWidth="xl" sx={{ mt: 4 }}>
                {/* Stats Row */}
                <Grid container spacing={3} sx={{ mb: 5 }}>
                    {[
                        {
                            icon: <TrophyIcon />, label: 'Strengths', bgcolor: 'rgba(16,185,129,0.1)', color: '#10B981',
                            value: Object.keys(strengths || {}).length,
                            sub: 'Mastered topics'
                        },
                        {
                            icon: <WarningIcon />, label: 'Weak Areas', bgcolor: 'rgba(245,158,11,0.1)', color: '#F59E0B',
                            value: Object.keys(weak_topics || {}).length,
                            sub: 'Needs attention'
                        },
                        {
                            icon: <FireIcon />, label: 'Study Streak', bgcolor: 'rgba(239,68,68,0.1)', color: '#EF4444',
                            value: dashStats ? `${dashStats.streak}d` : '--',
                            sub: 'Active days'
                        },
                        {
                            icon: <TrendingUpIcon />, label: 'Avg. Score', bgcolor: 'rgba(19,127,236,0.1)', color: '#137fec',
                            value: dashStats ? `${dashStats.avg_score}%` : '--',
                            sub: 'Overall accuracy'
                        },
                    ].map(s => (
                        <Grid item xs={6} sm={3} key={s.label}>
                            <Paper sx={{ p: 3, borderRadius: '16px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)' }}>
                                <Box sx={{ p: 1, borderRadius: '8px', bgcolor: s.bgcolor, color: s.color, display: 'inline-flex', mb: 2 }}>
                                    {React.cloneElement(s.icon, { fontSize: 'small' })}
                                </Box>
                                <Typography variant="h4" fontWeight={800} sx={{ color: s.color }}>{s.value}</Typography>
                                <Typography variant="body2" color="text.secondary">{s.sub}</Typography>
                            </Paper>
                        </Grid>
                    ))}
                </Grid>

                <Grid container spacing={4}>
                    {/* Left Column */}
                    <Grid item xs={12} lg={7}>
                        {/* Weak Areas */}
                        {weakList.length > 0 && (
                            <Box sx={{ mb: 5 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                                    <TrendingDownIcon color="warning" />
                                    <Typography variant="h6" fontWeight={700}>Focus Areas</Typography>
                                    <Chip label={`${weakList.length} topics`} size="small" color="warning" variant="outlined" sx={{ fontWeight: 700 }} />
                                </Box>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    {Object.entries(weak_topics || {}).slice(0, 5).map(([topic, score], idx) => {
                                        const pct = Math.round(score * 100);
                                        const masteryPct = Math.max(0, 100 - pct);
                                        return (
                                            <Paper key={idx} sx={{ p: 2.5, borderRadius: '12px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)', display: 'flex', alignItems: 'center', gap: 2.5 }}>
                                                <Box sx={{ flex: 1 }}>
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                                        <Typography variant="subtitle2" fontWeight={700}>{topic}</Typography>
                                                        <Chip
                                                            label={masteryPct < 40 ? 'Critical' : masteryPct < 60 ? 'Needs Work' : 'Fair'}
                                                            size="small"
                                                            color={masteryPct < 40 ? 'error' : masteryPct < 60 ? 'warning' : 'primary'}
                                                            variant="outlined"
                                                            sx={{ fontWeight: 700, fontSize: '0.65rem' }}
                                                        />
                                                    </Box>
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.75 }}>
                                                        <Typography variant="caption" color="text.secondary">Mastery</Typography>
                                                        <Typography variant="caption" fontWeight={700}>{masteryPct}%</Typography>
                                                    </Box>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={masteryPct}
                                                        color={masteryPct < 40 ? 'error' : masteryPct < 60 ? 'warning' : 'primary'}
                                                        sx={{ height: 6, borderRadius: 3, bgcolor: 'action.hover', '& .MuiLinearProgress-bar': { borderRadius: 3 } }}
                                                    />
                                                </Box>
                                                {noteId && (
                                                    <Button
                                                        variant="contained"
                                                        size="small"
                                                        onClick={() => navigate(`/quiz-entry?noteId=${noteId}`)}
                                                        sx={{ minWidth: 80, fontWeight: 700, flexShrink: 0, borderRadius: '8px' }}
                                                    >
                                                        Practice
                                                    </Button>
                                                )}
                                            </Paper>
                                        );
                                    })}
                                    {weakList.slice(0, 3).filter(item => !Object.keys(weak_topics || {}).find(t => item.includes(t))).map((item, idx) => (
                                        <Paper key={`text-${idx}`} sx={{ p: 2.5, borderRadius: '12px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)' }}>
                                            <Typography variant="body2" fontWeight={600}>{item}</Typography>
                                        </Paper>
                                    ))}
                                </Box>
                            </Box>
                        )}

                        {/* Strong Topics */}
                        {(Object.keys(strengths || {}).length > 0 || strongList.length > 0) && (
                            <Box sx={{ mb: 5 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                                    <CheckCircleIcon color="success" />
                                    <Typography variant="h6" fontWeight={700}>Strengths</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    {Object.entries(strengths || {}).slice(0, 4).map(([topic, score], idx) => (
                                        <Paper key={idx} sx={{ p: 2.5, borderRadius: '12px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                <Box sx={{ p: 1, borderRadius: '8px', bgcolor: 'rgba(16,185,129,0.1)', color: 'success.main', display: 'flex' }}>
                                                    <CheckCircleIcon fontSize="small" />
                                                </Box>
                                                <Box>
                                                    <Typography variant="subtitle2" fontWeight={700}>{topic}</Typography>
                                                    <Typography variant="caption" color="text.secondary">{Math.round(score * 100)}% Mastery</Typography>
                                                </Box>
                                            </Box>
                                            <Chip label="Mastered" size="small" color="success" variant="outlined" sx={{ fontWeight: 700 }} />
                                        </Paper>
                                    ))}
                                </Box>
                            </Box>
                        )}

                        {/* Resources */}
                        {resourceLines.length > 0 && (
                            <Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 3 }}>
                                    <ArticleIcon color="primary" />
                                    <Typography variant="h6" fontWeight={700}>Recommended Resources</Typography>
                                </Box>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                    {resourceLines.map((res, idx) => (
                                        <Paper key={idx} sx={{ p: 2, borderRadius: '12px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.04)' : 'divider', bgcolor: 'background.paper', display: 'flex', gap: 2, alignItems: 'center' }}>
                                            <Box sx={{ mt: 0.25, p: 0.75, borderRadius: '6px', bgcolor: 'rgba(19,127,236,0.08)', color: 'primary.main', display: 'flex', flexShrink: 0 }}>
                                                <ArticleIcon sx={{ fontSize: 16 }} />
                                            </Box>
                                            <Typography variant="body2" fontWeight={500} sx={{ lineHeight: 1.6 }}>{res}</Typography>
                                        </Paper>
                                    ))}
                                </Box>
                            </Box>
                        )}
                    </Grid>

                    {/* Right Column: Timeline */}
                    <Grid item xs={12} lg={5}>
                        {/* Study Timeline */}
                        <Paper sx={{ p: 4, mt: { xs: 4, lg: 0 }, borderRadius: '16px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)', height: '100%' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 4 }}>
                                <CalendarMonthIcon color="primary" />
                                <Typography variant="h6" fontWeight={700}>Study Timeline</Typography>
                            </Box>
                            <Box sx={{ position: 'relative', pl: 3, borderLeft: '2px solid', borderColor: 'divider', ml: 1 }}>
                                {/* Today */}
                                <Box sx={{ mb: 4, position: 'relative' }}>
                                    <Box sx={{ position: 'absolute', left: -19, top: 4, width: 12, height: 12, borderRadius: '50%', bgcolor: 'primary.main', border: `2px solid ${theme.palette.background.paper}`, zIndex: 1 }} />
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Checkbox size="small" />
                                            <Typography variant="overline" fontWeight={800} color="primary.main" sx={{ letterSpacing: '0.08em', display: 'block', mt: 0.5 }}>Today</Typography>
                                        </Box>
                                        <Paper sx={{ p: 2, borderRadius: '12px', bgcolor: 'rgba(19,127,236,0.05)', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.04)' : 'rgba(19,127,236,0.1)' }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                            <Typography variant="subtitle2" fontWeight={700} color="primary.main">Focus Session</Typography>
                                            <Chip label="Active" size="small" color="primary" sx={{ fontWeight: 700, height: 20, fontSize: '0.65rem' }} />
                                        </Box>
                                        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                            {practiceLines[0] || `Study ${Object.keys(weak_topics || {})[0] || 'weak topics'} with targeted practice`}
                                        </Typography>
                                        {noteId && (
                                            <Button variant="contained" fullWidth size="small" onClick={() => navigate(`/quiz-entry?noteId=${noteId}`)} sx={{ fontWeight: 700, borderRadius: '8px' }}>
                                                Start Session
                                            </Button>
                                        )}
                                    </Paper>
                                </Box>

                                {/* Practice items */}
                                {practiceLines.slice(0, 4).map((item, idx) => (
                                    <Box key={idx} sx={{ mb: 3, position: 'relative' }}>
                                        <Box sx={{ position: 'absolute', left: -19, top: 4, width: 12, height: 12, borderRadius: '50%', bgcolor: 'action.disabled', border: `2px solid ${theme.palette.background.paper}`, zIndex: 1 }} />
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Checkbox size="small" />
                                            <Typography variant="overline" fontWeight={700} color="text.secondary" sx={{ letterSpacing: '0.08em', display: 'block', mt: 0.5 }}>
                                                Day {idx + 2}+
                                            </Typography>
                                        </Box>
                                        <Paper sx={{ p: 2, borderRadius: '12px', border: '1px solid', borderColor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.6)', background: theme.palette.mode === 'dark' ? 'rgba(28, 37, 46, 0.65)' : 'rgba(255, 255, 255, 0.7)', backdropFilter: 'blur(20px)' }}>
                                            <Typography variant="body2" fontWeight={600}>{item}</Typography>
                                        </Paper>
                                    </Box>
                                ))}

                                {/* Revision */}
                                {revisionLines.length > 0 && (
                                    <Box sx={{ mb: 3, position: 'relative' }}>
                                        <Box sx={{ position: 'absolute', left: -19, top: 4, width: 12, height: 12, borderRadius: '50%', bgcolor: 'success.main', border: `2px solid ${theme.palette.background.paper}`, zIndex: 1 }} />
                                        <Typography variant="overline" fontWeight={700} color="success.main" sx={{ letterSpacing: '0.08em', display: 'block', mb: 1 }}>Final Revision</Typography>
                                        <Paper sx={{ p: 2.5, borderRadius: '10px', border: '1px solid', borderColor: 'rgba(16,185,129,0.2)', bgcolor: 'rgba(16,185,129,0.05)' }}>
                                            {revisionLines.slice(0, 3).map((line, i) => (
                                                <Typography key={i} variant="body2" color="text.secondary" sx={{ mb: i < revisionLines.length - 1 ? 0.75 : 0, display: 'flex', gap: 1, alignItems: 'center' }}>
                                                    <Checkbox size="small" sx={{ p: 0.5 }} /> {line}
                                                </Typography>
                                            ))}
                                        </Paper>
                                    </Box>
                                )}
                            </Box>
                        </Paper>

                        {/* Quick Links */}
                        <Paper sx={{ p: 3, borderRadius: '16px', border: '1px solid', borderColor: 'divider' }}>
                            <Typography variant="h6" fontWeight={700} sx={{ mb: 3 }}>Quick Actions</Typography>
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                {noteId && [
                                    { label: 'Start Quiz', icon: <PlayCircleIcon />, path: `/quiz-entry?noteId=${noteId}`, color: 'primary' },
                                    { label: 'View Flashcards', icon: <TrendingUpIcon />, path: '/flashcards', color: 'default' },
                                    { label: 'Lecture Summary', icon: <ArticleIcon />, path: '/summarize-lectures', color: 'default' },
                                ].map(a => (
                                    <Button
                                        key={a.label}
                                        variant={a.color === 'primary' ? 'contained' : 'outlined'}
                                        startIcon={a.icon}
                                        fullWidth
                                        onClick={() => navigate(a.path)}
                                        sx={{ fontWeight: 700, justifyContent: 'flex-start', borderColor: 'divider', color: a.color !== 'primary' ? 'text.primary' : undefined }}
                                    >
                                        {a.label}
                                    </Button>
                                ))}
                            </Box>
                        </Paper>
                    </Grid>
                </Grid>
            </Container>
        </Box>
    );
};

// ===================== MAIN =====================
export default function StudyPlan() {
    const [noteId, setNoteId] = useState('');
    const [planData, setPlanData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const location = useLocation();

    useEffect(() => {
        const searchParams = new URLSearchParams(location.search);
        const id = searchParams.get('noteId') || location.state?.noteId;
        if (id) setNoteId(id);
    }, [location.search, location.state]);

    const handleGenerate = async (params) => {
        setLoading(true);
        setError('');
        try {
            const res = await API.post('study-plan/', { note_id: noteId, ...params });
            setPlanData(res.data);
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to generate study plan. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    if (planData) {
        return <StudyPlanDashboard planData={planData} noteId={noteId} onReset={() => setPlanData(null)} />;
    }

    return (
        <>
            {error && (
                <Alert severity="error" onClose={() => setError('')} sx={{ m: 2, borderRadius: '12px' }}>{error}</Alert>
            )}
            <StudyPlanGenerator noteId={noteId} setNoteId={setNoteId} onGenerate={handleGenerate} loading={loading} />
        </>
    );
}
