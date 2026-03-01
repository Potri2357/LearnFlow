import React, { useState } from 'react';
import {
    Box, Typography, Button, Paper, Grid, Stack, useTheme,
    Avatar, CircularProgress, IconButton, Chip, LinearProgress,
    Alert, Snackbar
} from '@mui/material';
import {
    AutoAwesome as AutoAwesomeIcon,
    UploadFile as UploadFileIcon,
    Description as DescriptionIcon,
    Download as DownloadIcon,
    ZoomIn as ZoomInIcon,
    ZoomOut as ZoomOutIcon,
    Analytics as AnalyticsIcon,
    CheckCircle as CheckCircleIcon,
    Warning as WarningIcon,
    PlayCircle as PlayCircleIcon,
    Refresh as RefreshIcon,
    Assignment as AssignmentIcon,
    EmojiEvents as TrophyIcon,
    Lightbulb as LightbulbIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import API from '../api/api';

const GlassCard = ({ children, sx, ...props }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    return (
        <Paper
            elevation={0}
            sx={{
                background: isDark ? 'rgba(28, 37, 46, 0.8)' : 'rgba(255, 255, 255, 0.9)',
                backdropFilter: 'blur(16px)',
                border: '1px solid',
                borderColor: isDark ? '#2a3b4d' : 'rgba(0, 0, 0, 0.1)',
                borderRadius: '16px',
                ...sx
            }}
            {...props}
        >
            {children}
        </Paper>
    );
};

const ScoreBar = ({ label, score, color = 'primary' }) => (
    <Box>
        <Box display="flex" justifyContent="space-between" mb={1}>
            <Typography variant="body2" color="text.secondary" fontWeight={600}>{label}</Typography>
            <Typography variant="body2" fontWeight={800} color={`${color}.main`}>{score}/100</Typography>
        </Box>
        <LinearProgress
            variant="determinate"
            value={score}
            color={color}
            sx={{ height: 8, borderRadius: 4 }}
        />
    </Box>
);

export default function RubricEvaluator() {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [snack, setSnack] = useState({ open: false, msg: '', severity: 'error' });

    const onDrop = async (acceptedFiles) => {
        if (acceptedFiles.length === 0) return;
        const uploadedFile = acceptedFiles[0];
        setFile(uploadedFile);
        setResult(null);
        await analyzeAssignment(uploadedFile);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'application/pdf': ['.pdf'],
            'text/plain': ['.txt'],
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx']
        },
        multiple: false,
    });

    const analyzeAssignment = async (uploadFile) => {
        setLoading(true);
        try {
            const formData = new FormData();
            formData.append('file', uploadFile);
            formData.append('subject', 'General Assignment');

            const res = await API.post('/ai-tutor/evaluate/', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            setResult(res.data);
        } catch (error) {
            console.error(error);
            setSnack({ open: true, msg: 'Failed to analyze assignment. Please try again.', severity: 'error' });
        } finally {
            setLoading(false);
        }
    };

    const handleReset = () => {
        setFile(null);
        setResult(null);
    };

    const evaluation = result?.evaluation || {};

    const scoreColor = (score) => {
        if (score >= 80) return '#10b981';
        if (score >= 60) return '#f59e0b';
        return '#ef4444';
    };

    return (
        <Box sx={{ maxWidth: 1400, mx: 'auto', p: { xs: 2, md: 4 } }}>
            <Snackbar
                open={snack.open}
                autoHideDuration={5000}
                onClose={() => setSnack(s => ({ ...s, open: false }))}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert severity={snack.severity} variant="filled" onClose={() => setSnack(s => ({ ...s, open: false }))} sx={{ borderRadius: 2 }}>
                    {snack.msg}
                </Alert>
            </Snackbar>

            {/* Header */}
            <Box mb={5} display="flex" justifyContent="space-between" alignItems="flex-start" flexWrap="wrap" gap={2}>
                <Box>
                    <Box display="flex" alignItems="center" gap={1.5} mb={1}>
                        <Box sx={{
                            width: 44, height: 44, borderRadius: '12px',
                            background: 'linear-gradient(135deg, #2563EB, #7c3aed)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>
                            <AssignmentIcon sx={{ color: 'white', fontSize: 24 }} />
                        </Box>
                        <Box>
                            <Typography variant="h4" fontWeight={900} letterSpacing="-0.02em">AI Rubric Evaluator</Typography>
                            <Typography variant="body2" color="text.secondary">
                                {file ? `Evaluating: ${file.name}` : 'Upload your assignment for intelligent feedback'}
                            </Typography>
                        </Box>
                    </Box>
                </Box>
                {result && (
                    <Box display="flex" gap={2}>
                        <Button variant="outlined" startIcon={<RefreshIcon />} onClick={handleReset}
                            sx={{ fontWeight: 800, borderRadius: 2, borderColor: 'divider', color: 'text.primary' }}>
                            New Submission
                        </Button>
                        <Button variant="contained" startIcon={<DownloadIcon />}
                            sx={{ fontWeight: 800, borderRadius: 2, background: 'linear-gradient(135deg, #2563EB, #7c3aed)' }}>
                            Export Report
                        </Button>
                    </Box>
                )}
            </Box>

            {/* Upload zone */}
            {!result && !loading && (
                <GlassCard
                    sx={{
                        p: { xs: 6, md: 10 },
                        textAlign: 'center',
                        cursor: 'pointer',
                        border: '2px dashed',
                        borderColor: isDragActive ? 'primary.main' : 'divider',
                        transition: 'all 0.25s',
                        '&:hover': { borderColor: 'primary.main', bgcolor: isDark ? 'rgba(37,99,235,0.04)' : 'rgba(37,99,235,0.02)' }
                    }}
                    {...getRootProps()}
                >
                    <input {...getInputProps()} />
                    <Box sx={{ width: 80, height: 80, borderRadius: '20px', background: 'linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.15))', border: '2px solid rgba(37,99,235,0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', mx: 'auto', mb: 3 }}>
                        <UploadFileIcon sx={{ fontSize: 40, color: isDragActive ? 'primary.main' : 'text.secondary' }} />
                    </Box>
                    <Typography variant="h5" fontWeight={800} gutterBottom>
                        {isDragActive ? 'Drop it here!' : 'Upload Assignment for AI Evaluation'}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Drag and drop your PDF, DOCX, or TXT file, or click to browse.
                    </Typography>
                    <Box display="flex" gap={1} justifyContent="center">
                        {['PDF', 'DOCX', 'TXT'].map(fmt => (
                            <Chip key={fmt} label={fmt} size="small" sx={{ fontWeight: 700, borderRadius: 1, bgcolor: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)' }} />
                        ))}
                    </Box>
                </GlassCard>
            )}

            {/* Loading state */}
            {loading && (
                <GlassCard sx={{ p: 10, textAlign: 'center' }}>
                    <CircularProgress size={64} thickness={4} sx={{ mb: 4 }} />
                    <Typography variant="h5" fontWeight={800} gutterBottom>AI is Analyzing Your Submission…</Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ maxWidth: 480, mx: 'auto' }}>
                        Checking content accuracy, clarity, originality, and mapping to rubric standards. This takes a few seconds.
                    </Typography>
                </GlassCard>
            )}

            {/* Results */}
            {result && !loading && (
                <Grid container spacing={4}>
                    {/* Left: Document preview */}
                    <Grid item xs={12} lg={7}>
                        <GlassCard sx={{ display: 'flex', flexDirection: 'column', minHeight: 520, overflow: 'hidden' }}>
                            {/* preview toolbar */}
                            <Box sx={{
                                px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider',
                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                bgcolor: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)'
                            }}>
                                <Box display="flex" alignItems="center" gap={1.5}>
                                    <DescriptionIcon color="primary" fontSize="small" />
                                    <Typography variant="subtitle2" fontWeight={800}>{file?.name}</Typography>
                                </Box>
                                <Box display="flex" gap={0.5}>
                                    <IconButton size="small" sx={{ borderRadius: 1.5 }}>
                                        <ZoomOutIcon fontSize="small" />
                                    </IconButton>
                                    <IconButton size="small" sx={{ borderRadius: 1.5 }}>
                                        <ZoomInIcon fontSize="small" />
                                    </IconButton>
                                </Box>
                            </Box>
                            <Box p={4} flex={1} overflow="auto" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.9 }}>
                                <Typography variant="body1" color="text.secondary">
                                    {result.content}
                                </Typography>
                            </Box>
                        </GlassCard>
                    </Grid>

                    {/* Right: Evaluation panel */}
                    <Grid item xs={12} lg={5}>
                        <Stack spacing={3}>
                            {/* Overall score */}
                            <GlassCard sx={{ p: 3, overflow: 'hidden', position: 'relative' }}>
                                <Box sx={{ position: 'absolute', top: -30, right: -30, width: 120, height: 120, borderRadius: '50%', bgcolor: 'primary.main', opacity: 0.08 }} />
                                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                                    <Typography variant="h6" fontWeight={800}>Overall Score</Typography>
                                    <TrophyIcon sx={{ color: scoreColor(evaluation.overall_score), fontSize: 28 }} />
                                </Box>
                                <Box display="flex" alignItems="flex-end" gap={2} mb={3}>
                                    <Typography variant="h2" fontWeight={900} letterSpacing="-0.03em"
                                        sx={{ color: scoreColor(evaluation.overall_score) }}>
                                        {evaluation.overall_score}%
                                    </Typography>
                                    <Typography variant="subtitle1" fontWeight={700} color="text.secondary" pb={1}>
                                        {evaluation.quality}
                                    </Typography>
                                </Box>
                                <Stack spacing={2}>
                                    <ScoreBar label="Content Accuracy" score={evaluation.content_accuracy_score} />
                                    <ScoreBar label="Clarity & Logic" score={evaluation.clarity_logic_score} color="secondary" />
                                </Stack>
                            </GlassCard>

                            {/* Originality check */}
                            <GlassCard sx={{ p: 3 }}>
                                <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                                    <Typography variant="subtitle1" fontWeight={800}>Originality Check</Typography>
                                    <Chip
                                        label={evaluation.originality_pass ? 'PASS' : 'FAIL'}
                                        size="small"
                                        sx={{
                                            fontWeight: 800, borderRadius: 1,
                                            bgcolor: evaluation.originality_pass ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                                            color: evaluation.originality_pass ? '#10b981' : '#ef4444'
                                        }}
                                    />
                                </Box>
                                <Box display="flex" alignItems="center" gap={3}>
                                    <Box position="relative" display="inline-flex">
                                        <CircularProgress variant="determinate" value={100} size={70} sx={{ color: 'divider', position: 'absolute' }} thickness={5} />
                                        <CircularProgress variant="determinate" value={evaluation.originality_score} size={70} sx={{ color: '#10b981' }} thickness={5} />
                                        <Box sx={{ top: 0, left: 0, bottom: 0, right: 0, position: 'absolute', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                            <Typography variant="subtitle2" fontWeight={900}>{evaluation.originality_score}%</Typography>
                                        </Box>
                                    </Box>
                                    <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                        {evaluation.originality_insight}
                                    </Typography>
                                </Box>
                            </GlassCard>

                            {/* AI Feedback panel */}
                            <Box sx={{ p: 3, bgcolor: '#0f172a', borderRadius: 3, color: 'white', border: '1px solid rgba(37,99,235,0.3)' }}>
                                <Box display="flex" alignItems="center" gap={1.5} borderBottom="1px solid rgba(255,255,255,0.08)" pb={2} mb={3}>
                                    <AutoAwesomeIcon color="primary" fontSize="small" />
                                    <Typography variant="subtitle1" fontWeight={800}>AI Feedback</Typography>
                                </Box>

                                {/* Strengths */}
                                {evaluation.top_strengths?.length > 0 && (
                                    <Box mb={3}>
                                        <Typography variant="caption" fontWeight={800} color="rgba(16,185,129,1)" letterSpacing="0.1em" display="block" mb={1.5} textTransform="uppercase">
                                            ✦ Top Strengths
                                        </Typography>
                                        <Stack spacing={1.5}>
                                            {evaluation.top_strengths.map((str, i) => (
                                                <Box key={i} display="flex" gap={1.5} alignItems="flex-start">
                                                    <CheckCircleIcon sx={{ color: '#10b981', fontSize: 18, mt: 0.2, flexShrink: 0 }} />
                                                    <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)', lineHeight: 1.6 }}>{str}</Typography>
                                                </Box>
                                            ))}
                                        </Stack>
                                    </Box>
                                )}

                                {/* Suggestions */}
                                {evaluation.actionable_suggestions?.length > 0 && (
                                    <Box>
                                        <Typography variant="caption" fontWeight={800} color="rgba(245,158,11,1)" letterSpacing="0.1em" display="block" mb={1.5} textTransform="uppercase">
                                            ⬡ Actionable Suggestions
                                        </Typography>
                                        <Stack spacing={2}>
                                            {evaluation.actionable_suggestions.map((sug, i) => (
                                                <Box key={i} sx={{
                                                    p: 2, bgcolor: 'rgba(255,255,255,0.04)', borderRadius: 2,
                                                    borderLeft: '3px solid', borderColor: i === 0 ? '#f59e0b' : '#2563EB'
                                                }}>
                                                    <Box display="flex" alignItems="center" gap={1} mb={0.5}>
                                                        <LightbulbIcon sx={{ color: i === 0 ? '#f59e0b' : '#60a5fa', fontSize: 16 }} />
                                                        <Typography variant="subtitle2" fontWeight={800}>{sug.title}</Typography>
                                                    </Box>
                                                    <Typography variant="caption" sx={{ color: 'rgba(255,255,255,0.6)', lineHeight: 1.6 }}>{sug.description}</Typography>
                                                </Box>
                                            ))}
                                        </Stack>
                                    </Box>
                                )}
                            </Box>
                        </Stack>
                    </Grid>
                </Grid>
            )}
        </Box>
    );
}
