import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import API from '../api/api';
import {
    Box, Typography, Button, Paper, Grid, Stack, TextField,
    Switch, Chip, LinearProgress, IconButton, Avatar, useTheme,
    CircularProgress, Divider, Collapse, Table, TableBody, TableCell,
    TableContainer, TableHead, TableRow, Tooltip, Alert, Snackbar,
    FormControlLabel, Slider, InputAdornment, Select, MenuItem, FormControl, InputLabel
} from '@mui/material';
import {
    School as SchoolIcon,
    UploadFile as UploadIcon,
    Tune as TuneIcon,
    Analytics as AnalyticsIcon,
    Bookmark as BookmarkIcon,
    Refresh as RefreshIcon,
    TaskAlt as TaskAltIcon,
    Schedule as ScheduleIcon,
    TrendingUp as TrendingUpIcon,
    CheckCircle as CheckCircleIcon,
    Warning as WarningIcon,
    EmojiEvents as TrophyIcon,
    ExpandMore as ExpandMoreIcon,
    ExpandLess as ExpandLessIcon,
    Add as AddIcon,
    Delete as DeleteIcon,
    AutoAwesome as AutoAwesomeIcon,
    MenuBook as MenuBookIcon,
    Coffee as CoffeeIcon,
    FitnessCenter as FitnessCenterIcon
} from '@mui/icons-material';

// Shared Glass Card
const GlassCard = ({ children, sx, ...props }) => {
    const theme = useTheme();
    const isDark = theme.palette.mode === 'dark';
    return (
        <Paper elevation={0} sx={{
            background: isDark ? 'rgba(28, 37, 46, 0.7)' : 'rgba(255, 255, 255, 0.7)',
            backdropFilter: 'blur(16px)',
            border: '1px solid',
            borderColor: isDark ? '#2a3b4d' : 'rgba(0, 0, 0, 0.1)',
            borderRadius: '16px',
            ...sx
        }} {...props}>
            {children}
        </Paper>
    );
};

// Renders markdown content safely
const AnswerText = ({ text }) => {
    if (!text) return null;
    return (
        <Box sx={{
            fontSize: '0.9rem',
            lineHeight: 1.8,
            color: 'text.secondary',
            '& p': { mb: 1, mt: 0 },
            '& strong': { fontWeight: 800, color: 'primary.light' },
            '& ul, & ol': { pl: 3, mb: 1, mt: 0.5 }
        }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
        </Box>
    );
};
// Single question card with reveal toggle
const QuestionCard = ({ q, index }) => {
    const [revealed, setRevealed] = useState(false);
    const [bookmarked, setBookmarked] = useState(false);
    const theme = useTheme();

    const marksColor = q.marks >= 10 ? '#ef4444' : q.marks >= 5 ? '#f59e0b' : '#10b981';

    return (
        <Paper elevation={0} sx={{
            borderRadius: 3,
            border: '1px solid',
            borderColor: revealed ? 'primary.main' : 'divider',
            overflow: 'hidden',
            transition: 'all 0.3s ease',
            '&:hover': { borderColor: 'primary.main', boxShadow: theme.palette.mode === 'dark' ? '0 0 0 1px rgba(19,127,236,0.3)' : '0 4px 20px rgba(19,127,236,0.1)' }
        }}>
            {/* Question Header */}
            <Box sx={{ p: 3, bgcolor: theme.palette.mode === 'dark' ? 'rgba(25, 38, 51, 0.9)' : 'rgba(240,244,248,1)' }}>
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                    <Box display="flex" gap={1} flexWrap="wrap" alignItems="center">
                        <Avatar sx={{ width: 28, height: 28, bgcolor: 'primary.main', fontSize: '0.75rem', fontWeight: 800 }}>
                            {index + 1}
                        </Avatar>
                        <Chip label={q.topic} size="small" sx={{ bgcolor: 'rgba(19,127,236,0.12)', color: 'primary.main', fontWeight: 700, borderRadius: 1 }} />
                        <Chip
                            label={`${q.marks} Mark${q.marks > 1 ? 's' : ''}`}
                            size="small"
                            sx={{ bgcolor: `${marksColor}20`, color: marksColor, fontWeight: 800, borderRadius: 1 }}
                        />
                        {q.is_from_pattern && (
                            <Chip label="High-Yield Pattern" size="small" icon={<TrendingUpIcon sx={{ fontSize: 14 }} />}
                                sx={{ bgcolor: 'rgba(76,175,80,0.12)', color: '#4caf50', fontWeight: 700, borderRadius: 1 }} />
                        )}
                    </Box>
                    <Box display="flex" gap={1}>
                        <Tooltip title={bookmarked ? 'Remove bookmark' : 'Bookmark'}>
                            <IconButton size="small" onClick={() => setBookmarked(!bookmarked)}
                                sx={{ color: bookmarked ? 'warning.main' : 'text.secondary' }}>
                                <BookmarkIcon fontSize="small" />
                            </IconButton>
                        </Tooltip>
                    </Box>
                </Box>
                <Box sx={{ 
                     typography: 'subtitle1', 
                     fontWeight: 700, 
                     lineHeight: 1.5,
                     '& p': { m: 0 },
                     '& strong': { fontWeight: 900, color: 'primary.main' },
                     mb: 1
                }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {q.question_text || q.title || ''}
                    </ReactMarkdown>
                </Box>
                <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                    <Typography variant="caption" color="text.secondary">
                        Priority #{q.priority} &bull; {q.marks} mark{q.marks > 1 ? 's' : ''}
                    </Typography>
                    <Button
                        size="small"
                        variant={revealed ? 'contained' : 'outlined'}
                        endIcon={revealed ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                        onClick={() => setRevealed(!revealed)}
                        sx={{ borderRadius: 2, fontWeight: 700, fontSize: '0.78rem' }}
                    >
                        {revealed ? 'Hide Answer' : 'Reveal Answer'}
                    </Button>
                </Box>
            </Box>

            {/* Answer */}
            <Collapse in={revealed}>
                <Box sx={{ p: 3, borderTop: '1px solid', borderColor: 'divider', bgcolor: theme.palette.mode === 'dark' ? 'rgba(19,127,236,0.04)' : 'rgba(19,127,236,0.02)' }}>
                    <Box display="flex" alignItems="center" gap={1} mb={2}>
                        <CheckCircleIcon sx={{ fontSize: 18, color: '#10b981' }} />
                        <Typography variant="caption" fontWeight={800} color="primary.main" letterSpacing="0.08em" textTransform="uppercase">
                            Model Answer ({q.marks} marks)
                        </Typography>
                    </Box>
                    <AnswerText text={q.answer} />
                </Box>
            </Collapse>
        </Paper>
    );
};

// Mark Distribution Row
const MarkRow = ({ marks, count, onChange, onRemove }) => (
    <Box display="flex" gap={2} alignItems="center">
        <TextField
            label="Marks" type="number" size="small" value={marks}
            InputProps={{ inputProps: { min: 1 } }}
            sx={{ width: 100 }} disabled
        />
        <TextField
            label="No. of Questions" type="number" size="small" value={count}
            onChange={e => onChange(parseInt(e.target.value) || 1)}
            InputProps={{ inputProps: { min: 1 } }}
            sx={{ flex: 1 }}
        />
        <IconButton size="small" color="error" onClick={onRemove}><DeleteIcon fontSize="small" /></IconButton>
    </Box>
);

// =====================================================
// VIEW 1: Question Bank Generator
// =====================================================
const QuestionBankView = ({ syllabi, activeSyllabusId, setActiveSyllabusId, onSyllabusUploaded }) => {
    const theme = useTheme();
    const [questions, setQuestions] = useState([]);
    const [syllabusLoading, setSyllabusLoading] = useState(false);
    const [papersLoading, setPapersLoading] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [fetchingQuestions, setFetchingQuestions] = useState(false);
    const [secureCentumMode, setSecureCentumMode] = useState(true);
    const [uploadedSyllabusFile, setUploadedSyllabusFile] = useState(null);
    const [uploadedPapers, setUploadedPapers] = useState([]);
    const [snack, setSnack] = useState({ open: false, msg: '', severity: 'success' });

    // Mark distribution config
    const [markDistRows, setMarkDistRows] = useState([
        { marks: 2, count: 5 },
        { marks: 5, count: 5 },
        { marks: 10, count: 2 },
    ]);
    const [filterTab, setFilterTab] = useState('all');

    const totalQuestions = markDistRows.reduce((s, r) => s + r.count, 0);
    const totalMarks = markDistRows.reduce((s, r) => s + r.marks * r.count, 0);

    const activeSyllabus = syllabi.find(s => s.id === activeSyllabusId) || syllabi[0];

    useEffect(() => {
        if (activeSyllabusId) {
            setFetchingQuestions(true);
            API.get(`/exam/syllabus/${activeSyllabusId}/questions/`)
                .then(res => setQuestions(res.data.questions || []))
                .catch(err => console.error(err))
                .finally(() => setFetchingQuestions(false));
        } else {
            setQuestions([]);
        }
    }, [activeSyllabusId]);

    // Syllabus dropzone
    const onDropSyllabus = async (acceptedFiles) => {
        if (!acceptedFiles.length) return;
        const file = acceptedFiles[0];
        setSyllabusLoading(true);
        setUploadedSyllabusFile(file);
        const formData = new FormData();
        formData.append('title', file.name.replace(/\.[^.]+$/, ''));
        formData.append('file', file);
        try {
            const res = await API.post('/exam/syllabus/upload/', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
            onSyllabusUploaded(res.data);
            setSnack({ open: true, msg: `Syllabus "${res.data.title}" uploaded successfully!`, severity: 'success' });
        } catch (err) {
            console.error(err);
            setSnack({ open: true, msg: 'Syllabus upload failed. Please try again.', severity: 'error' });
            setUploadedSyllabusFile(null);
        } finally {
            setSyllabusLoading(false);
        }
    };

    // Papers dropzone — only enabled after a syllabus is picked
    const onDropPapers = async (acceptedFiles) => {
        if (!acceptedFiles.length || !activeSyllabusId) return;
        setPapersLoading(true);
        const formData = new FormData();
        acceptedFiles.forEach(f => formData.append('files', f));
        try {
            const res = await API.post(`/exam/syllabus/${activeSyllabusId}/papers/`, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
            setUploadedPapers(prev => [...prev, ...acceptedFiles.map(f => f.name)]);
            setSnack({ open: true, msg: `${res.data.uploaded_count} previous paper(s) uploaded!`, severity: 'success' });
        } catch (err) {
            console.error(err);
            setSnack({ open: true, msg: 'Paper upload failed.', severity: 'error' });
        } finally {
            setPapersLoading(false);
        }
    };

    const { getRootProps: getSyllabusProps, getInputProps: getSyllabusInput, isDragActive: isSyllabusDrag } = useDropzone({
        onDrop: onDropSyllabus,
        accept: { 'application/pdf': ['.pdf'], 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'] },
        multiple: false,
        disabled: syllabusLoading
    });

    const { getRootProps: getPapersProps, getInputProps: getPapersInput, isDragActive: isPapersDrag } = useDropzone({
        onDrop: onDropPapers,
        accept: { 'application/pdf': ['.pdf'] },
        disabled: !activeSyllabusId || papersLoading
    });

    const addMarkRow = () => {
        const usedMarks = markDistRows.map(r => r.marks);
        const candidates = [1, 2, 3, 5, 10, 15, 20].filter(m => !usedMarks.includes(m));
        if (candidates.length === 0) return;
        setMarkDistRows([...markDistRows, { marks: candidates[0], count: 2 }]);
    };

    const updateMarkRow = (index, field, value) => {
        const updated = [...markDistRows];
        updated[index] = { ...updated[index], [field]: value };
        setMarkDistRows(updated);
    };

    const removeMarkRow = (index) => {
        setMarkDistRows(markDistRows.filter((_, i) => i !== index));
    };

    const handleGenerate = async () => {
        if (!activeSyllabusId) {
            setSnack({ open: true, msg: 'Please upload or select a syllabus first.', severity: 'warning' });
            return;
        }
        setGenerating(true);
        const markDist = {};
        markDistRows.forEach(r => { markDist[String(r.marks)] = r.count; });

        try {
            const res = await API.post(`/exam/syllabus/${activeSyllabusId}/generate/`, {
                total_marks: totalMarks,
                num_questions: totalQuestions,
                mark_distribution: markDist,
                secure_centum_mode: secureCentumMode
            });
            setQuestions(res.data.questions || []);
            setSnack({ open: true, msg: `Generated ${res.data.questions_generated} questions!`, severity: 'success' });
        } catch (err) {
            console.error(err);
            setSnack({ open: true, msg: err?.response?.data?.details || 'Generation failed. Try again.', severity: 'error' });
        } finally {
            setGenerating(false);
        }
    };

    const filteredQuestions = filterTab === 'frequent'
        ? questions.filter(q => q.is_from_pattern)
        : filterTab === 'high'
        ? questions.filter(q => q.marks >= 10)
        : questions;

    return (
        <Box sx={{ maxWidth: 1400, mx: 'auto', p: { xs: 2, md: 4 } }}>
            <Snackbar open={snack.open} autoHideDuration={4000} onClose={() => setSnack(s => ({ ...s, open: false }))}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
                <Alert severity={snack.severity} variant="filled" onClose={() => setSnack(s => ({ ...s, open: false }))} sx={{ borderRadius: 2, fontWeight: 700 }}>
                    {snack.msg}
                </Alert>
            </Snackbar>

            <Box mb={4}>
                <Typography variant="h3" fontWeight={900} gutterBottom sx={{ letterSpacing: '-0.02em' }}>Question Bank Generator</Typography>
                <Typography variant="body1" color="text.secondary">Upload your syllabus, configure the exam pattern and let AI generate high-quality questions with detailed answers.</Typography>
            </Box>

            <Grid container spacing={3} alignItems="flex-start">
                {/* LEFT PANEL */}
                <Grid item xs={12} xl={4}>
                    <Stack spacing={3}>

                        {/* Step 1: Syllabus Selector + Upload */}
                        <GlassCard sx={{ p: 3 }}>
                            <Box display="flex" alignItems="center" gap={1.5} mb={2}>
                                <Box sx={{ width: 28, height: 28, borderRadius: '50%', bgcolor: 'primary.main', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 800, fontSize: '0.75rem' }}>1</Box>
                                <Typography variant="h6" fontWeight={800}>Syllabus</Typography>
                            </Box>

                            {/* Syllabus selector if existing */}
                            {syllabi.length > 0 && (
                                <Box mb={2}>
                                    <FormControl fullWidth size="small">
                                        <InputLabel>Active Syllabus</InputLabel>
                                        <Select
                                            value={activeSyllabusId || ''}
                                            label="Active Syllabus"
                                            onChange={e => setActiveSyllabusId(e.target.value)}
                                            sx={{ borderRadius: 2 }}
                                        >
                                            {syllabi.map(s => (
                                                <MenuItem key={s.id} value={s.id}>
                                                    <Box>
                                                        <Typography variant="body2" fontWeight={700}>{s.title}</Typography>
                                                        <Typography variant="caption" color="text.secondary">{s.question_count} question{s.question_count !== 1 ? 's' : ''} generated</Typography>
                                                    </Box>
                                                </MenuItem>
                                            ))}
                                        </Select>
                                    </FormControl>
                                    <Divider sx={{ my: 2 }}><Typography variant="caption" color="text.secondary" fontWeight={700}>OR UPLOAD NEW</Typography></Divider>
                                </Box>
                            )}

                            <Box {...getSyllabusProps()} sx={{
                                border: '2px dashed',
                                borderColor: isSyllabusDrag ? 'primary.main' : uploadedSyllabusFile ? '#10b981' : 'divider',
                                borderRadius: 2, p: 3, textAlign: 'center', cursor: 'pointer',
                                bgcolor: isSyllabusDrag ? 'rgba(19,127,236,0.04)' : 'transparent',
                                transition: 'all 0.2s',
                                '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(19,127,236,0.03)' }
                            }}>
                                <input {...getSyllabusInput()} />
                                {syllabusLoading ? (
                                    <Box display="flex" flexDirection="column" alignItems="center" gap={1}>
                                        <CircularProgress size={32} />
                                        <Typography variant="caption" color="text.secondary">Uploading syllabus…</Typography>
                                    </Box>
                                ) : uploadedSyllabusFile ? (
                                    <Box>
                                        <CheckCircleIcon sx={{ color: '#10b981', fontSize: 36, mb: 0.5 }} />
                                        <Typography variant="subtitle2" fontWeight={800} color="#10b981">{uploadedSyllabusFile.name}</Typography>
                                        <Typography variant="caption" color="text.secondary">Click to replace</Typography>
                                    </Box>
                                ) : (
                                    <Box>
                                        <UploadIcon sx={{ fontSize: 36, color: 'text.secondary', mb: 0.5 }} />
                                        <Typography variant="subtitle2" fontWeight={700}>Drop PDF or DOCX here</Typography>
                                        <Typography variant="caption" color="text.secondary">Up to 10 MB</Typography>
                                    </Box>
                                )}
                            </Box>
                        </GlassCard>

                        {/* Step 2: Previous Papers (Optional) */}
                        <GlassCard sx={{ p: 3, opacity: activeSyllabusId ? 1 : 0.6 }}>
                            <Box display="flex" alignItems="center" gap={1.5} mb={2}>
                                <Box sx={{ width: 28, height: 28, borderRadius: '50%', bgcolor: activeSyllabusId ? 'warning.main' : 'text.disabled', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 800, fontSize: '0.75rem' }}>2</Box>
                                <Typography variant="h6" fontWeight={800}>Previous Papers <Typography component="span" variant="caption" color="text.secondary">(Optional)</Typography></Typography>
                            </Box>
                            <Box {...getPapersProps()} sx={{
                                border: '2px dashed',
                                borderColor: isPapersDrag ? 'warning.main' : uploadedPapers.length > 0 ? '#f59e0b' : 'divider',
                                borderRadius: 2, p: 3, textAlign: 'center',
                                cursor: activeSyllabusId ? 'pointer' : 'not-allowed',
                                transition: 'all 0.2s',
                                '&:hover': activeSyllabusId ? { borderColor: 'warning.main', bgcolor: 'rgba(245,158,11,0.04)' } : {}
                            }}>
                                <input {...getPapersInput()} />
                                {papersLoading ? (
                                    <Box display="flex" flexDirection="column" alignItems="center" gap={1}>
                                        <CircularProgress size={28} color="warning" />
                                        <Typography variant="caption">Uploading papers…</Typography>
                                    </Box>
                                ) : uploadedPapers.length > 0 ? (
                                    <Box>
                                        <CheckCircleIcon sx={{ color: '#f59e0b', fontSize: 32, mb: 0.5 }} />
                                        <Typography variant="subtitle2" fontWeight={800} color="warning.main">{uploadedPapers.length} paper(s) added</Typography>
                                        <Typography variant="caption" color="text.secondary">Drop more to add</Typography>
                                    </Box>
                                ) : (
                                    <Box>
                                        <MenuBookIcon sx={{ fontSize: 32, color: 'text.secondary', mb: 0.5 }} />
                                        <Typography variant="subtitle2" fontWeight={700}>{activeSyllabusId ? 'Drop past papers (PDF)' : 'Upload syllabus first'}</Typography>
                                        <Typography variant="caption" color="text.secondary">Improves pattern accuracy</Typography>
                                    </Box>
                                )}
                            </Box>
                            {uploadedPapers.length > 0 && (
                                <Stack spacing={0.5} mt={1.5}>
                                    {uploadedPapers.map((p, i) => (
                                        <Chip key={i} label={p} size="small" icon={<CheckCircleIcon />} sx={{ borderRadius: 1, justifyContent: 'flex-start', bgcolor: 'rgba(245,158,11,0.08)', color: 'warning.main' }} />
                                    ))}
                                </Stack>
                            )}
                        </GlassCard>

                        {/* Step 3: Exam Pattern Config */}
                        <GlassCard sx={{ p: 3 }}>
                            <Box display="flex" alignItems="center" gap={1.5} mb={2}>
                                <Box sx={{ width: 28, height: 28, borderRadius: '50%', bgcolor: '#7c3aed', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 800, fontSize: '0.75rem' }}>3</Box>
                                <Typography variant="h6" fontWeight={800}>Exam Pattern</Typography>
                            </Box>

                            <Stack spacing={2}>
                                <Typography variant="caption" fontWeight={700} color="text.secondary" display="block">MARK DISTRIBUTION</Typography>
                                {markDistRows.map((row, i) => (
                                    <MarkRow
                                        key={i}
                                        marks={row.marks}
                                        count={row.count}
                                        onChange={(val) => updateMarkRow(i, 'count', val)}
                                        onRemove={() => removeMarkRow(i)}
                                    />
                                ))}
                                <Button startIcon={<AddIcon />} size="small" onClick={addMarkRow} sx={{ alignSelf: 'flex-start', fontWeight: 700 }}>
                                    Add mark type
                                </Button>

                                <Divider />

                                <Box sx={{ p: 2, borderRadius: 2, bgcolor: theme.palette.mode === 'dark' ? 'rgba(124,58,237,0.08)' : 'rgba(124,58,237,0.05)', border: '1px solid rgba(124,58,237,0.2)' }}>
                                    <Box display="flex" justifyContent="space-between">
                                        <Box><Typography variant="caption" color="text.secondary">Total Questions</Typography><Typography variant="h6" fontWeight={800}>{totalQuestions}</Typography></Box>
                                        <Box textAlign="right"><Typography variant="caption" color="text.secondary">Total Marks</Typography><Typography variant="h6" fontWeight={800}>{totalMarks}</Typography></Box>
                                    </Box>
                                </Box>

                                <Box display="flex" justifyContent="space-between" alignItems="center">
                                    <Box>
                                        <Typography variant="subtitle2" fontWeight={800}>Secure Centum Mode</Typography>
                                        <Typography variant="caption" color="text.secondary">Full syllabus coverage</Typography>
                                    </Box>
                                    <Switch checked={secureCentumMode} onChange={e => setSecureCentumMode(e.target.checked)} color="primary" />
                                </Box>

                                <Button
                                    variant="contained" fullWidth size="large"
                                    startIcon={generating ? <CircularProgress size={20} color="inherit" /> : <AutoAwesomeIcon />}
                                    onClick={handleGenerate}
                                    disabled={generating || !activeSyllabusId}
                                    sx={{ borderRadius: 2, py: 1.5, fontWeight: 800, background: 'linear-gradient(135deg, #2563EB, #7c3aed)', boxShadow: '0 8px 24px rgba(19,127,236,0.35)' }}
                                >
                                    {generating ? 'AI Generating…' : 'Generate Questions'}
                                </Button>
                            </Stack>
                        </GlassCard>
                    </Stack>
                </Grid>

                {/* RIGHT PANEL: Questions */}
                <Grid item xs={12} xl={8}>
                    <GlassCard sx={{ p: 3, minHeight: 600 }}>
                        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3} flexWrap="wrap" gap={2}>
                            <Box>
                                <Typography variant="h5" fontWeight={900}>High-Yield Question Bank</Typography>
                                <Typography variant="body2" color="text.secondary">
                                    {questions.length > 0 ? `${questions.length} questions · ${questions.reduce((s,q)=>s+q.marks,0)} total marks` : 'Generate questions from your syllabus'}
                                </Typography>
                            </Box>
                            {questions.length > 0 && (
                                <Box sx={{ p: 0.5, bgcolor: 'background.default', borderRadius: 2, border: '1px solid', borderColor: 'divider', display: 'flex', gap: 0.5 }}>
                                    {['all', 'frequent', 'high'].map(tab => (
                                        <Button key={tab} size="small"
                                            variant={filterTab === tab ? 'contained' : 'text'}
                                            onClick={() => setFilterTab(tab)}
                                            sx={{ borderRadius: 1.5, boxShadow: 'none', fontWeight: 700, fontSize: '0.75rem', textTransform: 'capitalize' }}>
                                            {tab === 'all' ? 'All' : tab === 'frequent' ? 'High-Yield' : 'Long Ans.'}
                                        </Button>
                                    ))}
                                </Box>
                            )}
                        </Box>

                        {generating && (
                            <Box p={6} display="flex" flexDirection="column" alignItems="center" gap={2}>
                                <CircularProgress size={52} thickness={4} />
                                <Typography variant="h6" fontWeight={800}>AI is analysing your syllabus…</Typography>
                                <Typography variant="body2" color="text.secondary">Generating questions with model answers. This may take a minute.</Typography>
                            </Box>
                        )}
                        {!generating && fetchingQuestions && <Box p={4} display="flex" justifyContent="center"><CircularProgress /></Box>}
                        {!generating && !fetchingQuestions && questions.length === 0 && (
                            <Box p={6} textAlign="center">
                                <AnalyticsIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2, opacity: 0.4 }} />
                                <Typography variant="h6" fontWeight={800} gutterBottom>No Questions Yet</Typography>
                                <Typography variant="body2" color="text.secondary">Upload your syllabus, configure the exam pattern, and click "Generate Questions".</Typography>
                            </Box>
                        )}

                        {!generating && filteredQuestions.length > 0 && (
                            <Stack spacing={2}>
                                {filteredQuestions.map((q, i) => <QuestionCard key={q.id || i} q={q} index={i} />)}
                            </Stack>
                        )}
                    </GlassCard>
                </Grid>
            </Grid>
        </Box>
    );
};

// =====================================================
// VIEW 2: AI Strategy Roadmap (Table layout, exam-focused)
// =====================================================
const StrategyRoadmapView = ({ syllabi, activeSyllabusId, setActiveSyllabusId }) => {
    const theme = useTheme();
    const [strategy, setStrategy] = useState(null);
    const [loading, setLoading] = useState(false);
    const nextTwoWeeks = new Date();
    nextTwoWeeks.setDate(nextTwoWeeks.getDate() + 14);
    const [examDate, setExamDate] = useState(nextTwoWeeks.toISOString().split('T')[0]);
    const [hours, setHours] = useState(4);
    const [snack, setSnack] = useState({ open: false, msg: '', severity: 'success' });

    const activeSyllabus = syllabi.find(s => s.id === activeSyllabusId) || syllabi[0];

    const daysRemaining = Math.max(1, Math.ceil((new Date(examDate) - new Date()) / (1000 * 60 * 60 * 24)));

    const generateRoadmap = async () => {
        if (!activeSyllabusId) {
            setSnack({ open: true, msg: 'Select a syllabus first.', severity: 'warning' });
            return;
        }
        setLoading(true);
        try {
            const res = await API.post(`/exam/syllabus/${activeSyllabusId}/strategy/`, {
                days_remaining: daysRemaining,
                hours_per_day: parseInt(hours, 10) || 4
            });
            setStrategy(res.data.strategy || []);
        } catch (err) {
            console.error(err);
            setSnack({ open: true, msg: 'Failed to generate roadmap. Please try again.', severity: 'error' });
        } finally {
            setLoading(false);
        }
    };

    const getTaskIcon = (type) => {
        if (type === 'break') return <CoffeeIcon sx={{ fontSize: 14 }} />;
        return <FitnessCenterIcon sx={{ fontSize: 14 }} />;
    };

    const getTaskColor = (type, theme) =>
        type === 'break'
            ? (theme.palette.mode === 'dark' ? 'rgba(255,152,0,0.12)' : 'rgba(255,152,0,0.08)')
            : (theme.palette.mode === 'dark' ? 'rgba(19,127,236,0.1)' : 'rgba(19,127,236,0.06)');

    return (
        <Box sx={{ maxWidth: 1400, mx: 'auto', p: { xs: 2, md: 4 } }}>
            <Snackbar open={snack.open} autoHideDuration={4000} onClose={() => setSnack(s => ({ ...s, open: false }))}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
                <Alert severity={snack.severity} variant="filled" onClose={() => setSnack(s => ({ ...s, open: false }))} sx={{ borderRadius: 2, fontWeight: 700 }}>
                    {snack.msg}
                </Alert>
            </Snackbar>

            <Box mb={4}>
                <Typography variant="h3" fontWeight={900} sx={{ letterSpacing: '-0.02em' }}>AI Exam Strategy Roadmap</Typography>
                <Typography variant="body1" color="text.secondary" mt={1}>
                    Get a personalised day-by-day exam preparation schedule based on your syllabus topics, exam date, and daily commitment.
                </Typography>
            </Box>

            {/* Config Card */}
            <GlassCard sx={{ p: 3, mb: 4 }}>
                <Grid container spacing={3} alignItems="flex-end">
                    <Grid item xs={12} md={4}>
                        <Typography variant="caption" fontWeight={800} color="text.secondary" display="block" gutterBottom>SYLLABUS</Typography>
                        <FormControl fullWidth size="small">
                            <Select
                                value={activeSyllabusId || ''}
                                displayEmpty
                                onChange={e => setActiveSyllabusId(e.target.value)}
                                sx={{ borderRadius: 2 }}
                            >
                                <MenuItem value="" disabled><em>Select a syllabus…</em></MenuItem>
                                {syllabi.map(s => <MenuItem key={s.id} value={s.id}>{s.title}</MenuItem>)}
                            </Select>
                        </FormControl>
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Typography variant="caption" fontWeight={800} color="text.secondary" display="block" gutterBottom>EXAM DATE</Typography>
                        <TextField type="date" fullWidth size="small" value={examDate} onChange={e => setExamDate(e.target.value)}
                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                    </Grid>
                    <Grid item xs={12} md={2}>
                        <Typography variant="caption" fontWeight={800} color="text.secondary" display="block" gutterBottom>HOURS / DAY</Typography>
                        <TextField type="number" fullWidth size="small" value={hours} onChange={e => setHours(e.target.value)}
                            InputProps={{ inputProps: { min: 1, max: 16 }, endAdornment: <InputAdornment position="end">hrs</InputAdornment> }}
                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2 } }} />
                    </Grid>
                    <Grid item xs={12} md={3}>
                        <Box sx={{ p: 2, borderRadius: 2, bgcolor: theme.palette.mode === 'dark' ? 'rgba(19,127,236,0.08)' : 'rgba(19,127,236,0.05)', border: '1px solid', borderColor: 'rgba(19,127,236,0.2)', mb: 1 }}>
                            <Typography variant="caption" color="text.secondary">Days remaining: <strong>{daysRemaining}</strong> · Total study: <strong>{daysRemaining * hours}h</strong></Typography>
                        </Box>
                        <Button fullWidth variant="contained" size="large"
                            startIcon={loading ? <CircularProgress size={16} color="inherit" /> : <AutoAwesomeIcon />}
                            onClick={generateRoadmap} disabled={loading || !activeSyllabusId}
                            sx={{ borderRadius: 2, fontWeight: 800, background: 'linear-gradient(135deg, #2563EB, #7c3aed)' }}>
                            {strategy ? 'Re-Generate' : 'Generate Roadmap'}
                        </Button>
                    </Grid>
                </Grid>
            </GlassCard>

            {/* Loading State */}
            {loading && (
                <GlassCard sx={{ p: 8, textAlign: 'center' }}>
                    <CircularProgress size={56} thickness={4} sx={{ mb: 3 }} />
                    <Typography variant="h6" fontWeight={800} gutterBottom>Building your personalised exam plan…</Typography>
                    <Typography variant="body2" color="text.secondary">AI is analysing your syllabus and allocating topics optimally.</Typography>
                </GlassCard>
            )}

            {/* Empty State */}
            {!loading && !strategy && (
                <GlassCard sx={{ p: 8, textAlign: 'center' }}>
                    <ScheduleIcon sx={{ fontSize: 72, color: 'text.secondary', opacity: 0.35, mb: 2 }} />
                    <Typography variant="h6" fontWeight={800} gutterBottom>No Roadmap Yet</Typography>
                    <Typography variant="body2" color="text.secondary">Configure your exam details above and click "Generate Roadmap" to get started.</Typography>
                </GlassCard>
            )}

            {/* INLINE TABLE SCHEDULE */}
            {!loading && strategy && strategy.length > 0 && (
                <Stack spacing={4}>
                    {/* Summary Bar */}
                    <Box display="flex" gap={3} flexWrap="wrap">
                        {[
                            { label: 'Total Days', value: strategy.length, color: 'primary.main' },
                            { label: 'Study Hours', value: `${strategy.reduce((s, d) => s + (d.tasks || []).filter(t => t.type !== 'break').reduce((a, t) => a + parseFloat(t.duration) || 0, 0), 0).toFixed(0)}h`, color: '#10b981' },
                            { label: 'Topics Covered', value: [...new Set(strategy.flatMap(d => (d.tasks || []).map(t => t.main_topic).filter(Boolean)))].length, color: '#7c3aed' },
                        ].map(stat => (
                            <GlassCard key={stat.label} sx={{ p: 2.5, flex: '1 0 140px', textAlign: 'center' }}>
                                <Typography variant="h4" fontWeight={900} sx={{ color: stat.color }}>{stat.value}</Typography>
                                <Typography variant="caption" color="text.secondary" fontWeight={700}>{stat.label}</Typography>
                            </GlassCard>
                        ))}
                    </Box>

                    {/* Day-by-day TABLE */}
                    {strategy.map((day, dayIdx) => (
                        <GlassCard key={dayIdx} sx={{ overflow: 'hidden', border: dayIdx === 0 ? '2px solid' : '1px solid', borderColor: dayIdx === 0 ? 'primary.main' : 'divider' }}>
                            {/* Day Header */}
                            <Box sx={{
                                px: 3, py: 2,
                                bgcolor: dayIdx === 0 ? 'rgba(19,127,236,0.12)' : (theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'),
                                borderBottom: '1px solid', borderColor: 'divider',
                                display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 2
                            }}>
                                <Box display="flex" alignItems="center" gap={2}>
                                    <Avatar sx={{ bgcolor: dayIdx === 0 ? 'primary.main' : 'text.disabled', fontWeight: 800, width: 36, height: 36, fontSize: '0.85rem' }}>
                                        D{day.day || dayIdx + 1}
                                    </Avatar>
                                    <Box>
                                        <Typography variant="subtitle1" fontWeight={800}>Day {day.day || dayIdx + 1}</Typography>
                                        <Typography variant="caption" color={dayIdx === 0 ? 'primary.main' : 'text.secondary'} fontWeight={600}>
                                            {day.focus || 'Study Session'}
                                        </Typography>
                                    </Box>
                                </Box>
                                <Box display="flex" gap={1}>
                                    <Chip label={`${hours} hrs`} size="small" icon={<ScheduleIcon sx={{ fontSize: 14 }} />}
                                        sx={{ bgcolor: 'background.default', fontWeight: 700, borderRadius: 1 }} />
                                    {dayIdx === 0 && <Chip label="Today" size="small" color="primary" sx={{ fontWeight: 800, borderRadius: 1 }} />}
                                </Box>
                            </Box>

                            {/* Inline Table */}
                            <TableContainer>
                                <Table size="small" sx={{ tableLayout: 'fixed' }}>
                                    <TableHead>
                                        <TableRow sx={{ bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.02)' }}>
                                            <TableCell sx={{ fontWeight: 800, fontSize: '0.75rem', color: 'text.secondary', width: '12%' }}>DURATION</TableCell>
                                            <TableCell sx={{ fontWeight: 800, fontSize: '0.75rem', color: 'text.secondary', width: '22%' }}>TOPIC</TableCell>
                                            <TableCell sx={{ fontWeight: 800, fontSize: '0.75rem', color: 'text.secondary', width: '54%' }}>SUBTOPICS / FOCUS AREAS</TableCell>
                                            <TableCell sx={{ fontWeight: 800, fontSize: '0.75rem', color: 'text.secondary', width: '12%', textAlign: 'center' }}>TYPE</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>
                                        {(day.tasks || []).map((task, taskIdx) => (
                                            <TableRow key={taskIdx} sx={{
                                                bgcolor: task.type === 'break'
                                                    ? (theme.palette.mode === 'dark' ? 'rgba(255,152,0,0.05)' : 'rgba(255,152,0,0.04)')
                                                    : 'transparent',
                                                '&:last-child td': { border: 0 },
                                                transiton: 'all 0.2s',
                                                '&:hover': { bgcolor: 'action.hover' }
                                            }}>
                                                <TableCell>
                                                    <Typography variant="body2" fontWeight={700} color={task.type === 'break' ? 'warning.main' : 'primary.main'}>
                                                        {task.duration}
                                                    </Typography>
                                                </TableCell>
                                                <TableCell>
                                                    <Typography variant="body2" fontWeight={700}>{task.main_topic}</Typography>
                                                </TableCell>
                                                <TableCell>
                                                    <Box display="flex" gap={0.5} flexWrap="wrap">
                                                        {(task.subtopics || []).map((st, si) => (
                                                            <Chip key={si} label={st} size="small" sx={{
                                                                height: 22, fontSize: '0.7rem', fontWeight: 600, borderRadius: 1,
                                                                bgcolor: task.type === 'break'
                                                                    ? 'rgba(255,152,0,0.1)'
                                                                    : (theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)'),
                                                                color: task.type === 'break' ? 'warning.main' : 'text.secondary'
                                                            }} />
                                                        ))}
                                                    </Box>
                                                </TableCell>
                                                <TableCell sx={{ textAlign: 'center' }}>
                                                    <Chip
                                                        icon={getTaskIcon(task.type)}
                                                        label={task.type === 'break' ? 'Break' : 'Study'}
                                                        size="small"
                                                        sx={{
                                                            height: 24, fontSize: '0.7rem', fontWeight: 700, borderRadius: 1,
                                                            bgcolor: task.type === 'break' ? 'rgba(255,152,0,0.15)' : 'rgba(19,127,236,0.12)',
                                                            color: task.type === 'break' ? 'warning.main' : 'primary.main',
                                                            '& .MuiChip-icon': { color: 'inherit' }
                                                        }}
                                                    />
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        </GlassCard>
                    ))}
                </Stack>
            )}
        </Box>
    );
};

// =====================================================
// MAIN COMPONENT
// =====================================================
export default function ExamPreparation() {
    const [view, setView] = useState('prep');
    const [syllabi, setSyllabi] = useState([]);
    const [activeSyllabusId, setActiveSyllabusId] = useState(null);

    useEffect(() => {
        API.get('/exam/syllabi/')
            .then(res => {
                const list = res.data.syllabi || [];
                setSyllabi(list);
                if (list.length > 0) setActiveSyllabusId(list[0].id);
            })
            .catch(err => console.error('Failed to fetch syllabi', err));
    }, []);

    const handleSyllabusUploaded = (newSyllabus) => {
        setSyllabi(prev => [newSyllabus, ...prev]);
        setActiveSyllabusId(newSyllabus.id);
    };

    return (
        <Box sx={{ minHeight: '100vh', pb: 10 }}>
            {/* Tab Navigation */}
            <Box sx={{
                borderBottom: '1px solid', borderColor: 'divider',
                bgcolor: 'background.paper', px: 4, pt: 2, mb: 0,
                display: 'flex', alignItems: 'center', gap: 4,
                position: 'sticky', top: 0, zIndex: 50
            }}>
                {[
                    { id: 'prep', label: 'Question Bank Generator', icon: <AnalyticsIcon sx={{ fontSize: 18 }} /> },
                    { id: 'roadmap', label: 'AI Strategy Roadmap', icon: <ScheduleIcon sx={{ fontSize: 18 }} /> }
                ].map(tab => (
                    <Button
                        key={tab.id}
                        startIcon={tab.icon}
                        onClick={() => setView(tab.id)}
                        sx={{
                            pb: 2, pt: 1, px: 1.5,
                            fontWeight: view === tab.id ? 800 : 600,
                            color: view === tab.id ? 'primary.main' : 'text.secondary',
                            borderBottom: '3px solid',
                            borderColor: view === tab.id ? 'primary.main' : 'transparent',
                            borderRadius: 0,
                            '&:hover': { bgcolor: 'transparent', color: 'text.primary' }
                        }}
                    >
                        {tab.label}
                    </Button>
                ))}
            </Box>

            <Box mt={4}>
                {view === 'prep' ? (
                    <QuestionBankView
                        syllabi={syllabi}
                        activeSyllabusId={activeSyllabusId}
                        setActiveSyllabusId={setActiveSyllabusId}
                        onSyllabusUploaded={handleSyllabusUploaded}
                    />
                ) : (
                    <StrategyRoadmapView
                        syllabi={syllabi}
                        activeSyllabusId={activeSyllabusId}
                        setActiveSyllabusId={setActiveSyllabusId}
                    />
                )}
            </Box>
        </Box>
    );
}
