// src/pages/Lectures.js
import React, { useState, useEffect, useCallback } from 'react';
import { 
  Box, Typography, Card, CardContent, IconButton, 
  Button, Chip, CircularProgress, Alert, 
  Dialog, DialogActions, DialogContent, DialogTitle,
  Grid, Paper, TextField, Stack, Tab, Tabs, InputAdornment,
  Divider, Tooltip, Fade, LinearProgress
} from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { 
  Delete as DeleteIcon, 
  Search as SearchIcon,
  CloudUpload as CloudUploadIcon,
  AutoAwesome as AutoAwesomeIcon,
  Description as DescriptionIcon,
  CalendarToday as CalendarIcon,
  CheckCircle as CheckCircleIcon,
  Close as CloseIcon,
  Quiz as QuizIcon,
  PictureAsPdf as PdfIcon,
  Image as ImageIcon,
  VideoFile as VideoIcon,
  AudioFile as AudioIcon,
  InsertDriveFile as FileIcon,
  Article as ArticleIcon,
  LibraryBooks as LibraryIcon,
  Visibility as PreviewIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  NavigateBefore as PrevIcon,
  NavigateNext as NextIcon,
  Download as DownloadIcon,
  OpenInNew as OpenInNewIcon,
  Notes as NotesIcon,
  Calculate as CalculateIcon,
  Lightbulb as LightbulbIcon,
  Refresh as RefreshIcon,
  FiberManualRecord as BulletIcon,
  StickyNote2 as StickyNote2Icon,
  Add as AddIcon,
  Edit as EditIcon,
  Save as SaveIcon,
  ColorLens as ColorLensIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useAuth } from '../context/AuthContext';
import API from '../api/api';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Set worker source
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

// --- Helper: Get File Type ---
const getFileType = (filename) => {
    if (!filename) return 'unknown';
    const ext = filename.split('.').pop().toLowerCase();
    if (ext === 'pdf') return 'pdf';
    if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'].includes(ext)) return 'image';
    if (['mp4', 'webm', 'ogg', 'mov'].includes(ext)) return 'video';
    if (['mp3', 'wav', 'aac'].includes(ext)) return 'audio';
    return 'unknown';
};

const getFileUrl = (file) => {
    if (!file) return null;
    return file.startsWith('http') ? file : `http://127.0.0.1:8000${file.startsWith('/') ? '' : '/'}${file}`;
};

// --- Markdown Renderer ---
const MarkdownContent = ({ children }) => (
    <Box
        sx={{
            '& h1, & h2, & h3, & h4': {
                fontWeight: 700,
                mt: 2,
                mb: 1,
                color: 'text.primary',
            },
            '& h2': { fontSize: '1.1rem', borderBottom: '1px solid', borderColor: 'divider', pb: 0.5 },
            '& h3': { fontSize: '1rem' },
            '& p': { lineHeight: 1.8, mb: 1, color: 'text.secondary' },
            '& ul, & ol': { pl: 2.5, mb: 1 },
            '& li': { mb: 0.5, color: 'text.secondary', lineHeight: 1.6 },
            '& strong': { fontWeight: 700, color: 'text.primary' },
            '& em': { fontStyle: 'italic' },
            '& code': {
                fontFamily: 'monospace',
                bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)',
                px: 0.75,
                py: 0.25,
                borderRadius: '4px',
                fontSize: '0.85em',
            },
            '& pre': {
                bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(0,0,0,0.3)' : 'rgba(0,0,0,0.04)',
                p: 2,
                borderRadius: 2,
                overflowX: 'auto',
                mb: 2,
            },
            '& blockquote': {
                borderLeft: '3px solid',
                borderColor: 'primary.main',
                pl: 2,
                my: 1.5,
                color: 'text.secondary',
            },
            '& table': { borderCollapse: 'collapse', width: '100%', mb: 2 },
            '& th, & td': { border: '1px solid', borderColor: 'divider', p: 1, textAlign: 'left' },
            '& th': { bgcolor: 'action.hover', fontWeight: 700 },
        }}
    >
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{children || ''}</ReactMarkdown>
    </Box>
);

// --- Full-Width PDF/File Viewer Component ---
const FileViewer = ({ lecture }) => {
    const [numPages, setNumPages] = useState(null);
    const [pageNumber, setPageNumber] = useState(1);
    const [scale, setScale] = useState(1.0);
    const [containerWidth, setContainerWidth] = useState(null);
    const containerRef = React.useRef(null);

    const fileType = getFileType(lecture?.file);
    const fileUrl = getFileUrl(lecture?.file);

    useEffect(() => {
        const updateWidth = () => {
            if (containerRef.current) setContainerWidth(containerRef.current.offsetWidth);
        };
        updateWidth();
        const observer = new ResizeObserver(updateWidth);
        if (containerRef.current) observer.observe(containerRef.current);
        return () => observer.disconnect();
    }, []);

    useEffect(() => {
        setPageNumber(1);
        setScale(1.0);
    }, [lecture?.file]);

    if (!lecture) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 2, py: 10 }}>
                <PreviewIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.3 }} />
                <Typography variant="h6" color="text.secondary">Select a lecture to preview</Typography>
                <Typography variant="body2" color="text.disabled">Click any lecture card to view its content here</Typography>
            </Box>
        );
    }

    if (!lecture.file) {
        return (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 2, py: 10 }}>
                <ArticleIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.3 }} />
                <Typography variant="h6" color="text.secondary">Text-based lecture</Typography>
                <Typography variant="body2" color="text.disabled" sx={{ maxWidth: 400, textAlign: 'center' }}>
                    {lecture.content ? lecture.content.slice(0, 300) + '...' : 'No preview available for this lecture type.'}
                </Typography>
            </Box>
        );
    }

    const renderContent = () => {
        switch (fileType) {
            case 'pdf':
                return (
                    <Document
                        file={fileUrl}
                        onLoadSuccess={({ numPages }) => { setNumPages(numPages); setPageNumber(1); }}
                        loading={<Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}><CircularProgress sx={{ color: 'white' }} /></Box>}
                        error={
                            <Box sx={{ textAlign: 'center', p: 3, color: 'white' }}>
                                <PdfIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
                                <Typography>Failed to load PDF</Typography>
                                <Button href={fileUrl} target="_blank" variant="outlined" sx={{ mt: 2, color: 'white', borderColor: 'white' }}>
                                    Open in Browser
                                </Button>
                            </Box>
                        }
                    >
                        <Page
                            pageNumber={pageNumber}
                            scale={scale}
                            renderTextLayer={true}
                            renderAnnotationLayer={false}
                            width={containerWidth ? (containerWidth - 48) * 0.9 : 800}
                        />
                    </Document>
                );
            case 'image':
                return (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', minHeight: 400 }}>
                        <img
                            src={fileUrl}
                            alt="Preview"
                            style={{
                                transform: `scale(${scale})`,
                                transformOrigin: 'top center',
                                transition: 'transform 0.2s',
                                maxWidth: '90%',
                                objectFit: 'contain',
                                borderRadius: 8,
                            }}
                        />
                    </Box>
                );
            case 'video':
                return (
                    <Box sx={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                        <video controls style={{ maxWidth: '90%', borderRadius: 12, boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
                            <source src={fileUrl} />
                        </video>
                    </Box>
                );
            case 'audio':
                return (
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, py: 8 }}>
                        <AudioIcon sx={{ fontSize: 120, color: 'primary.main', opacity: 0.8 }} />
                        <audio controls style={{ width: '80%' }}>
                            <source src={fileUrl} />
                        </audio>
                    </Box>
                );
            default:
                return (
                    <Box sx={{ textAlign: 'center', py: 10 }}>
                        <FileIcon sx={{ fontSize: 80, mb: 2, color: 'text.disabled' }} />
                        <Typography variant="h6" color="text.secondary" gutterBottom>Preview not available</Typography>
                        <Button variant="contained" href={fileUrl} download target="_blank" startIcon={<DownloadIcon />} sx={{ mt: 2 }}>
                            Download File
                        </Button>
                    </Box>
                );
        }
    };

    const showZoom = ['pdf', 'image'].includes(fileType);
    const showPageNav = fileType === 'pdf' && numPages;

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            {/* Viewer Toolbar */}
            <Box sx={{
                px: 3, py: 1.5,
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                borderBottom: '1px solid', borderColor: 'divider',
                bgcolor: 'background.paper',
                flexShrink: 0
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                    {fileType === 'pdf' && <PdfIcon sx={{ color: '#ef4444' }} />}
                    {fileType === 'image' && <ImageIcon sx={{ color: 'primary.main' }} />}
                    {fileType === 'video' && <VideoIcon sx={{ color: 'secondary.main' }} />}
                    {fileType === 'audio' && <AudioIcon sx={{ color: 'warning.main' }} />}
                    <Box>
                        <Typography variant="subtitle1" fontWeight={800} sx={{ lineHeight: 1.2 }}>
                            {lecture.title}
                        </Typography>
                        {showPageNav && (
                            <Typography variant="caption" color="text.secondary" fontWeight={600}>
                                Page {pageNumber} of {numPages}
                            </Typography>
                        )}
                    </Box>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {showPageNav && (
                        <>
                            <Tooltip title="Previous page">
                                <span>
                                    <IconButton size="small" disabled={pageNumber <= 1} onClick={() => setPageNumber(p => p - 1)}>
                                        <PrevIcon />
                                    </IconButton>
                                </span>
                            </Tooltip>
                            <Typography variant="caption" fontWeight={700} sx={{ px: 1, minWidth: 60, textAlign: 'center' }}>
                                {pageNumber} / {numPages}
                            </Typography>
                            <Tooltip title="Next page">
                                <span>
                                    <IconButton size="small" disabled={pageNumber >= numPages} onClick={() => setPageNumber(p => p + 1)}>
                                        <NextIcon />
                                    </IconButton>
                                </span>
                            </Tooltip>
                            <Divider orientation="vertical" flexItem sx={{ mx: 1, my: 0.5 }} />
                        </>
                    )}
                    {showZoom && (
                        <>
                            <Tooltip title="Zoom out">
                                <IconButton size="small" onClick={() => setScale(s => Math.max(0.4, +(s - 0.1).toFixed(1)))}>
                                    <ZoomOutIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Typography variant="caption" fontWeight={700} sx={{ minWidth: 48, textAlign: 'center' }}>
                                {Math.round(scale * 100)}%
                            </Typography>
                            <Tooltip title="Zoom in">
                                <IconButton size="small" onClick={() => setScale(s => Math.min(3.0, +(s + 0.1).toFixed(1)))}>
                                    <ZoomInIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Divider orientation="vertical" flexItem sx={{ mx: 1, my: 0.5 }} />
                        </>
                    )}
                    <Tooltip title="Open in new tab">
                        <IconButton size="small" component="a" href={fileUrl} target="_blank">
                            <OpenInNewIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>

            {/* Content Area */}
            <Box
                ref={containerRef}
                sx={{
                    flex: 1,
                    overflowY: 'auto',
                    p: 3,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    bgcolor: fileType === 'pdf' ? '#1e1e2e' : 'background.default',
                    '& .react-pdf__Page': {
                        boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
                        borderRadius: '4px',
                        overflow: 'hidden',
                        margin: '0 auto 16px',
                    }
                }}
            >
                {renderContent()}
            </Box>
        </Box>
    );
};

// --- Notes/Formula/KeyPoints Panel ---
const StudyAidsPanel = ({ lecture: initialLecture, lectureId }) => {
    const [lecture, setLecture] = useState(initialLecture);
    const [generating, setGenerating] = useState(false);
    const [error, setError] = useState('');
    const [activeTab, setActiveTab] = useState(0);

    // Keep in sync with parent
    useEffect(() => { setLecture(initialLecture); }, [initialLecture]);

    const hasContent = lecture?.study_notes || (lecture?.formulas?.length > 0) || (lecture?.key_points?.length > 0);

    const generateAids = async () => {
        setGenerating(true);
        setError('');
        try {
            const res = await API.post(`lectures/${lectureId}/generate-study-aids/`);
            setLecture(prev => ({
                ...prev,
                study_notes: res.data.study_notes,
                formulas: res.data.formulas,
                key_points: res.data.key_points,
            }));
        } catch (err) {
            const errData = err?.response?.data;
            setError(errData?.error || 'Failed to generate study aids. Please try again.');
        } finally {
            setGenerating(false);
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
                <Box>
                    <Typography variant="subtitle1" fontWeight={800}>AI Study Aids</Typography>
                    <Typography variant="caption" color="text.secondary">
                        {hasContent ? 'Study notes, formulas, and key points for this lecture' : 'Generate AI-powered study aids from this lecture PDF'}
                    </Typography>
                </Box>
                <Button
                    variant={hasContent ? 'outlined' : 'contained'}
                    size="small"
                    startIcon={generating ? <CircularProgress size={16} color="inherit" /> : <RefreshIcon />}
                    onClick={generateAids}
                    disabled={generating}
                    sx={{ borderRadius: 2, fontWeight: 700 }}
                >
                    {generating ? 'Generating...' : hasContent ? 'Regenerate' : 'Generate Now'}
                </Button>
            </Box>

            {generating && <LinearProgress color="primary" />}

            {error && (
                <Box sx={{ px: 3, pt: 2 }}>
                    <Alert severity="error" onClose={() => setError('')}>{error}</Alert>
                </Box>
            )}

            {!hasContent && !generating && !error && (
                <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', flex: 1, py: 6, px: 3, textAlign: 'center' }}>
                    <Box sx={{
                        width: 72, height: 72, borderRadius: '50%', mb: 2,
                        background: 'linear-gradient(135deg, rgba(3,140,127,0.1) 0%, rgba(2,115,115,0.2) 100%)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                        <AutoAwesomeIcon sx={{ fontSize: 36, color: 'primary.main' }} />
                    </Box>
                    <Typography variant="h6" fontWeight={700} gutterBottom>No Study Aids Yet</Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 320, mb: 3 }}>
                        Click "Generate Now" to extract study notes, formulas, and key points from this lecture's PDF using AI.
                    </Typography>
                    <Button variant="contained" onClick={generateAids} startIcon={<AutoAwesomeIcon />} sx={{ borderRadius: 2 }}>
                        Generate Study Aids
                    </Button>
                </Box>
            )}

            {hasContent && (
                <>
                    {/* Sub-tabs */}
                    <Box sx={{ px: 3, borderBottom: '1px solid', borderColor: 'divider', flexShrink: 0 }}>
                        <Tabs value={activeTab} onChange={(_, v) => setActiveTab(v)} sx={{ '& .MuiTab-root': { fontWeight: 700, minHeight: 48, fontSize: '0.85rem' } }}>
                            <Tab icon={<NotesIcon fontSize="small" />} iconPosition="start" label="Notes" />
                            <Tab icon={<CalculateIcon fontSize="small" />} iconPosition="start" label={`Formulas${lecture.formulas?.length ? ` (${lecture.formulas.length})` : ''}`} />
                            <Tab icon={<LightbulbIcon fontSize="small" />} iconPosition="start" label={`Key Points${lecture.key_points?.length ? ` (${lecture.key_points.length})` : ''}`} />
                        </Tabs>
                    </Box>

                    <Box sx={{ flex: 1, overflowY: 'auto', p: 3 }}>
                        {/* Notes Tab */}
                        {activeTab === 0 && (
                            <Box>
                                {lecture.study_notes ? (
                                    <MarkdownContent>{lecture.study_notes}</MarkdownContent>
                                ) : (
                                    <Alert severity="info" variant="outlined">No study notes generated yet.</Alert>
                                )}
                            </Box>
                        )}

                        {/* Formulas Tab */}
                        {activeTab === 1 && (
                            <Stack spacing={2}>
                                {lecture.formulas && lecture.formulas.length > 0 ? (
                                    lecture.formulas.map((f, idx) => (
                                        <Paper
                                            key={idx}
                                            elevation={0}
                                            sx={{
                                                borderRadius: '12px',
                                                border: '1px solid',
                                                borderColor: 'divider',
                                                overflow: 'hidden',
                                            }}
                                        >
                                            <Box sx={{
                                                px: 2.5, py: 1.5,
                                                background: (theme) => theme.palette.mode === 'dark'
                                                    ? 'rgba(3,140,127,0.12)'
                                                    : 'linear-gradient(135deg, rgba(3,140,127,0.06) 0%, rgba(2,115,115,0.06) 100%)',
                                                borderBottom: '1px solid', borderColor: 'divider',
                                                display: 'flex', alignItems: 'center', gap: 1,
                                            }}>
                                                <CalculateIcon sx={{ fontSize: 18, color: 'primary.main' }} />
                                                <Typography variant="subtitle2" fontWeight={800} color="primary.main">
                                                    {f.name || `Formula ${idx + 1}`}
                                                </Typography>
                                            </Box>
                                            <Box sx={{ px: 2.5, py: 2 }}>
                                                <Paper
                                                    elevation={0}
                                                    sx={{
                                                        px: 2, py: 1.5, mb: 1.5,
                                                        fontFamily: 'monospace',
                                                        fontSize: '1rem',
                                                        fontWeight: 700,
                                                        letterSpacing: '0.025em',
                                                        bgcolor: (theme) => theme.palette.mode === 'dark'
                                                            ? 'rgba(255,255,255,0.06)'
                                                            : 'rgba(0,0,0,0.03)',
                                                        borderRadius: '8px',
                                                        borderLeft: '3px solid',
                                                        borderColor: 'primary.main',
                                                        color: 'text.primary',
                                                    }}
                                                >
                                                    {f.formula || '—'}
                                                </Paper>
                                                {f.description && (
                                                    <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                                                        {f.description}
                                                    </Typography>
                                                )}
                                            </Box>
                                        </Paper>
                                    ))
                                ) : (
                                    <Alert severity="info" variant="outlined">No formulas extracted yet.</Alert>
                                )}
                            </Stack>
                        )}

                        {/* Key Points Tab */}
                        {activeTab === 2 && (
                            <Stack spacing={1.5}>
                                {lecture.key_points && lecture.key_points.length > 0 ? (
                                    lecture.key_points.map((pt, idx) => (
                                        <Paper
                                            key={idx}
                                            elevation={0}
                                            sx={{
                                                p: 2,
                                                borderRadius: '12px',
                                                border: '1px solid',
                                                borderColor: 'divider',
                                                display: 'flex',
                                                alignItems: 'flex-start',
                                                gap: 1.5,
                                                transition: 'all 0.15s',
                                                '&:hover': {
                                                    borderColor: 'primary.main',
                                                    bgcolor: (theme) => theme.palette.mode === 'dark'
                                                        ? 'rgba(3,140,127,0.05)'
                                                        : 'rgba(3,140,127,0.03)',
                                                    transform: 'translateX(4px)',
                                                },
                                            }}
                                        >
                                            <Box sx={{
                                                width: 28, height: 28, minWidth: 28,
                                                borderRadius: '8px',
                                                background: 'linear-gradient(135deg, #038C7F 0%, #027373 100%)',
                                                color: 'white',
                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                fontWeight: 800, fontSize: '13px', mt: 0.1,
                                            }}>
                                                {idx + 1}
                                            </Box>
                                            <Typography variant="body2" sx={{ lineHeight: 1.7, color: 'text.primary', flex: 1 }}>
                                                {pt}
                                            </Typography>
                                        </Paper>
                                    ))
                                ) : (
                                    <Alert severity="info" variant="outlined">No key points extracted yet.</Alert>
                                )}
                            </Stack>
                        )}
                    </Box>
                </>
            )}
        </Box>
    );
};

// ─── Class Notes (Sticky Notes) Panel ───────────────────────────────────────
const NOTE_COLORS = [
    { hex: '#FFF9C4', label: 'Yellow' },
    { hex: '#BBDEFB', label: 'Blue' },
    { hex: '#C8E6C9', label: 'Green' },
    { hex: '#FFCCBC', label: 'Peach' },
    { hex: '#E1BEE7', label: 'Purple' },
    { hex: '#F8BBD9', label: 'Pink' },
    { hex: '#B2EBF2', label: 'Cyan' },
    { hex: '#FFE0B2', label: 'Orange' },
];

const ClassNotesPanel = ({ lectureId }) => {
    const [notes, setNotes] = useState([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [editingId, setEditingId] = useState(null);
    // New note form state
    const [newTitle, setNewTitle] = useState('');
    const [newContent, setNewContent] = useState('');
    const [newColor, setNewColor] = useState('#FFF9C4');
    // Edit state
    const [editTitle, setEditTitle] = useState('');
    const [editContent, setEditContent] = useState('');
    const [editColor, setEditColor] = useState('#FFF9C4');
    const theme = useTheme();

    const fetchNotes = useCallback(async () => {
        setLoading(true);
        try {
            const res = await API.get(`/sticky-notes/?lecture_id=${lectureId}`);
            setNotes(res.data || []);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    }, [lectureId]);

    useEffect(() => { fetchNotes(); }, [fetchNotes]);

    const handleCreate = async () => {
        if (!newContent.trim()) return;
        try {
            const res = await API.post('/sticky-notes/', {
                title: newTitle.trim() || 'Class Note',
                content: newContent,
                color: newColor,
                lecture_note_id: lectureId,
            });
            setNotes(prev => [res.data, ...prev]);
            setNewTitle(''); setNewContent(''); setNewColor('#FFF9C4');
            setCreating(false);
        } catch (e) { console.error(e); }
    };

    const handleStartEdit = (note) => {
        setEditingId(note.id);
        setEditTitle(note.title);
        setEditContent(note.content);
        setEditColor(note.color);
    };

    const handleSaveEdit = async (noteId) => {
        try {
            const res = await API.put(`/sticky-notes/${noteId}/`, {
                title: editTitle,
                content: editContent,
                color: editColor,
            });
            setNotes(prev => prev.map(n => n.id === noteId ? { ...n, ...res.data } : n));
            setEditingId(null);
        } catch (e) { console.error(e); }
    };

    const handleDelete = async (noteId) => {
        try {
            await API.delete(`/sticky-notes/${noteId}/`);
            setNotes(prev => prev.filter(n => n.id !== noteId));
        } catch (e) { console.error(e); }
    };

    const isDark = theme.palette.mode === 'dark';

    // Color picker row
    const ColorPicker = ({ value, onChange }) => (
        <Box display="flex" gap={0.75} flexWrap="wrap" mt={1}>
            {NOTE_COLORS.map(c => (
                <Tooltip key={c.hex} title={c.label}>
                    <Box
                        onClick={() => onChange(c.hex)}
                        sx={{
                            width: 22, height: 22, borderRadius: '50%',
                            bgcolor: c.hex,
                            border: value === c.hex ? '3px solid' : '2px solid transparent',
                            borderColor: value === c.hex ? 'primary.main' : 'transparent',
                            cursor: 'pointer',
                            outline: value === c.hex ? '1px solid' : 'none',
                            outlineColor: 'primary.main',
                            transition: 'transform 0.15s',
                            '&:hover': { transform: 'scale(1.2)' },
                        }}
                    />
                </Tooltip>
            ))}
        </Box>
    );

    return (
        <Box sx={{ p: 3, height: '100%', overflowY: 'auto' }}>
            {/* Header */}
            <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
                <Box display="flex" alignItems="center" gap={1}>
                    <StickyNote2Icon sx={{ color: '#f59e0b' }} />
                    <Typography variant="subtitle1" fontWeight={700}>Class Notes</Typography>
                    {notes.length > 0 && (
                        <Chip label={`${notes.length} note${notes.length > 1 ? 's' : ''}`} size="small" sx={{ bgcolor: 'rgba(245,158,11,0.12)', color: '#f59e0b', fontWeight: 700 }} />
                    )}
                </Box>
                <Button
                    size="small" variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setCreating(true)}
                    sx={{ borderRadius: 2, fontWeight: 700, bgcolor: '#f59e0b', '&:hover': { bgcolor: '#d97706' } }}
                >
                    New Note
                </Button>
            </Box>

            {/* New Note Form */}
            {creating && (
                <Paper elevation={0} sx={{
                    mb: 3, p: 2.5, borderRadius: 2,
                    border: '2px solid',
                    borderColor: 'primary.main',
                    bgcolor: isDark ? 'rgba(30,40,50,0.9)' : '#fff',
                }}>
                    <Typography variant="caption" fontWeight={800} color="primary.main" sx={{ letterSpacing: '0.08em', textTransform: 'uppercase', display: 'block', mb: 1.5 }}>
                        New Class Note
                    </Typography>
                    <TextField
                        fullWidth size="small" placeholder="Note title (e.g. Chapter 3 Definitions)"
                        value={newTitle} onChange={e => setNewTitle(e.target.value)}
                        sx={{ mb: 1.5, '& .MuiOutlinedInput-root': { borderRadius: 1.5 } }}
                    />
                    <TextField
                        fullWidth multiline rows={5}
                        placeholder="Paste definitions, hint points, formulas...&#10;&#10;Tip: You can use markdown! **bold**, *italic*, `code`, - bullet lists"
                        value={newContent} onChange={e => setNewContent(e.target.value)}
                        sx={{ mb: 1.5, '& .MuiOutlinedInput-root': { borderRadius: 1.5, fontFamily: 'monospace', fontSize: '0.9rem' } }}
                    />
                    <Box display="flex" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1}>
                        <Box display="flex" alignItems="center" gap={1}>
                            <ColorLensIcon sx={{ fontSize: 18, color: 'text.secondary' }} />
                            <Typography variant="caption" color="text.secondary" fontWeight={700}>Note Color:</Typography>
                            <ColorPicker value={newColor} onChange={setNewColor} />
                        </Box>
                        <Box display="flex" gap={1}>
                            <Button size="small" onClick={() => { setCreating(false); setNewContent(''); setNewTitle(''); }} color="inherit" sx={{ borderRadius: 2 }}>Cancel</Button>
                            <Button size="small" variant="contained" startIcon={<SaveIcon />} onClick={handleCreate} disabled={!newContent.trim()}
                                sx={{ borderRadius: 2, fontWeight: 700 }}>Save Note</Button>
                        </Box>
                    </Box>
                </Paper>
            )}

            {/* Loading */}
            {loading && <Box display="flex" justifyContent="center" py={6}><CircularProgress /></Box>}

            {/* Empty state */}
            {!loading && notes.length === 0 && !creating && (
                <Box sx={{ textAlign: 'center', py: 6 }}>
                    <StickyNote2Icon sx={{ fontSize: 64, color: '#f59e0b', opacity: 0.35, mb: 2 }} />
                    <Typography variant="h6" fontWeight={800} gutterBottom>No Class Notes Yet</Typography>
                    <Typography variant="body2" color="text.secondary" mb={2}>
                        Create personal notes, paste important definitions, hint points, and key formulas from this lecture.
                    </Typography>
                    <Button variant="outlined" startIcon={<AddIcon />} onClick={() => setCreating(true)}
                        sx={{ borderRadius: 2, fontWeight: 700, borderColor: '#f59e0b', color: '#f59e0b' }}>
                        Create First Note
                    </Button>
                </Box>
            )}

            {/* Notes Grid */}
            {!loading && notes.length > 0 && (
                <Grid container spacing={2}>
                    {notes.map(note => (
                        <Grid item xs={12} sm={6} key={note.id}>
                            {editingId === note.id ? (
                                /* Edit Mode */
                                <Paper elevation={0} sx={{
                                    p: 2, borderRadius: 2, border: '2px solid',
                                    borderColor: 'primary.main',
                                    bgcolor: isDark ? 'rgba(30,40,55,0.95)' : '#fff',
                                }}>
                                    <TextField
                                        fullWidth size="small" value={editTitle}
                                        onChange={e => setEditTitle(e.target.value)}
                                        placeholder="Note title"
                                        sx={{ mb: 1, '& .MuiOutlinedInput-root': { borderRadius: 1.5 } }}
                                    />
                                    <TextField
                                        fullWidth multiline rows={5} value={editContent}
                                        onChange={e => setEditContent(e.target.value)}
                                        sx={{ mb: 1, '& .MuiOutlinedInput-root': { borderRadius: 1.5, fontFamily: 'monospace', fontSize: '0.88rem' } }}
                                    />
                                    <ColorPicker value={editColor} onChange={setEditColor} />
                                    <Box display="flex" justifyContent="flex-end" gap={1} mt={1.5}>
                                        <Button size="small" onClick={() => setEditingId(null)} color="inherit">Cancel</Button>
                                        <Button size="small" variant="contained" startIcon={<SaveIcon />}
                                            onClick={() => handleSaveEdit(note.id)} sx={{ borderRadius: 2, fontWeight: 700 }}>Save</Button>
                                    </Box>
                                </Paper>
                            ) : (
                                /* View Mode */
                                <Paper elevation={0} sx={{
                                    borderRadius: 2,
                                    border: '1px solid',
                                    borderColor: 'divider',
                                    overflow: 'hidden',
                                    height: '100%',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    transition: 'box-shadow 0.2s',
                                    '&:hover': { boxShadow: '0 4px 16px rgba(0,0,0,0.12)' },
                                }}>
                                    {/* Colored header strip */}
                                    <Box sx={{ bgcolor: note.color, px: 2, py: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                        <Typography variant="subtitle2" fontWeight={800} sx={{ color: 'rgba(0,0,0,0.75)', flex: 1, mr: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                            {note.title}
                                        </Typography>
                                        <Box display="flex" gap={0.5}>
                                            <Tooltip title="Edit">
                                                <IconButton size="small" onClick={() => handleStartEdit(note)} sx={{ p: 0.5, color: 'rgba(0,0,0,0.5)', '&:hover': { color: 'rgba(0,0,0,0.8)' } }}>
                                                    <EditIcon sx={{ fontSize: 16 }} />
                                                </IconButton>
                                            </Tooltip>
                                            <Tooltip title="Delete">
                                                <IconButton size="small" onClick={() => handleDelete(note.id)} sx={{ p: 0.5, color: 'rgba(0,0,0,0.5)', '&:hover': { color: '#ef4444' } }}>
                                                    <DeleteIcon sx={{ fontSize: 16 }} />
                                                </IconButton>
                                            </Tooltip>
                                        </Box>
                                    </Box>
                                    {/* Content */}
                                    <Box sx={{ p: 2, flex: 1, overflowY: 'auto', maxHeight: 260 }}>
                                        <Box sx={{
                                            fontSize: '0.875rem',
                                            lineHeight: 1.7,
                                            color: 'text.secondary',
                                            '& p': { m: 0, mb: 0.5 },
                                            '& strong': { fontWeight: 700, color: 'text.primary' },
                                            '& ul, & ol': { pl: 2, mb: 0.5, mt: 0.5 },
                                            '& li': { mb: 0.25 },
                                            '& code': { fontFamily: 'monospace', bgcolor: 'action.hover', px: 0.5, borderRadius: '3px', fontSize: '0.8rem' },
                                        }}>
                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{note.content}</ReactMarkdown>
                                        </Box>
                                    </Box>
                                    <Divider />
                                    <Box sx={{ px: 2, py: 0.75 }}>
                                        <Typography variant="caption" color="text.disabled">
                                            {new Date(note.updated_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}
                                        </Typography>
                                    </Box>
                                </Paper>
                            )}
                        </Grid>
                    ))}
                </Grid>
            )}
        </Box>
    );
};

// --- Details Modal Component ---
const LectureDetailsModal = ({ open, onClose, lecture, details, detailLecture, loading, onGenerateQuestions, generating }) => {
    const [tabValue, setTabValue] = useState(0);

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="md"
            fullWidth
            PaperProps={{ sx: { borderRadius: 3, height: '85vh', display: 'flex', flexDirection: 'column' } }}
        >
            <DialogTitle sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', pb: 0, flexShrink: 0 }}>
                <Box>
                    <Typography variant="h6" fontWeight={700}>{lecture?.title}</Typography>
                    <Typography variant="caption" color="text.secondary">Generated Content & Analysis</Typography>
                </Box>
                <IconButton onClick={onClose}><CloseIcon /></IconButton>
            </DialogTitle>

            <Box sx={{ borderBottom: 1, borderColor: 'divider', px: 3, flexShrink: 0 }}>
                <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)} variant="scrollable" scrollButtons="auto" sx={{ '& .MuiTab-root': { fontWeight: 700 } }}>
                    <Tab icon={<QuizIcon fontSize="small" />} iconPosition="start" label="Questions" />
                    <Tab icon={<ArticleIcon fontSize="small" />} iconPosition="start" label="Summary" />
                    <Tab icon={<NotesIcon fontSize="small" />} iconPosition="start" label="Study Aids" />
                    <Tab icon={<StickyNote2Icon fontSize="small" />} iconPosition="start" label="Class Notes" sx={{ color: '#f59e0b', '&.Mui-selected': { color: '#f59e0b' } }} />
                </Tabs>
            </Box>

            <DialogContent sx={{ p: 0, flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                {loading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
                        <CircularProgress />
                    </Box>
                ) : (
                    <Box sx={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                        {/* Questions Tab */}
                        {tabValue === 0 && (
                            <Box sx={{ p: 3, height: '100%', overflowY: 'auto' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <QuizIcon color="primary" />
                                        <Typography variant="subtitle1" fontWeight={700}>Generated Questions</Typography>
                                        {details?.questions?.length > 0 && (
                                            <Chip label={`${details.questions.length} questions`} size="small" color="primary" variant="outlined" />
                                        )}
                                    </Box>
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        startIcon={generating ? <CircularProgress size={14} /> : <AutoAwesomeIcon />}
                                        onClick={() => onGenerateQuestions(lecture.id)}
                                        disabled={generating}
                                    >
                                        {generating ? 'Generating...' : 'Regenerate'}
                                    </Button>
                                </Box>

                                <Stack spacing={2}>
                                    {details?.questions?.map((q, idx) => (
                                        <Paper key={q.id} variant="outlined" sx={{ p: 2.5, borderRadius: 2 }}>
                                            {/* Question text with markdown */}
                                            <Box sx={{
                                                fontWeight: 600, mb: 1.5,
                                                '& p': { m: 0, fontWeight: 600, fontSize: '0.95rem', lineHeight: 1.6 },
                                                '& strong': { fontWeight: 800 },
                                                '& code': { fontFamily: 'monospace', bgcolor: 'action.hover', px: 0.5, borderRadius: '3px', fontSize: '0.88rem' },
                                            }}>
                                                <Typography component="span" variant="body1" fontWeight={700} sx={{ mr: 0.5 }}>{idx + 1}.</Typography>
                                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.question_text || ''}</ReactMarkdown>
                                            </Box>
                                            {/* Options */}
                                            <Stack spacing={0.5} sx={{ mb: 1.5 }}>
                                                {['A', 'B', 'C', 'D'].map(opt => q[`option_${opt.toLowerCase()}`] && (
                                                    <Box key={opt} sx={{
                                                        px: 1.5, py: 0.75, borderRadius: 1,
                                                        bgcolor: q.correct_option === opt ? 'rgba(16, 185, 129, 0.1)' : 'action.hover',
                                                        border: '1px solid',
                                                        borderColor: q.correct_option === opt ? 'success.main' : 'transparent',
                                                        display: 'flex', alignItems: 'flex-start', gap: 1,
                                                    }}>
                                                        {q.correct_option === opt && <CheckCircleIcon sx={{ fontSize: 16, color: 'success.main', mt: 0.3, flexShrink: 0 }} />}
                                                        <Box sx={{ '& p': { m: 0, fontSize: '0.875rem' }, '& strong': { fontWeight: 700 } }}>
                                                            <Typography component="span" variant="body2" fontWeight={700}>{opt}. </Typography>
                                                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{q[`option_${opt.toLowerCase()}`] || ''}</ReactMarkdown>
                                                        </Box>
                                                    </Box>
                                                ))}
                                            </Stack>
                                            {/* Explanation */}
                                            {q.explanation && (
                                                <Box sx={{
                                                    mt: 1, p: 1.5, borderRadius: 1.5,
                                                    bgcolor: 'rgba(245,158,11,0.06)',
                                                    border: '1px solid rgba(245,158,11,0.2)',
                                                    '& p': { m: 0, fontSize: '0.8rem', color: 'text.secondary' },
                                                    '& strong': { fontWeight: 700, color: '#f59e0b' },
                                                }}>
                                                    <Typography variant="caption" fontWeight={800} color="#f59e0b" display="block" mb={0.5}>💡 Explanation</Typography>
                                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.explanation}</ReactMarkdown>
                                                </Box>
                                            )}
                                        </Paper>
                                    ))}
                                    {!details?.questions?.length && (
                                        <Alert
                                            severity="info"
                                            variant="outlined"
                                            action={
                                                <Button color="primary" size="small" onClick={() => onGenerateQuestions(lecture.id)} disabled={generating}>
                                                    Generate Now
                                                </Button>
                                            }
                                        >
                                            No questions generated yet. Click to generate AI-powered MCQs from this lecture.
                                        </Alert>
                                    )}
                                </Stack>
                            </Box>
                        )}

                        {/* Summary Tab */}
                        {tabValue === 1 && (
                            <Box sx={{ p: 3, height: '100%', overflowY: 'auto' }}>
                                {details?.summary ? (
                                    <Typography variant="body1" sx={{ lineHeight: 1.8, color: 'text.secondary' }}>
                                        {details.summary}
                                    </Typography>
                                ) : (
                                    <Alert severity="info" variant="outlined">
                                        No summary available for this lecture yet.
                                    </Alert>
                                )}
                            </Box>
                        )}

                        {/* Study Aids Tab */}
                        {tabValue === 2 && lecture && (
                            <StudyAidsPanel
                                lecture={detailLecture || details}
                                lectureId={lecture.id}
                            />
                        )}

                        {/* Class Notes Tab */}
                        {tabValue === 3 && lecture && (
                            <ClassNotesPanel lectureId={lecture.id} />
                        )}
                    </Box>
                )}
            </DialogContent>

            <DialogActions sx={{ p: 2, borderTop: 1, borderColor: 'divider', flexShrink: 0 }}>
                <Button onClick={onClose} color="inherit">Close</Button>
                <Button
                    variant="contained"
                    startIcon={<QuizIcon />}
                    onClick={() => window.location.href = `/quiz-mode?noteId=${lecture?.id}&n=10`}
                    disabled={!details?.questions?.length}
                >
                    Start Quiz
                </Button>
            </DialogActions>
        </Dialog>
    );
};

// --- Lecture Card Component ---
const LectureCard = ({ lecture, onDelete, onViewDetails, onSelect, isSelected }) => {
    const fileType = getFileType(lecture.file);
    const fileTypeIcon = {
        pdf: <PdfIcon sx={{ color: '#ef4444', fontSize: 32 }} />,
        image: <ImageIcon sx={{ color: 'primary.main', fontSize: 32 }} />,
        video: <VideoIcon sx={{ color: 'secondary.main', fontSize: 32 }} />,
        audio: <AudioIcon sx={{ color: 'warning.main', fontSize: 32 }} />,
    }[fileType] || <DescriptionIcon sx={{ fontSize: 32 }} />;

    return (
        <Paper
            onClick={() => onSelect && onSelect(lecture)}
            elevation={0}
            sx={{
                p: 2.5,
                borderRadius: '16px',
                border: '2px solid',
                borderColor: isSelected ? 'primary.main' : 'divider',
                bgcolor: isSelected ? (theme) => theme.palette.mode === 'dark' ? 'rgba(37,99,235,0.08)' : 'rgba(37,99,235,0.04)' : 'background.paper',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                '&:hover': {
                    borderColor: 'primary.main',
                    transform: 'translateY(-2px)',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                }
            }}
        >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {/* File Type Icon */}
                <Box sx={{
                    width: 52, height: 52, borderRadius: '12px', flexShrink: 0,
                    background: isSelected
                        ? 'linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(37,99,235,0.25) 100%)'
                        : 'linear-gradient(135deg, rgba(37,99,235,0.05) 0%, rgba(37,99,235,0.15) 100%)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                    {fileTypeIcon}
                </Box>

                {/* Content */}
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 0.25, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {lecture.title}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'text.secondary' }}>
                            <CalendarIcon sx={{ fontSize: 13 }} />
                            <Typography variant="caption">
                                {new Date(lecture.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                            </Typography>
                        </Box>
                        <Chip
                            label="Processed"
                            size="small"
                            icon={<CheckCircleIcon sx={{ fontSize: '14px !important' }} />}
                            sx={{
                                height: 24, fontSize: '0.75rem', px: 0.5,
                                bgcolor: 'rgba(16, 185, 129, 0.15)',
                                color: '#059669', fontWeight: 800,
                                borderRadius: '6px',
                                border: '1px solid rgba(16, 185, 129, 0.2)',
                                '& .MuiChip-icon': { color: '#059669' }
                            }}
                        />
                        {/* Study aids badge */}
                        {(lecture.study_notes || lecture.key_points?.length > 0) && (
                            <Chip
                                label="Notes Ready"
                                size="small"
                                icon={<NotesIcon sx={{ fontSize: '14px !important' }} />}
                                sx={{
                                    height: 24, fontSize: '0.75rem', px: 0.5,
                                    bgcolor: 'rgba(3,140,127,0.12)',
                                    color: '#038C7F', fontWeight: 800,
                                    borderRadius: '6px',
                                    border: '1px solid rgba(3,140,127,0.2)',
                                    '& .MuiChip-icon': { color: '#038C7F' }
                                }}
                            />
                        )}
                    </Box>
                </Box>

                {/* Actions */}
                <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', flexShrink: 0 }} onClick={e => e.stopPropagation()}>
                    <Tooltip title="View details & questions">
                        <IconButton
                            size="small"
                            onClick={() => onViewDetails(lecture)}
                            sx={{ color: 'primary.main', '&:hover': { bgcolor: 'rgba(37,99,235,0.1)' } }}
                        >
                            <AutoAwesomeIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete lecture">
                        <IconButton
                            size="small"
                            onClick={() => onDelete(lecture)}
                            sx={{ color: 'text.disabled', '&:hover': { color: 'error.main', bgcolor: 'rgba(239,68,68,0.1)' } }}
                        >
                            <DeleteIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>
        </Paper>
    );
};

// --- Upload Panel ---
const UploadPanel = ({ onUploadSuccess }) => {
    const [uploadTab, setUploadTab] = useState('upload');
    const [title, setTitle] = useState('');
    const [content, setContent] = useState('');
    const [pdfFile, setPdfFile] = useState(null);
    const [uploading, setUploading] = useState(false);
    const [showSuccess, setShowSuccess] = useState(false);
    const [error, setError] = useState('');

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles?.length > 0) setPdfFile(acceptedFiles[0]);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'application/pdf': ['.pdf'] },
        multiple: false
    });

    const handleUpload = async () => {
        if (!title.trim()) return;
        setUploading(true);
        setError('');

        try {
            if (uploadTab === 'text') {
                await API.post('upload-note/', { title, content });
            } else {
                const formData = new FormData();
                formData.append('title', title);
                formData.append('file', pdfFile);
                await API.post('upload-pdf/', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
            }

            setTitle('');
            setContent('');
            setPdfFile(null);
            setShowSuccess(true);
            setTimeout(() => setShowSuccess(false), 4000);
            onUploadSuccess();
        } catch (err) {
            setError('Upload failed. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    const isDisabled = uploading || !title.trim() ||
        (uploadTab === 'text' && !content.trim()) ||
        (uploadTab === 'upload' && !pdfFile);

    return (
        <Paper elevation={0} sx={{ p: 3, borderRadius: '20px', border: '1px solid', borderColor: 'divider', position: 'sticky', top: 24 }}>
            <Typography variant="h6" fontWeight={800} gutterBottom>Add Material</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Upload a PDF or paste text content
            </Typography>

            {/* Tab Toggle */}
            <Box sx={{ p: 0.5, bgcolor: 'action.hover', borderRadius: '12px', mb: 3, display: 'flex', gap: 0.5 }}>
                {['upload', 'text'].map(tab => (
                    <Button
                        key={tab}
                        fullWidth
                        size="small"
                        variant={uploadTab === tab ? 'contained' : 'text'}
                        onClick={() => setUploadTab(tab)}
                        sx={{ 
                            borderRadius: '10px', 
                            boxShadow: uploadTab === tab ? '0 4px 12px rgba(0,0,0,0.1)' : 0, 
                            fontWeight: 700, 
                            py: 1,
                            color: uploadTab === tab ? 'white' : 'text.secondary'
                        }}
                    >
                        {tab === 'upload' ? 'Upload PDF' : 'Paste Text'}
                    </Button>
                ))}
            </Box>

            <Stack spacing={2.5}>
                <TextField
                    label="Lecture Title"
                    fullWidth
                    size="small"
                    value={title}
                    onChange={e => setTitle(e.target.value)}
                    sx={{ '& .MuiOutlinedInput-root': { borderRadius: '10px' } }}
                />

                {uploadTab === 'text' ? (
                    <TextField
                        label="Content"
                        multiline
                        rows={7}
                        fullWidth
                        size="small"
                        value={content}
                        onChange={e => setContent(e.target.value)}
                        sx={{ '& .MuiOutlinedInput-root': { borderRadius: '10px' } }}
                    />
                ) : (
                    <Box
                        {...getRootProps()}
                        sx={{
                            height: 160,
                            border: '2px dashed',
                            borderColor: isDragActive ? 'primary.main' : pdfFile ? 'success.main' : 'divider',
                            borderRadius: '12px',
                            display: 'flex', flexDirection: 'column',
                            alignItems: 'center', justifyContent: 'center',
                            cursor: 'pointer',
                            bgcolor: isDragActive ? 'rgba(37,99,235,0.05)' : pdfFile ? 'rgba(16,185,129,0.05)' : 'background.default',
                            transition: 'all 0.2s',
                            '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(37,99,235,0.05)' }
                        }}
                    >
                        <input {...getInputProps()} />
                        {pdfFile ? (
                            <>
                                <CheckCircleIcon sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
                                <Typography variant="subtitle2" fontWeight={700} color="success.main" align="center" sx={{ px: 2 }}>
                                    {pdfFile.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                    {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                                </Typography>
                            </>
                        ) : (
                            <>
                                <CloudUploadIcon sx={{ fontSize: 40, color: 'text.disabled', mb: 1 }} />
                                <Typography variant="subtitle2" fontWeight={600} color="text.secondary">
                                    {isDragActive ? 'Drop here!' : 'Click or drag PDF'}
                                </Typography>
                                <Typography variant="caption" color="text.disabled">Max 25MB</Typography>
                            </>
                        )}
                    </Box>
                )}

                <Button
                    fullWidth
                    variant="contained"
                    size="large"
                    onClick={handleUpload}
                    disabled={isDisabled}
                    sx={{ py: 1.5, borderRadius: '10px', fontWeight: 700 }}
                >
                    {uploading ? <CircularProgress size={22} color="inherit" /> : '✨ Generate Content'}
                </Button>

                <Fade in={showSuccess}>
                    <Alert severity="success" sx={{ borderRadius: '10px' }}>
                        Uploaded successfully! Processing your material...
                    </Alert>
                </Fade>
                {error && (
                    <Alert severity="error" sx={{ borderRadius: '10px' }} onClose={() => setError('')}>
                        {error}
                    </Alert>
                )}
            </Stack>
        </Paper>
    );
};


// --- Main Page ---
export default function Lectures() {
    const { user } = useAuth();
    const navigate = useNavigate();

    const [lectures, setLectures] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeTab, setActiveTab] = useState(0); // 0=Library, 1=Preview
    const [selectedLecture, setSelectedLecture] = useState(null);

    // Details Modal State
    const [detailsOpen, setDetailsOpen] = useState(false);
    const [detailsLecture, setDetailsLecture] = useState(null);
    const [detailsData, setDetailsData] = useState(null);
    const [detailLecture, setDetailLecture] = useState(null); // full lecture obj with notes/formulas/key_points
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [generating, setGenerating] = useState(false);

    // Delete Dialog State
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
            console.error('Failed to fetch lectures:', err);
        } finally {
            setLoading(false);
        }
    };

    const filteredLectures = lectures.filter(l =>
        l.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleSelectLecture = (lecture) => {
        setSelectedLecture(lecture);
        setActiveTab(1); // Switch to Preview tab
    };

    const handleViewDetails = async (lecture) => {
        setDetailsLecture(lecture);
        setDetailsOpen(true);
        setLoadingDetails(true);
        setDetailLecture(null);
        try {
            const res = await API.get(`lectures/${lecture.id}/`);
            setDetailsData(res.data);
            // The detail response includes study_notes, formulas, key_points
            setDetailLecture(res.data);
        } catch (err) {
            console.error(err);
        } finally {
            setLoadingDetails(false);
        }
    };

    const handleGenerateQuestions = async (noteId) => {
        setGenerating(true);
        try {
            await API.post('generate-mcqs/', { note_id: noteId, count: 10 });
            const res = await API.get(`lectures/${noteId}/`);
            setDetailsData(res.data);
        } catch (err) {
            console.error('Failed to generate questions:', err);
        } finally {
            setGenerating(false);
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
            setLectures(prev => prev.filter(l => l.id !== lectureToDelete.id));
            if (selectedLecture?.id === lectureToDelete.id) {
                setSelectedLecture(null);
                setActiveTab(0);
            }
            setDeleteDialogOpen(false);
            setLectureToDelete(null);
        } catch (err) {
            alert('Failed to delete lecture');
        }
    };

    return (
        <Box>
            {/* Header */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" fontWeight={900} gutterBottom>Lecture Library</Typography>
                <Typography color="text.secondary">Upload and manage your study materials</Typography>
            </Box>

            <Grid container spacing={3} sx={{ alignItems: 'flex-start' }}>
                {/* Left: Upload Panel */}
                <Grid item xs={12} lg={3} md={4}>
                    <UploadPanel onUploadSuccess={fetchLectures} />
                </Grid>

                {/* Right: Tabbed Panel */}
                <Grid item xs={12} lg={9} md={8}>
                    <Paper elevation={0} sx={{ borderRadius: '20px', border: '1px solid', borderColor: 'divider', overflow: 'hidden', minHeight: 600 }}>
                        {/* Tab Header */}
                        <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', px: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                            <Tabs
                                value={activeTab}
                                onChange={(_, v) => setActiveTab(v)}
                                sx={{ '& .MuiTab-root': { fontWeight: 700, minHeight: 56 } }}
                            >
                                <Tab
                                    icon={<LibraryIcon fontSize="small" />}
                                    iconPosition="start"
                                    label={`Library (${lectures.length})`}
                                />
                                <Tab
                                    icon={<PreviewIcon fontSize="small" />}
                                    iconPosition="start"
                                    label={selectedLecture ? `Preview: ${selectedLecture.title.slice(0, 20)}${selectedLecture.title.length > 20 ? '…' : ''}` : 'Preview'}
                                    sx={{ color: selectedLecture ? 'primary.main' : 'text.secondary' }}
                                />
                            </Tabs>

                            {/* Search - shown on library tab */}
                            {activeTab === 0 && (
                                <TextField
                                    size="small"
                                    placeholder="Search lectures..."
                                    value={searchQuery}
                                    onChange={e => setSearchQuery(e.target.value)}
                                    InputProps={{
                                        startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" sx={{ color: 'text.disabled' }} /></InputAdornment>,
                                        sx: { borderRadius: '10px', fontSize: '0.875rem' }
                                    }}
                                    sx={{ width: 260, my: 1 }}
                                />
                            )}
                        </Box>

                        {/* Tab 0: Library */}
                        {activeTab === 0 && (
                            <Box sx={{ p: 3 }}>
                                {loading ? (
                                    <Box sx={{ display: 'flex', justifyContent: 'center', py: 10 }}>
                                        <CircularProgress />
                                    </Box>
                                ) : filteredLectures.length === 0 ? (
                                    <Box sx={{ textAlign: 'center', py: 10 }}>
                                        <LibraryIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.3, mb: 2 }} />
                                        <Typography variant="h6" color="text.secondary">
                                            {searchQuery ? 'No lectures match your search' : 'No lectures yet'}
                                        </Typography>
                                        <Typography variant="body2" color="text.disabled">
                                            {searchQuery ? 'Try a different search term' : 'Upload your first lecture using the panel on the left'}
                                        </Typography>
                                    </Box>
                                ) : (
                                    <Stack spacing={2}>
                                        {filteredLectures.map(lecture => (
                                            <LectureCard
                                                key={lecture.id}
                                                lecture={lecture}
                                                isSelected={selectedLecture?.id === lecture.id}
                                                onSelect={handleSelectLecture}
                                                onDelete={handleDeleteClick}
                                                onViewDetails={handleViewDetails}
                                            />
                                        ))}
                                    </Stack>
                                )}
                            </Box>
                        )}

                        {/* Tab 1: Preview */}
                        {activeTab === 1 && (
                            <Box sx={{ height: 'calc(100vh - 260px)', minHeight: 500, display: 'flex', flexDirection: 'column' }}>
                                <FileViewer lecture={selectedLecture} />
                            </Box>
                        )}
                    </Paper>
                </Grid>
            </Grid>

            {/* Details Modal */}
            <LectureDetailsModal
                open={detailsOpen}
                onClose={() => setDetailsOpen(false)}
                lecture={detailsLecture}
                details={detailsData}
                detailLecture={detailLecture}
                loading={loadingDetails}
                onGenerateQuestions={handleGenerateQuestions}
                generating={generating}
            />

            {/* Delete Confirmation Dialog */}
            <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)} maxWidth="xs" fullWidth PaperProps={{ sx: { borderRadius: 3 } }}>
                <DialogTitle>Delete Lecture?</DialogTitle>
                <DialogContent>
                    <Typography>
                        Are you sure you want to delete <strong>"{lectureToDelete?.title}"</strong>? 
                        This will also delete all associated questions and data. This action cannot be undone.
                    </Typography>
                </DialogContent>
                <DialogActions sx={{ p: 2, gap: 1 }}>
                    <Button onClick={() => setDeleteDialogOpen(false)} variant="outlined" fullWidth>Cancel</Button>
                    <Button onClick={confirmDelete} variant="contained" color="error" fullWidth>Delete</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
}
