// src/pages/Lectures.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
    Box, Typography, IconButton, 
    Button, Chip, CircularProgress, Alert, 
    Dialog, DialogActions, DialogContent, DialogTitle,
    Grid, Paper, TextField, Stack, Tab, Tabs, InputAdornment,
    Divider, Tooltip, Fade, Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import {
  Add as AddIcon,
  Article as ArticleIcon,
  AutoAwesome as AutoAwesomeIcon,
  CalendarToday as CalendarIcon,
  CheckCircle as CheckCircleIcon,
  CloudUpload as CloudUploadIcon,
  Close as CloseIcon,
  Delete as DeleteIcon,
  Description as DescriptionIcon,
  Download as DownloadIcon,
  ExpandMore as ExpandMoreIcon,
  Image as ImageIcon,
  LibraryBooks as LibraryIcon,
  NavigateBefore as PrevIcon,
  NavigateNext as NextIcon,
  Notes as NotesIcon,
  OpenInNew as OpenInNewIcon,
  PictureAsPdf as PdfIcon,
  Preview as PreviewIcon,
  Quiz as QuizIcon,
  Search as SearchIcon,
  VideoFile as VideoIcon,
  AudioFile as AudioIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import NotesSidebar from '../components/NotesSidebar';
import { captureSelectedText } from '../components/PDFTextSelector';
import { subjectToColor } from '../utils/subjectColors';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { useDropzone } from 'react-dropzone';
import { useAuth } from '../context/AuthContext';
import API from '../api/api';
import { useNavigate } from 'react-router-dom';

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

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

const FileViewer = ({ lecture }) => {
    const [numPages, setNumPages] = useState(0);
    const [pageNumber, setPageNumber] = useState(1);
    const [scale, setScale] = useState(1.0);
    const [selectedText, setSelectedText] = useState('');
    const viewerRef = useRef(null);

    const fileType = getFileType(lecture?.file);
    const fileUrl = getFileUrl(lecture?.file);

    useEffect(() => {
        setNumPages(0);
        setPageNumber(1);
        setScale(1.0);
        setSelectedText('');
    }, [lecture?.id, lecture?.file]);

    const handleMouseUp = () => {
        const text = captureSelectedText();
        setSelectedText(text);
        if (text) {
            window.lastSelectedPdfText = text;
        }
    };

    const dispatchSelection = () => {
        const text = selectedText || captureSelectedText() || window.lastSelectedPdfText || '';
        if (!text.trim()) return;
        window.lastSelectedPdfText = text;
        window.dispatchEvent(new CustomEvent('lf:pdf-selection', {
            detail: { text, lectureId: lecture?.id, pageNumber },
        }));
        setSelectedText('');
        const selection = window.getSelection?.();
        if (selection && selection.removeAllRanges) selection.removeAllRanges();
    };

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

    const showPdfSelectionButton = fileType === 'pdf' && Boolean(selectedText);

    return (
        <Box ref={viewerRef} sx={{ display: 'flex', flexDirection: 'column', height: '100%', position: 'relative' }} onMouseUp={handleMouseUp}>
            <Box sx={{
                px: 2.5, py: 1.5,
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                borderBottom: '1px solid', borderColor: 'divider',
                bgcolor: 'background.paper',
                flexShrink: 0,
                gap: 2,
                position: 'sticky',
                top: 0,
                zIndex: 1,
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.25, minWidth: 0 }}>
                    {fileType === 'pdf' && <PdfIcon sx={{ color: '#ef4444' }} />}
                    {fileType === 'image' && <ImageIcon sx={{ color: 'primary.main' }} />}
                    {fileType === 'video' && <VideoIcon sx={{ color: 'secondary.main' }} />}
                    {fileType === 'audio' && <AudioIcon sx={{ color: 'warning.main' }} />}
                    <Box sx={{ minWidth: 0 }}>
                        <Typography variant="subtitle1" fontWeight={800} noWrap>
                            {lecture.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            {fileType === 'pdf' ? `Page ${pageNumber}${numPages ? ` / ${numPages}` : ''}` : 'Preview'}
                        </Typography>
                    </Box>
                </Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                    {fileType === 'pdf' && (
                        <>
                            <Tooltip title="Previous page"><span><IconButton size="small" disabled={pageNumber <= 1} onClick={() => setPageNumber((p) => Math.max(1, p - 1))}><PrevIcon fontSize="small" /></IconButton></span></Tooltip>
                            <Tooltip title="Next page"><span><IconButton size="small" disabled={numPages > 0 && pageNumber >= numPages} onClick={() => setPageNumber((p) => p + 1)}><NextIcon fontSize="small" /></IconButton></span></Tooltip>
                            <Divider orientation="vertical" flexItem />
                            <Tooltip title="Zoom out"><IconButton size="small" onClick={() => setScale((s) => Math.max(0.4, +(s - 0.1).toFixed(1)))}><DownloadIcon sx={{ transform: 'rotate(180deg)' }} fontSize="small" /></IconButton></Tooltip>
                            <Typography variant="caption" fontWeight={700} sx={{ minWidth: 48, textAlign: 'center' }}>{Math.round(scale * 100)}%</Typography>
                            <Tooltip title="Zoom in"><IconButton size="small" onClick={() => setScale((s) => +(s + 0.1).toFixed(1))}><DownloadIcon fontSize="small" /></IconButton></Tooltip>
                        </>
                    )}
                    <Tooltip title="Open in new tab"><IconButton size="small" component="a" href={fileUrl} target="_blank" rel="noreferrer"><OpenInNewIcon fontSize="small" /></IconButton></Tooltip>
                </Box>
            </Box>

            <Box sx={{ flex: 1, overflowY: 'auto', p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', bgcolor: fileType === 'pdf' ? '#1e1e2e' : 'background.default' }}>
                {fileType === 'pdf' ? (
                    <Document
                        file={fileUrl}
                        loading={<Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}><CircularProgress sx={{ color: 'white' }} /></Box>}
                        onLoadSuccess={({ numPages: loadedPages }) => setNumPages(loadedPages)}
                        error={<Alert severity="error">Failed to load PDF</Alert>}
                    >
                        <Page
                            pageNumber={Math.min(pageNumber, numPages || pageNumber)}
                            scale={scale}
                            renderTextLayer
                            renderAnnotationLayer={false}
                            width={undefined}
                        />
                    </Document>
                ) : fileType === 'image' ? (
                    <img src={fileUrl} alt={lecture.title} style={{ maxWidth: '100%', borderRadius: 12, objectFit: 'contain' }} />
                ) : fileType === 'video' ? (
                    <video controls style={{ width: '100%', maxWidth: 960, borderRadius: 12 }}><source src={fileUrl} /></video>
                ) : fileType === 'audio' ? (
                    <audio controls style={{ width: '100%' }}><source src={fileUrl} /></audio>
                ) : (
                    <Alert severity="info">Preview not available</Alert>
                )}
            </Box>

            {showPdfSelectionButton && (
                <Box sx={{ position: 'absolute', right: 16, bottom: 16, zIndex: 5 }}>
                    <Button variant="contained" color="secondary" onClick={dispatchSelection} sx={{ boxShadow: '0 12px 28px rgba(124,58,237,0.28)' }}>
                        Add selection to notes
                    </Button>
                </Box>
            )}
        </Box>
    );
};

const StudyAidsPanel = ({ lecture }) => {
    const [tabValue, setTabValue] = useState(0);
    const formulas = lecture?.formulas || [];
    const keyPoints = lecture?.key_points || [];

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Box sx={{ px: 3, py: 2, borderBottom: '1px solid', borderColor: 'divider', flexShrink: 0 }}>
                <Typography variant="subtitle1" fontWeight={800}>AI Study Aids</Typography>
                <Typography variant="caption" color="text.secondary">
                    Study notes, formulas, and key points for this lecture
                </Typography>
            </Box>

            <Box sx={{ px: 3, borderBottom: '1px solid', borderColor: 'divider', flexShrink: 0 }}>
                <Tabs value={tabValue} onChange={(_, value) => setTabValue(value)} variant="scrollable" scrollButtons="auto">
                    <Tab label="Notes" />
                    <Tab label={`Formulas (${formulas.length})`} />
                    <Tab label={`Key Points (${keyPoints.length})`} />
                </Tabs>
            </Box>

            <Box sx={{ p: 3, flex: 1, overflowY: 'auto' }}>
                {tabValue === 0 && (
                    lecture?.study_notes ? (
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{lecture.study_notes}</ReactMarkdown>
                    ) : (
                        <Alert severity="info" variant="outlined">No study notes generated yet.</Alert>
                    )
                )}

                {tabValue === 1 && (
                    <Stack spacing={1.5}>
                        {formulas.length > 0 ? formulas.map((formula, index) => (
                            <Paper key={index} elevation={0} sx={{ p: 2, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                                <Typography variant="subtitle2" fontWeight={800} gutterBottom>
                                    {formula.name || `Formula ${index + 1}`}
                                </Typography>
                                <Typography variant="body2" sx={{ fontFamily: 'monospace', mb: formula.description ? 1 : 0 }}>
                                    {formula.formula || '—'}
                                </Typography>
                                {formula.description && <Typography variant="body2" color="text.secondary">{formula.description}</Typography>}
                            </Paper>
                        )) : <Alert severity="info" variant="outlined">No formulas extracted yet.</Alert>}
                    </Stack>
                )}

                {tabValue === 2 && (
                    <Stack spacing={1.25}>
                        {keyPoints.length > 0 ? keyPoints.map((point, index) => (
                            <Paper key={index} elevation={0} sx={{ p: 2, borderRadius: 2, border: '1px solid', borderColor: 'divider' }}>
                                <Typography variant="body2">{point}</Typography>
                            </Paper>
                        )) : <Alert severity="info" variant="outlined">No key points extracted yet.</Alert>}
                    </Stack>
                )}
            </Box>
        </Box>
    );
};

// NotesSidebar moved to frontend/src/components/NotesSidebar.js

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
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1.5 }}>
                                                <Box sx={{
                                                    fontWeight: 600,
                                                    '& p': { m: 0, fontWeight: 600, fontSize: '0.95rem', lineHeight: 1.6 },
                                                    '& strong': { fontWeight: 800 },
                                                    '& code': { fontFamily: 'monospace', bgcolor: 'action.hover', px: 0.5, borderRadius: '3px', fontSize: '0.88rem' },
                                                }}>
                                                    <Typography component="span" variant="body1" fontWeight={700} sx={{ mr: 0.5 }}>{idx + 1}.</Typography>
                                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{q.question_text || ''}</ReactMarkdown>
                                                </Box>
                                                <Box sx={{ display: 'flex', gap: 0.5, flexShrink: 0, ml: 2 }}>
                                                    {q.question_type && (
                                                        <Chip 
                                                            label={q.question_type.replace('_', ' ').toUpperCase()} 
                                                            size="small" 
                                                            sx={{ height: 20, fontSize: '0.65rem', fontWeight: 800, bgcolor: 'rgba(38,70,83,0.1)', color: '#264653' }} 
                                                        />
                                                    )}
                                                    {q.blooms_level && (
                                                        <Chip 
                                                            label={q.blooms_level.toUpperCase()} 
                                                            size="small" 
                                                            sx={{ height: 20, fontSize: '0.65rem', fontWeight: 800, bgcolor: 'rgba(231,111,81,0.1)', color: '#e76f51' }} 
                                                        />
                                                    )}
                                                    {q.is_high_yield && (
                                                        <Chip 
                                                            label="HIGH YIELD" 
                                                            size="small" 
                                                            sx={{ height: 20, fontSize: '0.65rem', fontWeight: 800, bgcolor: 'rgba(239,68,68,0.1)', color: '#ef4444' }} 
                                                        />
                                                    )}
                                                </Box>
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
    const subjectColor = subjectToColor(lecture.subject || 'General');
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
                borderColor: isSelected ? subjectColor : 'divider',
                bgcolor: isSelected ? (theme) => theme.palette.mode === 'dark' ? 'rgba(37,99,235,0.08)' : 'rgba(37,99,235,0.04)' : 'background.paper',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                '&:hover': {
                    borderColor: subjectColor,
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
                        ? `linear-gradient(135deg, ${subjectColor}22 0%, ${subjectColor}38 100%)`
                        : `linear-gradient(135deg, ${subjectColor}12 0%, ${subjectColor}26 100%)`,
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
                        <Chip
                            label={lecture.subject || 'General'}
                            size="small"
                            sx={{
                                height: 24,
                                fontSize: '0.72rem',
                                fontWeight: 800,
                                bgcolor: `${subjectColor}16`,
                                color: subjectColor,
                                borderRadius: '6px',
                            }}
                        />
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
    useAuth();
    useNavigate();

    const [lectures, setLectures] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedLecture, setSelectedLecture] = useState(null);
    const [uploadDialogOpen, setUploadDialogOpen] = useState(false);

    // Details Modal State
    const [detailsOpen, setDetailsOpen] = useState(false);
    const [detailsLecture, setDetailsLecture] = useState(null);
    const [detailsData, setDetailsData] = useState(null);
    const [detailLecture, setDetailLecture] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [generating, setGenerating] = useState(false);

    // Delete Dialog State
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [lectureToDelete, setLectureToDelete] = useState(null);

    const fetchLectures = useCallback(async () => {
        try {
            const response = await API.get('lectures/');
            setLectures(response.data);
            if (!selectedLecture && response.data.length > 0) {
                setSelectedLecture(response.data[0]);
            }
        } catch (err) {
            console.error('Failed to fetch lectures:', err);
        } finally {
            setLoading(false);
        }
    }, [selectedLecture]);

    useEffect(() => {
        fetchLectures();
    }, [fetchLectures]);

    const filteredLectures = lectures.filter(l =>
        l.title.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const groupedLectures = filteredLectures.reduce((acc, lecture) => {
        const subject = lecture.subject || 'General';
        if (!acc[subject]) acc[subject] = [];
        acc[subject].push(lecture);
        return acc;
    }, {});

    const handleSelectLecture = (lecture) => {
        setSelectedLecture(lecture);
    };

    const handleViewDetails = async (lecture) => {
        setDetailsLecture(lecture);
        setDetailsOpen(true);
        setLoadingDetails(true);
        setDetailLecture(null);
        try {
            const res = await API.get(`lectures/${lecture.id}/`);
            setDetailsData(res.data);
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
            }
            setDeleteDialogOpen(false);
            setLectureToDelete(null);
        } catch (err) {
            alert('Failed to delete lecture');
        }
    };

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 120px)', minHeight: 600 }}>
            {/* Header */}
            <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexShrink: 0 }}>
                <Box>
                    <Typography variant="h4" fontWeight={900} gutterBottom>Lecture Library</Typography>
                    <Typography color="text.secondary">Organized by subject for seamless studying.</Typography>
                </Box>
                <Button variant="contained" size="large" startIcon={<AddIcon />} onClick={() => setUploadDialogOpen(true)} sx={{ borderRadius: 2, fontWeight: 700, px: 4, py: 1.5, boxShadow: '0 4px 14px rgba(37,99,235,0.30)' }}>
                    Add Material
                </Button>
            </Box>

            <Grid container spacing={3} sx={{ flex: 1, minHeight: 0 }}>
                {/* Left: Library Accordion */}
                <Grid item xs={12} lg={3} md={4} sx={{ height: '100%' }}>
                    <Paper elevation={0} sx={{ borderRadius: '20px', border: '1px solid', borderColor: 'divider', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                        <Box sx={{ p: 2.5, borderBottom: '1px solid', borderColor: 'divider', bgcolor: 'background.paper' }}>
                            <TextField
                                fullWidth
                                size="small"
                                placeholder="Search lectures..."
                                value={searchQuery}
                                onChange={e => setSearchQuery(e.target.value)}
                                InputProps={{
                                    startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" sx={{ color: 'text.disabled' }} /></InputAdornment>,
                                    sx: { borderRadius: '12px', bgcolor: 'action.hover', '& fieldset': { border: 'none' } }
                                }}
                            />
                        </Box>
                        
                        <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
                            {loading ? (
                                <Box sx={{ display: 'flex', justifyContent: 'center', py: 5 }}><CircularProgress /></Box>
                            ) : Object.keys(groupedLectures).length === 0 ? (
                                <Box sx={{ textAlign: 'center', py: 8 }}>
                                    <LibraryIcon sx={{ fontSize: 60, color: 'text.disabled', opacity: 0.3, mb: 2 }} />
                                    <Typography variant="subtitle1" color="text.secondary" fontWeight={600}>No lectures found</Typography>
                                </Box>
                            ) : (
                                Object.entries(groupedLectures).map(([subject, subLectures], idx) => (
                                    <Accordion key={subject} defaultExpanded={idx === 0} elevation={0} sx={{ mb: 1, border: '1px solid', borderColor: 'divider', borderRadius: '12px !important', overflow: 'hidden', '&:before': { display: 'none' } }}>
                                        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ bgcolor: 'rgba(37,99,235,0.03)', px: 2, '& .MuiAccordionSummary-content': { my: 1.5 } }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                                <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: subjectToColor(subject) }} />
                                                <Typography variant="subtitle2" fontWeight={800}>{subject}</Typography>
                                                <Chip label={subLectures.length} size="small" sx={{ height: 20, fontSize: '0.7rem', fontWeight: 700 }} />
                                            </Box>
                                        </AccordionSummary>
                                        <AccordionDetails sx={{ p: 1.5, bgcolor: 'background.paper' }}>
                                            <Stack spacing={1}>
                                                {subLectures.map(lecture => (
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
                                        </AccordionDetails>
                                    </Accordion>
                                ))
                            )}
                        </Box>
                    </Paper>
                </Grid>

                {/* Right: Persistent File Viewer & Notes Sidebar */}
                <Grid item xs={12} lg={9} md={8} sx={{ height: '100%' }}>
                    {selectedLecture ? (
                        <Box sx={{ display: 'flex', height: '100%', gap: 3 }}>
                            {/* Viewer Section */}
                            <Paper elevation={0} sx={{ borderRadius: '20px', border: '1px solid', borderColor: 'divider', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 3 }}>
                                {/* Sticky Action Bar */}
                                <Box sx={{ 
                                    p: 2, px: 3, 
                                    borderBottom: '1px solid', borderColor: 'divider', 
                                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                    bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.02)' : 'rgba(0,0,0,0.01)',
                                }}>
                                    <Box sx={{ flex: 1, minWidth: 0, mr: 2 }}>
                                        <Typography variant="subtitle1" fontWeight={800} noWrap>{selectedLecture.title}</Typography>
                                        <Typography variant="caption" color="text.secondary">Reading mode • {selectedLecture.subject || 'General'}</Typography>
                                    </Box>
                                    <Button 
                                        variant="contained" 
                                        color="secondary"
                                        startIcon={<AutoAwesomeIcon />} 
                                        onClick={() => handleViewDetails(selectedLecture)}
                                        sx={{ borderRadius: '10px', fontWeight: 700, whiteSpace: 'nowrap' }}
                                    >
                                        Generate Study Aids
                                    </Button>
                                </Box>
                                {/* PDF Viewer Area */}
                                <Box sx={{ flex: 1, overflowY: 'auto', p: 0, display: 'flex', flexDirection: 'column' }}>
                                    <FileViewer lecture={selectedLecture} />
                                </Box>
                            </Paper>

                            {/* Notes Sidebar Section */}
                            <Paper elevation={0} sx={{ borderRadius: '20px', border: '1px solid', borderColor: 'divider', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1, minWidth: 280 }}>
                                <NotesSidebar lectureId={selectedLecture.id} />
                            </Paper>
                        </Box>
                    ) : (
                        <Paper elevation={0} sx={{ borderRadius: '20px', border: '1px solid', borderColor: 'divider', height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: 2 }}>
                                <PreviewIcon sx={{ fontSize: 80, color: 'text.disabled', opacity: 0.2 }} />
                                <Typography variant="h6" color="text.secondary" fontWeight={700}>No Lecture Selected</Typography>
                                <Typography variant="body2" color="text.disabled">Select a lecture from the library to view it here.</Typography>
                            </Box>
                        </Paper>
                    )}
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

            {/* Upload Modal */}
            <Dialog open={uploadDialogOpen} onClose={() => setUploadDialogOpen(false)} maxWidth="md" fullWidth PaperProps={{ sx: { borderRadius: '24px', p: 1 } }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2, px: 2, pt: 1 }}>
                    <Typography variant="h6" fontWeight={800}>Add New Material</Typography>
                    <IconButton onClick={() => setUploadDialogOpen(false)}><CloseIcon /></IconButton>
                </Box>
                <DialogContent sx={{ pt: 0 }}>
                    <UploadPanel onUploadSuccess={() => { fetchLectures(); setUploadDialogOpen(false); }} />
                </DialogContent>
            </Dialog>

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
