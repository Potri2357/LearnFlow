import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography, Paper, Button, IconButton, Chip, Stack, TextField, Tooltip, CircularProgress } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import { Delete as DeleteIcon, Edit as EditIcon, ContentPaste as ContentPasteIcon, Notes as NotesIcon } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import API from '../api/api';

const NOTE_COLORS = [
  { hex: '#f9edca', label: 'Sand' },
  { hex: '#c7ece9', label: 'Teal' },
  { hex: '#fde3c5', label: 'Clay' },
  { hex: '#fad5cc', label: 'Coral' },
  { hex: '#d6e3e7', label: 'Ink' },
];

const NOTE_TYPES = [
  { value: 'lecture', label: 'Lecture Note' },
  { value: 'hint', label: 'Hint' },
  { value: 'exam', label: 'Exam Note' },
  { value: 'formula', label: 'Formula' },
];

const NotesSidebar = ({ lectureId }) => {
  const [notes, setNotes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [examMode, setExamMode] = useState(false);

  // Form state
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [color, setColor] = useState('#f9edca');
  const [noteType, setNoteType] = useState('lecture');

  // Edit state
  const [editTitle, setEditTitle] = useState('');
  const [editContent, setEditContent] = useState('');
  const [editColor, setEditColor] = useState('#f9edca');
  const [editNoteType, setEditNoteType] = useState('lecture');

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

  // Listen for PDF selection events dispatched from FileViewer
  useEffect(() => {
    const handler = (e) => {
      const text = e?.detail?.text || window.lastSelectedPdfText;
      if (text) {
        setContent(prev => prev + (prev ? '\n\n' : '') + text);
        setCreating(true);
        // focus textarea is handled by browser; no ref here
      }
    };
    window.addEventListener('lf:pdf-selection', handler);
    return () => window.removeEventListener('lf:pdf-selection', handler);
  }, []);

  const handleCreate = async () => {
    if (!content.trim()) return;
    try {
      const res = await API.post('/sticky-notes/', {
        title: title.trim() || 'Untitled Note',
        content,
        color,
        note_type: noteType,
        lecture_note_id: lectureId,
      });
      setNotes(prev => [res.data, ...prev]);
      setTitle(''); setContent(''); setColor('#f9edca'); setNoteType('lecture');
      setCreating(false);
    } catch (e) { console.error(e); }
  };

  const handleStartEdit = (note) => {
    setEditingId(note.id);
    setEditTitle(note.title);
    setEditContent(note.content);
    setEditColor(note.color);
    setEditNoteType(note.note_type || 'lecture');
  };

  const handleSaveEdit = async (noteId) => {
    try {
      const res = await API.put(`/sticky-notes/${noteId}/`, {
        title: editTitle,
        content: editContent,
        color: editColor,
        note_type: editNoteType,
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

  const handleGrabText = (setContentFunc) => {
    const selected = window.getSelection().toString() || window.lastSelectedPdfText;
    if (selected) {
      setContentFunc(prev => prev + (prev ? '\n\n' : '') + selected);
    } else {
      alert('No text selected from PDF!');
    }
  };

  const ColorPicker = ({ value, onChange }) => (
    <Box display="flex" gap={0.5} flexWrap="wrap">
      {NOTE_COLORS.map(c => (
        <Tooltip key={c.hex} title={c.label}>
          <Box
            onClick={() => onChange(c.hex)}
            sx={{
              width: 20, height: 20, borderRadius: '50%',
              bgcolor: c.hex,
              border: value === c.hex ? '2px solid' : '2px solid transparent',
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

  const TypeSelector = ({ value, onChange }) => (
    <Box display="flex" gap={0.5} flexWrap="wrap">
      {NOTE_TYPES.map(t => (
        <Chip
          key={t.value}
          label={t.label}
          size="small"
          onClick={() => onChange(t.value)}
          sx={{
            fontWeight: 700, fontSize: '0.7rem',
            bgcolor: value === t.value ? 'primary.main' : 'action.hover',
            color: value === t.value ? 'white' : 'text.primary',
            '&:hover': { bgcolor: value === t.value ? 'primary.dark' : 'action.selected' }
          }}
        />
      ))}
    </Box>
  );

  const isDark = theme.palette.mode === 'dark';

  return (
    <Box sx={{ p: 2, height: '100%', overflowY: 'auto', bgcolor: 'background.paper' }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Box display="flex" alignItems="center" gap={1}>
          <NotesIcon sx={{ color: 'primary.main' }} />
          <Typography variant="subtitle1" fontWeight={800}>Notes</Typography>
        </Box>
        <Box display="flex" alignItems="center" gap={1}>
          <Tooltip title="Filter by Exam Notes">
            <Chip
              label="Exam Mode"
              size="small"
              onClick={() => setExamMode(!examMode)}
              color={examMode ? "primary" : "default"}
              sx={{ fontWeight: 700, fontSize: '0.7rem' }}
            />
          </Tooltip>
          <Button
            size="small" variant="contained"
            onClick={() => setCreating(true)}
            sx={{ borderRadius: 2, fontWeight: 700, minWidth: 0, px: 1.5 }}
          >
            Add
          </Button>
        </Box>
      </Box>

      {/* New Note Form */}
      {creating && (
        <Paper elevation={0} sx={{ mb: 2, p: 2, borderRadius: 2, border: '2px solid', borderColor: 'primary.main', bgcolor: isDark ? 'rgba(30,40,50,0.9)' : '#fff' }}>
          <TypeSelector value={noteType} onChange={setNoteType} />
          <Box sx={{ my: 1.5 }}>
            <TextField fullWidth size="small" placeholder="Note Title" value={title} onChange={e => setTitle(e.target.value)} sx={{ mb: 1, '& .MuiOutlinedInput-root': { borderRadius: 1.5 } }} />
            <TextField fullWidth multiline rows={4} placeholder="Type here or drag text from PDF..." value={content} onChange={e => setContent(e.target.value)} onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); const droppedText = e.dataTransfer.getData('text'); if (droppedText) { setContent(prev => prev + (prev ? '\n\n' : '') + droppedText); } }} sx={{ mb: 1, '& .MuiOutlinedInput-root': { borderRadius: 1.5, fontSize: '0.85rem' } }} />
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
              <ColorPicker value={color} onChange={setColor} />
              <Tooltip title="Grab selected text from PDF">
                <IconButton size="small" onClick={() => handleGrabText(setContent)} sx={{ color: 'primary.main' }}>
                  <ContentPasteIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
          <Box display="flex" gap={1} justifyContent="flex-end">
            <Button size="small" onClick={() => { setCreating(false); setContent(''); setTitle(''); }} color="inherit" sx={{ borderRadius: 2 }}>Cancel</Button>
            <Button size="small" variant="contained" onClick={handleCreate} disabled={!content.trim()} sx={{ borderRadius: 2, fontWeight: 700 }}>Save</Button>
          </Box>
        </Paper>
      )}

      {/* Loading & Empty State */}
      {loading && <Box display="flex" justifyContent="center" py={4}><CircularProgress size={24} /></Box>}
      {!loading && notes.length === 0 && !creating && (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <NotesIcon sx={{ fontSize: 48, color: 'text.disabled', opacity: 0.5, mb: 1 }} />
          <Typography variant="body2" color="text.secondary">No notes yet. Add one!</Typography>
        </Box>
      )}

      {/* Notes List */}
      {!loading && notes.length > 0 && (
        <Stack spacing={2}>
          {notes.filter(n => !examMode || n.note_type === 'exam').map(note => (
            <Box key={note.id}>
              {editingId === note.id ? (
                <Paper elevation={0} sx={{ p: 2, borderRadius: 2, border: '2px solid', borderColor: 'primary.main', bgcolor: isDark ? 'rgba(30,40,55,0.95)' : '#fff' }}>
                  <TypeSelector value={editNoteType} onChange={setEditNoteType} />
                  <Box sx={{ my: 1 }}>
                    <TextField fullWidth size="small" value={editTitle} onChange={e => setEditTitle(e.target.value)} sx={{ mb: 1 }} />
                    <TextField fullWidth multiline rows={4} value={editContent} onChange={e => setEditContent(e.target.value)} sx={{ mb: 1, fontSize: '0.85rem' }} />
                  </Box>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                    <ColorPicker value={editColor} onChange={setEditColor} />
                    <IconButton size="small" onClick={() => handleGrabText(setEditContent)} sx={{ color: 'primary.main' }}>
                      <ContentPasteIcon fontSize="small" />
                    </IconButton>
                  </Box>
                  <Box display="flex" justifyContent="flex-end" gap={1}>
                    <Button size="small" onClick={() => setEditingId(null)} color="inherit">Cancel</Button>
                    <Button size="small" variant="contained" onClick={() => handleSaveEdit(note.id)} sx={{ borderRadius: 2, fontWeight: 700 }}>Save</Button>
                  </Box>
                </Paper>
              ) : (
                <Paper elevation={0} sx={{ borderRadius: 2, border: '1px solid', borderColor: 'divider', overflow: 'hidden', display: 'flex', flexDirection: 'column', transition: 'box-shadow 0.2s', '&:hover': { boxShadow: '0 4px 12px rgba(0,0,0,0.08)' } }}>
                  <Box sx={{ bgcolor: note.color, px: 1.5, py: 1, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                      <Typography variant="subtitle2" fontWeight={800} sx={{ color: 'rgba(0,0,0,0.8)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{note.title || 'Untitled'}</Typography>
                      <Typography variant="caption" sx={{ color: 'rgba(0,0,0,0.6)', fontWeight: 700, textTransform: 'uppercase', fontSize: '0.65rem' }}>{note.note_type}</Typography>
                    </Box>
                    <Box display="flex" gap={0}>
                      <IconButton size="small" onClick={() => handleStartEdit(note)} sx={{ p: 0.5, color: 'rgba(0,0,0,0.5)', '&:hover': { color: 'rgba(0,0,0,0.8)' } }}><EditIcon sx={{ fontSize: 16 }} /></IconButton>
                      <IconButton size="small" onClick={() => handleDelete(note.id)} sx={{ p: 0.5, color: 'rgba(0,0,0,0.5)', '&:hover': { color: '#ef4444' } }}><DeleteIcon sx={{ fontSize: 16 }} /></IconButton>
                    </Box>
                  </Box>
                  <Box sx={{ p: 1.5, fontSize: '0.85rem', lineHeight: 1.6, color: 'text.secondary', '& p': { m: 0, mb: 0.5 } }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{note.content}</ReactMarkdown>
                  </Box>
                </Paper>
              )}
            </Box>
          ))}
        </Stack>
      )}
    </Box>
  );
};

export default NotesSidebar;
