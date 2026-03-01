// src/pages/Dashboard.js
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  LinearProgress,
  Avatar,
  Chip,
  useTheme,
  Menu,
  MenuItem,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Checkbox,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  CircularProgress,
  Container,
} from "@mui/material";
import {
  Notifications as NotificationsIcon,
  Schedule as ScheduleIcon,
  Quiz as QuizIcon,
  School as SchoolIcon,
  Analytics as AnalyticsIcon,
  Psychology as PsychologyIcon,
  UploadFile as UploadFileIcon,
  AutoStories as AutoStoriesIcon,
  PlayCircle as PlayCircleIcon,
  Check as CheckIcon,
  Upload as UploadIcon,
  Warning as WarningIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckCircleIcon,
  Whatshot as WhatshotIcon,
  Assignment as AssignmentIcon,
  Add as AddIcon 
} from "@mui/icons-material";
import { useAuth } from "../context/AuthContext";
import API from "../api/api";

const StatCard = ({ title, value, subtext, icon, color }) => (
  <Card sx={{ 
      height: '100%',
      minHeight: 160,
      bgcolor: 'background.paper', 
      borderRadius: '16px', 
      border: '1px solid', 
      borderColor: 'divider',
      boxShadow: 'none',
      position: 'relative'
    }}>
    <CardContent sx={{ p: '24px !important', display: 'flex', flexDirection: 'column', gap: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 1 }}>
        <Box sx={{ 
            width: 48,
            height: 48,
            borderRadius: '12px', 
            bgcolor: (theme) => `rgba(${theme.palette.mode === 'dark' ? '255,255,255,0.05' : '0,0,0,0.05'})`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: color || 'primary.main',
            flexShrink: 0
        }}>
            {icon}
        </Box>
        {subtext && (
          <Chip 
              label={subtext} 
              size="small" 
              sx={{ 
                  bgcolor: 'rgba(11, 218, 91, 0.15)', 
                  color: '#0bda5b', 
                  fontWeight: 700, 
                  fontSize: '0.7rem',
                  height: 24,
                  borderRadius: '6px',
                  maxWidth: '100%',
                  '& .MuiChip-label': {
                      px: 1,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                  }
              }} 
          />
        )}
      </Box>
      <Box sx={{ mt: 'auto', display: 'flex', flexDirection: 'column' }}>
        <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary', mb: 0.5, lineHeight: 1 }}>
            {value}
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 600 }}>
            {title}
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

const QuickActionCard = ({ title, subtitle, icon, color, onClick }) => (
  <Box 
    onClick={onClick}
    sx={{ 
      p: 3, 
      minHeight: 180,
      borderRadius: '16px', 
      border: '1px solid', 
      borderColor: 'divider', 
      bgcolor: 'background.paper',
      cursor: 'pointer',
      transition: 'all 0.2s',
      position: 'relative',
      overflow: 'hidden',
      display: 'flex', 
      flexDirection: 'column', 
      gap: 2,
      '&:hover': {
          borderColor: 'primary.main',
          bgcolor: 'rgba(19, 127, 236, 0.05)',
          transform: 'translateY(-2px)'
      }
    }}
  >
      <Box sx={{ position: 'absolute', top: 20, right: 20, color: 'primary.main', opacity: 0.8 }}>
           {React.cloneElement(icon, { sx: { fontSize: 32 } })}
      </Box>

      <Box sx={{ flex: 1 }} />
      <Box sx={{ zIndex: 1, pr: 4 }}>
          <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary', mb: 0.5, lineHeight: 1.2 }}>{title}</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.4 }}>{subtitle}</Typography>
      </Box>
  </Box>
);

const MasteryBar = ({ subject, percentage, color }) => (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 600 }}>{subject}</Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 700 }}>{percentage}%</Typography>
        </Box>
        <LinearProgress 
            variant="determinate" 
            value={percentage} 
            sx={{ 
                height: 6, 
                borderRadius: 3, 
                bgcolor: 'rgba(255,255,255,0.05)',
                '& .MuiLinearProgress-bar': { bgcolor: color, borderRadius: 3 }
            }} 
        />
    </Box>
);

const WeakTopicItem = ({ topic, subject, accuracy, noteId, onPlay }) => (
    <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 2, 
        borderRadius: '12px', 
        bgcolor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        '&:hover': {
            borderColor: 'primary.main',
            bgcolor: 'rgba(19, 127, 236, 0.05)'
        },
        transition: 'all 0.2s'
    }}>
        <Box>
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>{topic}</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>{subject} • Accuracy {accuracy}%</Typography>
        </Box>
        <IconButton 
            size="small" 
            sx={{ color: 'primary.main', bgcolor: 'rgba(19, 127, 236, 0.1)', '&:hover': { bgcolor: 'rgba(19, 127, 236, 0.2)' } }}
            onClick={() => onPlay && onPlay(topic, subject, noteId)}
        >
            <PlayCircleIcon fontSize="small" />
        </IconButton>
    </Box>
);

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const theme = useTheme();
  
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [notifications, setNotifications] = useState([]);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [todoOpen, setTodoOpen] = useState(false);
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState("");
  const [weakTopicDialog, setWeakTopicDialog] = useState({ open: false, topic: null, subject: null, noteId: null, data: null, loading: false });
  const [practiceAllDialog, setPracticeAllDialog] = useState({ open: false, questionCount: 10, selectedTopics: [] });

  useEffect(() => {
      fetchStats();
      fetchNotifications();
      loadTodos();
      
      // Auto-refresh when component mounts or when navigating back
      const handleVisibilityChange = () => {
          if (!document.hidden) {
              fetchStats();
          }
      };
      
      document.addEventListener('visibilitychange', handleVisibilityChange);
      
      return () => {
          document.removeEventListener('visibilitychange', handleVisibilityChange);
      };
  }, []);

  const fetchStats = async () => {
      try {
          const response = await API.get('dashboard/stats/');
          setStats(response.data);
      } catch (error) {
          console.error("Failed to fetch dashboard stats", error);
      } finally {
          setLoading(false);
      }
  };

  const fetchNotifications = async () => {
      try {
          const response = await API.get('notifications/');
          setNotifications(response.data);
      } catch (error) {
          console.error("Failed to fetch notifications", error);
      }
  };

  const loadTodos = () => {
      const saved = localStorage.getItem('dashboard_todos');
      if (saved) {
          setTodos(JSON.parse(saved));
      }
  };

  const saveTodos = (newTodos) => {
      localStorage.setItem('dashboard_todos', JSON.stringify(newTodos));
      setTodos(newTodos);
  };

  const addTodo = () => {
      if (newTodo.trim()) {
          const newTodos = [...todos, { id: Date.now(), text: newTodo, completed: false }];
          saveTodos(newTodos);
          setNewTodo("");
      }
  };

  const toggleTodo = (id) => {
      const newTodos = todos.map(t => t.id === id ? { ...t, completed: !t.completed } : t);
      saveTodos(newTodos);
  };

  const deleteTodo = (id) => {
      const newTodos = todos.filter(t => t.id !== id);
      saveTodos(newTodos);
  };

  const markNotificationRead = async (id) => {
      try {
          await API.post(`notifications/${id}/mark-read/`);
          fetchNotifications();
      } catch (error) {
          console.error("Failed to mark notification as read", error);
      }
  };

  const markAllNotificationsRead = async () => {
      try {
          await API.post('notifications/mark-all-read/');
          fetchNotifications();
      } catch (error) {
          console.error("Failed to mark all notifications as read", error);
      }
  };

  const handleWeakTopicExplain = async (topic, subject, noteId) => {
      setWeakTopicDialog({ open: true, topic, subject, noteId, data: null, loading: true });
      
      try {
          // Add timeout of 20 seconds
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 20000);
          
          const response = await API.post('weak-topic/explain/', 
              { topic, subject },
              { signal: controller.signal }
          );
          
          clearTimeout(timeoutId);
          setWeakTopicDialog(prev => ({ ...prev, data: response.data.data, loading: false }));
      } catch (error) {
          console.error("Failed to fetch weak topic explanation", error);
          
          let errorMessage = "Failed to load explanation. ";
          if (error.name === 'AbortError' || error.code === 'ECONNABORTED') {
              errorMessage = "Request timed out. The AI is taking too long to respond. ";
          } else if (error.response?.status === 401) {
              errorMessage = "Please log in to access this feature. ";
          } else if (error.response?.status === 500) {
              errorMessage = "Server error. Please try again. ";
          }
          
          setWeakTopicDialog(prev => ({ 
              ...prev, 
              loading: false, 
              data: { 
                  error: errorMessage,
                  canRetry: true
              } 
          }));
      }
  };

  const handleRetryExplanation = () => {
      if (weakTopicDialog.topic && weakTopicDialog.subject) {
          handleWeakTopicExplain(weakTopicDialog.topic, weakTopicDialog.subject, weakTopicDialog.noteId);
      }
  };

  const handleStartPractice = (topic) => {
      const nid = weakTopicDialog.noteId;
      setWeakTopicDialog({ open: false, topic: null, subject: null, noteId: null, data: null, loading: false });
      if (nid) {
          navigate(`/quiz-mode?noteId=${nid}&n=10`);
      } else {
          // Fallback if no noteId (shouldn't happen with new backend)
          navigate('/quiz', { state: { topic } });
      }
  };

  const handlePracticeAll = () => {
      // Initialize with all weak topics selected by default
      if (!data?.weak_topics?.length) return;
      const allTopics = data.weak_topics.map(w => w.topic);
      setPracticeAllDialog({ open: true, questionCount: 10, selectedTopics: allTopics });
  };

  const handleStartPracticeAll = () => {
      // Find note IDs for selected topics
      const selectedNoteIds = data.weak_topics
          .filter(w => practiceAllDialog.selectedTopics.includes(w.topic) && w.note_id)
          .map(w => w.note_id);
      
      const uniqueNoteIds = [...new Set(selectedNoteIds)];
      
      setPracticeAllDialog({ open: false, questionCount: 10, selectedTopics: [] });
      
      if (uniqueNoteIds.length > 0) {
          const idsParam = uniqueNoteIds.join(',');
          const n = practiceAllDialog.questionCount || 10;
          navigate(`/quiz-mode?noteIds=${idsParam}&n=${n}`, { 
              state: { 
                  weakTopics: practiceAllDialog.selectedTopics, // Pass for tracking/validation if needed
                  isPracticeAll: true
              } 
          });
      } else {
          // Fallback to old behavior if no note IDs found (e.g. general topics)
           navigate('/quiz', { 
              state: { 
                  weakTopics: practiceAllDialog.selectedTopics, 
                  questionCount: practiceAllDialog.questionCount,
                  isPracticeAll: true
              } 
          });
      }
  };

  const handleToggleTopic = (topic) => {
      setPracticeAllDialog(prev => {
          const isSelected = prev.selectedTopics.includes(topic);
          const newSelected = isSelected
              ? prev.selectedTopics.filter(t => t !== topic)
              : [...prev.selectedTopics, topic];
          return { ...prev, selectedTopics: newSelected };
      });
  };

  // Defaults if loading or error
  const data = stats || {
      study_time: "0h 0m",
      questions_answered: 0,
      topics_mastered: 0,
      avg_score: 0,
      streak: 0,
      mastery_data: [],
      weak_topics: [],
      recent_activity: []
  };

  const unreadCount = notifications.filter(n => !n.is_read).length;

  const subjectColors = ['#8B5CF6', '#F43F5E', '#F59E0B', '#10B981', '#3B82F6', '#06B6D4'];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {/* Header */}
      <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2, mb: 1 }}>
        <Box>
            <Typography variant="h2" sx={{ fontWeight: 900, color: 'text.primary', letterSpacing: '-0.02em', mb: 1 }}>
                Welcome back, {user?.first_name || 'Student'}!
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <WhatshotIcon sx={{ color: '#0bda5b', fontSize: 24 }} />
                <Typography variant="body1" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                    You're on a <Box component="span" sx={{ color: '#0bda5b', fontWeight: 700 }}>{data.streak}-day streak</Box>. Keep it up!
                </Typography>
            </Box>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, mt: { xs: 2, md: 0 } }}>
            <Button 
                variant="outlined" 
                startIcon={<NotificationsIcon />}
                onClick={(e) => setNotificationAnchor(e.currentTarget)}
                sx={{ 
                    borderColor: 'divider', 
                    color: 'text.primary', 
                    bgcolor: 'background.paper',
                    textTransform: 'none',
                    fontWeight: 700,
                    borderRadius: '12px',
                    display: { xs: 'none', sm: 'flex' },
                    '&:hover': { bgcolor: 'rgba(255,255,255,0.05)', borderColor: 'divider' }
                }}
            >
                <Badge badgeContent={unreadCount} color="error" variant="dot">
                    Updates
                </Badge>
            </Button>
            <Button 
                variant="contained" 
                onClick={() => setTodoOpen(true)}
                sx={{ 
                    borderRadius: '12px', 
                    fontWeight: 700, 
                    boxShadow: 'none',
                    textTransform: 'none',
                    px: 3
                }}
            >
                Daily Goal Check-in
            </Button>
        </Box>
      </Box>

      {/* Notifications Menu */}
      <Menu
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={() => setNotificationAnchor(null)}
        PaperProps={{
            sx: { width: 360, maxHeight: 400 }
        }}
      >
        <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'divider', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6" sx={{ fontWeight: 700 }}>Notifications</Typography>
            {unreadCount > 0 && (
                <Button 
                    size="small" 
                    onClick={markAllNotificationsRead}
                    sx={{ textTransform: 'none', fontSize: '0.75rem' }}
                >
                    Mark all as read
                </Button>
            )}
        </Box>
        {notifications.length === 0 ? (
            <MenuItem disabled>
                <Typography variant="body2" color="text.secondary">No notifications</Typography>
            </MenuItem>
        ) : (
            notifications.map((notif) => (
                <MenuItem 
                    key={notif.id} 
                    onClick={() => markNotificationRead(notif.id)}
                    sx={{ 
                        bgcolor: notif.is_read ? 'transparent' : 'action.hover',
                        whiteSpace: 'normal',
                        py: 1.5
                    }}
                >
                    <Box>
                        <Typography variant="body2" sx={{ fontWeight: notif.is_read ? 400 : 600 }}>
                            {notif.message}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                            {new Date(notif.created_at).toLocaleString()}
                        </Typography>
                    </Box>
                </MenuItem>
            ))
        )}
      </Menu>

      {/* To-Do List Dialog */}
      <Dialog open={todoOpen} onClose={() => setTodoOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Daily Goals & To-Do List</DialogTitle>
        <DialogContent>
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <TextField
                    fullWidth
                    size="small"
                    placeholder="Add a new task..."
                    value={newTodo}
                    onChange={(e) => setNewTodo(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && addTodo()}
                />
                <Button variant="contained" onClick={addTodo} startIcon={<AddIcon />}>
                    Add
                </Button>
            </Box>
            <List>
                {todos.map((todo) => (
                    <ListItem
                        key={todo.id}
                        secondaryAction={
                            <IconButton edge="end" onClick={() => deleteTodo(todo.id)}>
                                <DeleteIcon />
                            </IconButton>
                        }
                        sx={{ 
                            bgcolor: 'background.paper', 
                            mb: 1, 
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider'
                        }}
                    >
                        <ListItemIcon>
                            <Checkbox
                                checked={todo.completed}
                                onChange={() => toggleTodo(todo.id)}
                            />
                        </ListItemIcon>
                        <ListItemText 
                            primary={todo.text}
                            sx={{ 
                                textDecoration: todo.completed ? 'line-through' : 'none',
                                color: todo.completed ? 'text.secondary' : 'text.primary'
                            }}
                        />
                    </ListItem>
                ))}
                {todos.length === 0 && (
                    <Typography variant="body2" color="text.secondary" align="center" sx={{ py: 4 }}>
                        No tasks yet. Add your first goal!
                    </Typography>
                )}
            </List>
        </DialogContent>
        <DialogActions>
            <Button onClick={() => setTodoOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>

      {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
              <CircularProgress />
          </Box>
      ) : (
          <>
              {/* Stats Grid */}
              <Grid container spacing={3}>
                  <Grid item xs={6} md={3}>
                      <StatCard 
                        title="Current Streak" 
                        value={`${data.streak || 0} Days`} 
                        subtext="keep it up!" 
                        color="#ff5722" 
                        icon={<WhatshotIcon sx={{ fontSize: 24 }} />} 
                      />
                  </Grid>
                  <Grid item xs={6} md={3}>
                      <StatCard 
                        title="Questions Answered" 
                        value={data.questions_answered} 
                        subtext="total" 
                        color="#137fec" 
                        icon={<QuizIcon sx={{ fontSize: 24 }} />} 
                      />
                  </Grid>
                  <Grid item xs={6} md={3}>
                      <StatCard 
                        title="Topics Mastered" 
                        value={data.topics_mastered} 
                        subtext=">80% mastery" 
                        color="#0bda5b" 
                        icon={<SchoolIcon sx={{ fontSize: 24 }} />} 
                      />
                  </Grid>
                  <Grid item xs={6} md={3}>
                      <StatCard 
                        title="Avg. Quiz Score" 
                        value={`${data.avg_score}%`} 
                        subtext="lifetime" 
                        color="#f59e0b" 
                        icon={<AnalyticsIcon sx={{ fontSize: 24 }} />} 
                      />
                  </Grid>
              </Grid>

              {/* Quick Actions */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>Quick Actions</Typography>
                  </Box>
                  <Grid container spacing={3}>
                      <Grid item xs={12} sm={6} md={3}>
                          <QuickActionCard 
                            title="Start Adaptive Quiz" 
                            subtitle="Test your knowledge on weak areas and get instant feedback." 
                            icon={<PsychologyIcon />} 
                            color="primary"
                            onClick={() => navigate('/quiz')}
                          />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                          <QuickActionCard 
                            title="Upload Lecture Notes" 
                            subtitle="AI parses your PDFs to generate summaries and flashcards." 
                            icon={<UploadFileIcon />} 
                            color="primary"
                            onClick={() => navigate('/lectures')}
                          />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                           <QuickActionCard 
                            title="Generate Study Guide" 
                            subtitle="Create a personalized study plan for your upcoming exams." 
                            icon={<AutoStoriesIcon />} 
                            color="primary"
                            onClick={() => navigate('/analysis')}
                          />
                      </Grid>
                      <Grid item xs={12} sm={6} md={3}>
                           <QuickActionCard 
                            title="Exam Preparation" 
                            subtitle="Generate strategies and practice with past papers." 
                            icon={<AssignmentIcon />} 
                            color="primary"
                            onClick={() => navigate('/exam-preparation')}
                          />
                      </Grid>
                  </Grid>
              </Box>

              {/* Bottom Section: Mastery & Weak Topics/Activity */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="h5" sx={{ fontWeight: 700, color: 'text.primary' }}>Performance Overview</Typography>
                  </Box>
                  <Grid container spacing={3}>
                  {/* Subject Mastery */}
                  <Grid item xs={12} md={4}>
                      <Card sx={{ height: '100%', minHeight: 400 }}>
                          <CardContent sx={{ p: '24px !important', display: 'flex', flexDirection: 'column', height: '100%', gap: 3 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 2 }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                      <Avatar sx={{ bgcolor: 'rgba(139, 92, 246, 0.1)', color: '#8B5CF6', width: 40, height: 40, borderRadius: '10px' }}>
                                          <SchoolIcon />
                                      </Avatar>
                                      <Box>
                                          <Typography variant="h6" sx={{ fontWeight: 800, color: 'text.primary', lineHeight: 1.2 }}>Subject Mastery</Typography>
                                          <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>Average progress</Typography>
                                      </Box>
                                  </Box>
                                  <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary' }}>
                                      {data.mastery_data.length > 0 ? Math.round(data.mastery_data.reduce((acc, curr) => acc + curr.percentage, 0) / data.mastery_data.length) : 0}%
                                  </Typography>
                              </Box>
                              
                              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, justifyContent: 'center', flex: 1 }}>
                                  {data.mastery_data.length > 0 ? (
                                      data.mastery_data.map((m, idx) => (
                                          <MasteryBar key={idx} subject={m.subject} percentage={m.percentage} color={subjectColors[idx % subjectColors.length]} />
                                      ))
                                  ) : (
                                      <Typography variant="body2" color="text.secondary" align="center">No mastery data yet. Take some quizzes!</Typography>
                                  )}
                              </Box>
                          </CardContent>
                      </Card>
                  </Grid>

                  {/* Weak Topics */}
                  <Grid item xs={12} md={4}>
                      <Card sx={{ height: '100%', minHeight: 400 }}>
                          <CardContent sx={{ p: '24px !important', display: 'flex', flexDirection: 'column', height: '100%', gap: 3 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 2 }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                      <Avatar sx={{ bgcolor: 'rgba(248, 113, 113, 0.1)', color: '#F87171', width: 40, height: 40, borderRadius: '10px' }}>
                                          <WarningIcon />
                                      </Avatar>
                                      <Typography variant="h6" sx={{ fontWeight: 800, color: 'text.primary', lineHeight: 1.2 }}>Weak Topics</Typography>
                                  </Box>
                                  <Button 
                                    size="small" 
                                    variant="outlined" 
                                    onClick={handlePracticeAll}
                                    sx={{ borderRadius: '8px', textTransform: 'none', fontWeight: 700, borderColor: 'primary.main', color: 'primary.main', px: 2, py: 0.5 }}
                                  >
                                      Practice All
                                  </Button>
                              </Box>
                              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, flex: 1 }}>
                                  {data.weak_topics.length > 0 ? (
                                      data.weak_topics.map((w, idx) => (
                                          <WeakTopicItem 
                                              key={idx} 
                                              topic={w.topic} 
                                              subject={w.subject} 
                                              accuracy={w.accuracy} 
                                              noteId={w.note_id}
                                              onPlay={handleWeakTopicExplain}
                                          />
                                      ))
                                  ) : (
                                      <Typography variant="body2" color="text.secondary" align="center" sx={{ my: 'auto' }}>Great job! No weak topics detected.</Typography>
                                  )}
                              </Box>
                          </CardContent>
                      </Card>
                  </Grid>

                  {/* Recent Activity */}
                  <Grid item xs={12} md={4}>
                      <Card sx={{ height: '100%', minHeight: 400 }}>
                            <CardContent sx={{ p: '24px !important', display: 'flex', flexDirection: 'column', height: '100%', gap: 3 }}>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                  <Avatar sx={{ bgcolor: 'rgba(16, 185, 129, 0.1)', color: '#10B981', width: 40, height: 40, borderRadius: '10px' }}>
                                      <ScheduleIcon />
                                  </Avatar>
                                  <Typography variant="h6" sx={{ fontWeight: 800, color: 'text.primary', lineHeight: 1.2 }}>Recent Activity</Typography>
                              </Box>
                              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, flex: 1, mt: 1 }}>
                                  {data.recent_activity.length > 0 ? (
                                      data.recent_activity.map((act, idx) => (
                                          <Box key={idx} sx={{ display: 'flex', gap: 2, alignItems: 'flex-start' }}>
                                              <Avatar sx={{ width: 36, height: 36, bgcolor: act.type === 'upload' ? 'rgba(19, 127, 236, 0.1)' : 'rgba(16, 185, 129, 0.1)', color: act.type === 'upload' ? 'primary.main' : '#10B981' }}>
                                                  {act.type === 'upload' ? <UploadIcon sx={{ fontSize: 18 }} /> : <CheckIcon sx={{ fontSize: 18 }} />}
                                              </Avatar>
                                              <Box>
                                                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>{act.text}</Typography>
                                                  <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 500 }}>{act.subtext}</Typography>
                                              </Box>
                                          </Box>
                                      ))
                                  ) : (
                                      <Typography variant="body2" color="text.secondary" align="center" sx={{ my: 'auto' }}>No recent activity.</Typography>
                                  )}
                              </Box>
                            </CardContent>
                      </Card>
                  </Grid>
              </Grid>
              </Box>

          {/* Practice All Dialog */}
          <Dialog 
              open={practiceAllDialog.open} 
              onClose={() => setPracticeAllDialog({ open: false, questionCount: 10 })}
              maxWidth="sm" 
              fullWidth
          >
              <DialogTitle>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <QuizIcon sx={{ color: 'primary.main' }} />
                      <Typography variant="h6" sx={{ fontWeight: 700 }}>
                          Practice Weak Topics
                      </Typography>
                  </Box>
              </DialogTitle>
              <DialogContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 1 }}>
                      {/* Weak Topics Multi-Select */}
                      <Box>
                          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5, color: 'text.primary' }}>
                              Select topics to practice:
                          </Typography>
                          {data.weak_topics && data.weak_topics.length > 0 ? (
                              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                  {data.weak_topics.map((topic, idx) => {
                                      const isSelected = practiceAllDialog.selectedTopics.includes(topic.topic);
                                      return (
                                          <Box 
                                              key={idx} 
                                              onClick={() => handleToggleTopic(topic.topic)}
                                              sx={{ 
                                                  display: 'flex', 
                                                  alignItems: 'center', 
                                                  gap: 1,
                                                  p: 1.5,
                                                  borderRadius: '8px',
                                                  bgcolor: isSelected 
                                                      ? (theme) => theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.2)' : 'rgba(239, 246, 255, 1)'
                                                      : (theme) => theme.palette.mode === 'dark' ? 'rgba(35, 54, 72, 0.3)' : 'rgba(248, 250, 252, 1)',
                                                  border: '2px solid',
                                                  borderColor: isSelected ? 'primary.main' : 'divider',
                                                  cursor: 'pointer',
                                                  transition: 'all 0.2s',
                                                  '&:hover': {
                                                      borderColor: 'primary.main',
                                                      bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.15)' : 'rgba(239, 246, 255, 0.8)'
                                                  }
                                              }}
                                          >
                                              <Checkbox 
                                                  checked={isSelected}
                                                  onChange={() => handleToggleTopic(topic.topic)}
                                                  sx={{ p: 0 }}
                                              />
                                              <Box sx={{ flex: 1 }}>
                                                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                                                      {topic.topic}
                                                  </Typography>
                                                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                      {topic.subject} • {topic.accuracy}% accuracy
                                                  </Typography>
                                              </Box>
                                          </Box>
                                      );
                                  })}
                              </Box>
                          ) : (
                              <Typography variant="body2" color="text.secondary">
                                  No weak topics found. Great job!
                              </Typography>
                          )}
                      </Box>

                      {/* Question Count Selector */}
                      <Box>
                          <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1.5, color: 'text.primary' }}>
                              Number of Questions:
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1.5 }}>
                              {[5, 10, 15, 20].map((count) => (
                                  <Button
                                      key={count}
                                      variant={practiceAllDialog.questionCount === count ? "contained" : "outlined"}
                                      onClick={() => setPracticeAllDialog(prev => ({ ...prev, questionCount: count }))}
                                      sx={{ 
                                          flex: 1,
                                          fontWeight: 700,
                                          textTransform: 'none'
                                      }}
                                  >
                                      {count}
                                  </Button>
                              ))}
                          </Box>
                      </Box>
                  </Box>
              </DialogContent>
              <DialogActions sx={{ px: 3, pb: 3 }}>
                  <Button 
                      onClick={() => setPracticeAllDialog({ open: false, questionCount: 10 })}
                      sx={{ textTransform: 'none' }}
                  >
                      Cancel
                  </Button>
                  <Button 
                      variant="contained" 
                      onClick={handleStartPracticeAll}
                      startIcon={<QuizIcon />}
                      disabled={!practiceAllDialog.selectedTopics || practiceAllDialog.selectedTopics.length === 0}
                      sx={{ textTransform: 'none', fontWeight: 700 }}
                  >
                      Start Practice
                  </Button>
              </DialogActions>
          </Dialog>

          {/* Weak Topic Explanation Dialog */}
          <Dialog 
              open={weakTopicDialog.open} 
              onClose={() => setWeakTopicDialog({ open: false, topic: null, subject: null, noteId: null, data: null, loading: false })}
              maxWidth="md" 
              fullWidth
          >
              <DialogTitle sx={{ pb: 1 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <PlayCircleIcon sx={{ color: 'primary.main' }} />
                      <Typography variant="h6" sx={{ fontWeight: 700 }}>
                          Learn: {weakTopicDialog.topic}
                      </Typography>
                  </Box>
                  {weakTopicDialog.subject && (
                      <Typography variant="caption" color="text.secondary">
                          {weakTopicDialog.subject}
                      </Typography>
                  )}
              </DialogTitle>
              <DialogContent>
                  {weakTopicDialog.loading ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, py: 6 }}>
                          <CircularProgress />
                          <Typography variant="body2" color="text.secondary">
                              Generating personalized explanation...
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                              This may take a few moments
                          </Typography>
                      </Box>
                  ) : weakTopicDialog.data?.error ? (
                      <Box sx={{ py: 4, textAlign: 'center', display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <Typography color="error" sx={{ mb: 1 }}>{weakTopicDialog.data.error}</Typography>
                          {weakTopicDialog.data.canRetry && (
                              <Button 
                                  variant="outlined" 
                                  onClick={handleRetryExplanation}
                                  sx={{ mx: 'auto' }}
                              >
                                  Try Again
                              </Button>
                          )}
                      </Box>
                  ) : weakTopicDialog.data ? (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                          {/* Explanation */}
                          {weakTopicDialog.data.explanation && (
                              <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                      Understanding the Topic
                                  </Typography>
                                  <Typography variant="body2" sx={{ color: 'text.secondary', whiteSpace: 'pre-line' }}>
                                      {weakTopicDialog.data.explanation}
                                  </Typography>
                              </Box>
                          )}

                          {/* Key Concepts */}
                          {weakTopicDialog.data.key_concepts && weakTopicDialog.data.key_concepts.length > 0 && (
                              <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                      Key Concepts
                                  </Typography>
                                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                      {weakTopicDialog.data.key_concepts.map((concept, idx) => (
                                          <Box key={idx} sx={{ display: 'flex', gap: 1 }}>
                                              <CheckCircleIcon sx={{ fontSize: 20, color: 'primary.main', mt: 0.2 }} />
                                              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                                  {concept}
                                              </Typography>
                                          </Box>
                                      ))}
                                  </Box>
                              </Box>
                          )}

                          {/* Common Mistakes */}
                          {weakTopicDialog.data.common_mistakes && weakTopicDialog.data.common_mistakes.length > 0 && (
                              <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                      Common Mistakes to Avoid
                                  </Typography>
                                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                      {weakTopicDialog.data.common_mistakes.map((mistake, idx) => (
                                          <Box key={idx} sx={{ display: 'flex', gap: 1 }}>
                                              <WarningIcon sx={{ fontSize: 20, color: '#F87171', mt: 0.2 }} />
                                              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                                  {mistake}
                                              </Typography>
                                          </Box>
                                      ))}
                                  </Box>
                              </Box>
                          )}

                          {/* Practice Tips */}
                          {weakTopicDialog.data.practice_tips && weakTopicDialog.data.practice_tips.length > 0 && (
                              <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                      Practice Tips
                                  </Typography>
                                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                                      {weakTopicDialog.data.practice_tips.map((tip, idx) => (
                                          <Box key={idx} sx={{ 
                                              p: 2, 
                                              borderRadius: '8px', 
                                              bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(35, 54, 72, 0.5)' : 'rgba(239, 246, 255, 1)',
                                              border: '1px solid',
                                              borderColor: 'divider'
                                          }}>
                                              <Typography variant="body2" sx={{ color: 'text.primary', fontWeight: 600 }}>
                                                  {idx + 1}. {tip}
                                              </Typography>
                                          </Box>
                                      ))}
                                  </Box>
                              </Box>
                          )}

                          {/* Resources */}
                          {weakTopicDialog.data.resources && weakTopicDialog.data.resources.length > 0 && (
                              <Box>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                      Learning Resources
                                  </Typography>
                                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                                      {weakTopicDialog.data.resources.map((resource, idx) => (
                                          <Box key={idx} sx={{ 
                                              p: 2, 
                                              borderRadius: '8px', 
                                              bgcolor: 'background.paper',
                                              border: '1px solid',
                                              borderColor: 'divider'
                                          }}>
                                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                                  {resource.type === 'article' && <AutoStoriesIcon sx={{ fontSize: 18, color: 'primary.main' }} />}
                                                  {resource.type === 'video' && <PlayCircleIcon sx={{ fontSize: 18, color: 'primary.main' }} />}
                                                  {resource.type === 'practice' && <QuizIcon sx={{ fontSize: 18, color: 'primary.main' }} />}
                                                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.primary' }}>
                                                      {resource.title}
                                                  </Typography>
                                              </Box>
                                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                                  {resource.description}
                                              </Typography>
                                          </Box>
                                      ))}
                                  </Box>
                              </Box>
                          )}

                          {/* Study Approach */}
                          {weakTopicDialog.data.study_approach && (
                              <Box sx={{ 
                                  p: 3, 
                                  borderRadius: '12px', 
                                  bgcolor: (theme) => theme.palette.mode === 'dark' ? 'rgba(19, 127, 236, 0.1)' : 'rgba(239, 246, 255, 1)',
                                  border: '2px solid',
                                  borderColor: 'primary.main'
                              }}>
                                  <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'primary.main' }}>
                                      Recommended Study Approach
                                  </Typography>
                                  <Typography variant="body2" sx={{ color: 'text.secondary', whiteSpace: 'pre-line' }}>
                                      {weakTopicDialog.data.study_approach}
                                  </Typography>
                              </Box>
                          )}
                      </Box>
                  ) : null}
              </DialogContent>
              <DialogActions sx={{ px: 3, pb: 3 }}>
                  <Button 
                      onClick={() => setWeakTopicDialog({ open: false, topic: null, subject: null, noteId: null, data: null, loading: false })}
                      sx={{ textTransform: 'none' }}
                  >
                      Close
                  </Button>
                  <Button 
                      variant="contained" 
                      onClick={() => handleStartPractice(weakTopicDialog.topic)}
                      startIcon={<QuizIcon />}
                      sx={{ textTransform: 'none', fontWeight: 700 }}
                  >
                      Start Practice Quiz
                  </Button>
              </DialogActions>
          </Dialog>
          </>
      )}
    </Box>
  );
}
