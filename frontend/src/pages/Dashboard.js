// src/pages/Dashboard.js
import React, { useState, useEffect } from "react";
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  IconButton,
  Avatar,
  Chip,
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
  CircularProgress,
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
  Add as AddIcon,
} from "@mui/icons-material";
import { useAuth } from "../context/AuthContext";
import API from "../api/api";
import { LoadingSkeletonPack, SurfaceCard } from "../components/ui";
import { SUBJECT_COLORS } from "../theme";

/* ── Vibrant palette ─────────────────────────────────────────────────────── */
const VCOLORS = {
  streak: {
    bg: "linear-gradient(135deg,#F97316,#EF4444)",
    glow: "#F9731630",
    text: "#fff",
  },
  questions: {
    bg: "linear-gradient(135deg,#6366F1,#8B5CF6)",
    glow: "#6366F130",
    text: "#fff",
  },
  mastered: {
    bg: "linear-gradient(135deg,#10B981,#06B6D4)",
    glow: "#10B98130",
    text: "#fff",
  },
  score: {
    bg: "linear-gradient(135deg,#F59E0B,#EC4899)",
    glow: "#F59E0B30",
    text: "#fff",
  },
};

const StatCard = ({ title, value, subtext, icon, colorKey }) => {
  const cv = VCOLORS[colorKey] || VCOLORS.streak;
  return (
    <Card
      sx={{
        height: "100%",
        minHeight: 160,
        position: "relative",
        overflow: "hidden",
        borderRadius: "22px",
        border: "none",
        background: cv.bg,
        boxShadow: `0 8px 32px ${cv.glow}`,
        transition: "all 0.3s cubic-bezier(.4,0,.2,1)",
        "&:hover": {
          transform: "translateY(-6px) scale(1.02)",
          boxShadow: `0 18px 48px ${cv.glow}`,
        },
        "&::after": {
          content: '""',
          position: "absolute",
          top: "-40%",
          right: "-20%",
          width: "180px",
          height: "180px",
          borderRadius: "50%",
          background: "rgba(255,255,255,0.12)",
          pointerEvents: "none",
        },
      }}
    >
      <CardContent
        sx={{
          p: "24px !important",
          display: "flex",
          flexDirection: "column",
          height: "100%",
          position: "relative",
          zIndex: 1,
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            mb: "auto",
          }}
        >
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: "14px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              bgcolor: "rgba(255,255,255,0.2)",
              color: "#fff",
            }}
          >
            {icon}
          </Box>
          {subtext && (
            <Chip
              label={subtext}
              size="small"
              sx={{
                bgcolor: "rgba(255,255,255,0.22)",
                color: "#fff",
                fontWeight: 700,
                fontSize: "0.68rem",
                height: 22,
                borderRadius: "6px",
                "& .MuiChip-label": { px: 1 },
              }}
            />
          )}
        </Box>
        <Box sx={{ mt: 3 }}>
          <Typography
            variant="h3"
            sx={{
              fontWeight: 900,
              color: "#fff",
              mb: 0.5,
              lineHeight: 1,
              letterSpacing: "-0.02em",
              fontSize: { xs: "1.6rem", md: "2rem" },
            }}
          >
            {value}
          </Typography>
          <Typography
            variant="body2"
            sx={{ color: "rgba(255,255,255,0.8)", fontWeight: 600 }}
          >
            {title}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

const QA_COLORS = [
  {
    grad: "linear-gradient(135deg,#6366F1,#8B5CF6)",
    glow: "#6366F120",
    icon: "#6366F1",
  },
  {
    grad: "linear-gradient(135deg,#10B981,#06B6D4)",
    glow: "#10B98120",
    icon: "#10B981",
  },
  {
    grad: "linear-gradient(135deg,#F59E0B,#F97316)",
    glow: "#F59E0B20",
    icon: "#F59E0B",
  },
  {
    grad: "linear-gradient(135deg,#EC4899,#F43F5E)",
    glow: "#EC489920",
    icon: "#EC4899",
  },
];

const QuickActionCard = ({ title, subtitle, icon, onClick, colorIdx = 0 }) => {
  const c = QA_COLORS[colorIdx % QA_COLORS.length];
  return (
    <Box
      onClick={onClick}
      sx={{
        p: 3,
        minHeight: 190,
        borderRadius: "20px",
        border: "2px solid transparent",
        background: `linear-gradient(#fff,#fff) padding-box, ${c.grad} border-box`,
        bgcolor: "background.paper",
        cursor: "pointer",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        transition: "all 0.28s ease",
        "&:hover": {
          boxShadow: `0 16px 40px ${c.glow}`,
          transform: "translateY(-5px)",
          "& .qa-icon-box": {
            background: c.grad,
            color: "#fff",
            transform: "scale(1.1) rotate(5deg)",
          },
          "& .qa-arrow": { opacity: 1, transform: "translateX(0)" },
          "&::before": { opacity: 1 },
        },
        "&::before": {
          content: '""',
          position: "absolute",
          inset: 0,
          opacity: 0,
          background: `radial-gradient(circle at 30% 50%, ${c.glow} 0%, transparent 70%)`,
          transition: "opacity 0.3s ease",
          pointerEvents: "none",
        },
      }}
    >
      <Box
        className="qa-icon-box"
        sx={{
          width: 50,
          height: 50,
          borderRadius: "14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: `${c.icon}18`,
          color: c.icon,
          transition: "all 0.28s ease",
          mb: "auto",
          "& svg": { fontSize: "1.4rem" },
        }}
      >
        {icon}
      </Box>
      <Box sx={{ mt: 3 }}>
        <Typography
          variant="h6"
          sx={{
            fontWeight: 800,
            color: "text.primary",
            mb: 0.5,
            lineHeight: 1.25,
            fontSize: "0.95rem",
          }}
        >
          {title}
        </Typography>
        <Typography
          variant="body2"
          sx={{ color: "text.secondary", lineHeight: 1.5, fontSize: "0.8rem" }}
        >
          {subtitle}
        </Typography>
      </Box>
      <Box
        className="qa-arrow"
        sx={{
          position: "absolute",
          bottom: 16,
          right: 16,
          color: c.icon,
          opacity: 0,
          transform: "translateX(-8px)",
          transition: "all 0.25s ease",
          fontSize: "1.2rem",
          fontWeight: 900,
        }}
      >
        →
      </Box>
    </Box>
  );
};

const MasteryBar = ({ subject, percentage, color }) => (
  <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
    <Box
      sx={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <Typography
        variant="body2"
        sx={{ color: "text.primary", fontWeight: 600, fontSize: "0.82rem" }}
      >
        {subject}
      </Typography>
      <Typography
        variant="caption"
        sx={{ fontWeight: 800, color: color || "primary.main" }}
      >
        {percentage}%
      </Typography>
    </Box>
    <Box
      sx={{
        position: "relative",
        height: 8,
        borderRadius: 4,
        bgcolor: "rgba(15,23,42,0.06)",
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          left: 0,
          top: 0,
          bottom: 0,
          width: `${percentage}%`,
          borderRadius: 4,
          background: `linear-gradient(90deg, ${color || "#2563EB"}cc, ${color || "#2563EB"})`,
          transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
        }}
      />
    </Box>
  </Box>
);

const getSeverityColor = (pct) => {
  if (pct < 40) return { bg: 'rgba(239,68,68,0.12)', color: '#EF4444', label: 'Critical' };
  if (pct < 70) return { bg: 'rgba(245,158,11,0.12)', color: '#F59E0B', label: 'Needs Work' };
  return { bg: 'rgba(16,185,129,0.12)', color: '#10B981', label: 'Fair' };
};

const DashboardTopicChip = ({ topic, subject, accuracy, noteId, onPlay }) => {
  const sev = getSeverityColor(accuracy);
  return (
    <Tooltip
      title={
        <Box sx={{ p: 0.5, textAlign: 'center' }}>
          <Typography variant="subtitle2" fontWeight={700}>{topic}</Typography>
          <Typography variant="caption" display="block">{subject} • {accuracy}% accuracy</Typography>
          <Typography variant="caption" color="primary.light" sx={{ mt: 1, display: 'block', fontWeight: 600 }}>
            Click to practice & explain
          </Typography>
        </Box>
      }
      arrow
      placement="top"
    >
      <Chip
        label={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span style={{ fontWeight: 600 }}>{topic}</span>
            <span style={{ opacity: 0.7, fontSize: '0.75em' }}>{accuracy}%</span>
          </Box>
        }
        onClick={() => onPlay(topic, subject, noteId)}
        size="small"
        sx={{
          bgcolor: sev.bg,
          color: sev.color,
          border: `1px solid ${sev.color}40`,
          cursor: 'pointer',
          height: 26,
          '&:hover': {
            bgcolor: sev.bg,
            borderColor: sev.color,
            transform: 'translateY(-1px)',
            boxShadow: `0 4px 8px ${sev.color}30`,
          },
          transition: 'all 0.15s ease',
        }}
      />
    </Tooltip>
  );
};

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [statsError, setStatsError] = useState("");
  const [notifications, setNotifications] = useState([]);
  const [notificationAnchor, setNotificationAnchor] = useState(null);
  const [todoOpen, setTodoOpen] = useState(false);
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState("");
  const [weakTopicDialog, setWeakTopicDialog] = useState({
    open: false,
    topic: null,
    subject: null,
    noteId: null,
    data: null,
    loading: false,
  });
  const [practiceAllDialog, setPracticeAllDialog] = useState({
    open: false,
    questionCount: 10,
    selectedTopics: [],
  });

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

    document.addEventListener("visibilitychange", handleVisibilityChange);

    // Real-time polling every 30 seconds
    const interval = setInterval(() => {
      if (!document.hidden) {
        fetchStats();
      }
    }, 30000);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      clearInterval(interval);
    };
  }, []);

  const fetchStats = async () => {
    setLoading(true);
    setStatsError("");
    try {
      const response = await API.get("dashboard/stats/");
      setStats(response.data);
    } catch (error) {
      console.error("Failed to fetch dashboard stats", error);
      setStatsError("Could not load your dashboard statistics.");
    } finally {
      setLoading(false);
    }
  };

  const fetchNotifications = async () => {
    try {
      const response = await API.get("notifications/");
      setNotifications(response.data);
    } catch (error) {
      console.error("Failed to fetch notifications", error);
    }
  };

  const loadTodos = () => {
    const saved = localStorage.getItem("dashboard_todos");
    if (saved) {
      setTodos(JSON.parse(saved));
    }
  };

  const saveTodos = (newTodos) => {
    localStorage.setItem("dashboard_todos", JSON.stringify(newTodos));
    setTodos(newTodos);
  };

  const addTodo = () => {
    if (newTodo.trim()) {
      const newTodos = [
        ...todos,
        { id: Date.now(), text: newTodo, completed: false },
      ];
      saveTodos(newTodos);
      setNewTodo("");
    }
  };

  const toggleTodo = (id) => {
    const newTodos = todos.map((t) =>
      t.id === id ? { ...t, completed: !t.completed } : t,
    );
    saveTodos(newTodos);
  };

  const deleteTodo = (id) => {
    const newTodos = todos.filter((t) => t.id !== id);
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
      await API.post("notifications/mark-all-read/");
      fetchNotifications();
    } catch (error) {
      console.error("Failed to mark all notifications as read", error);
    }
  };

  const handleWeakTopicExplain = async (topic, subject, noteId) => {
    setWeakTopicDialog({
      open: true,
      topic,
      subject,
      noteId,
      data: null,
      loading: true,
    });

    try {
      // Add timeout of 20 seconds
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 20000);

      const response = await API.post(
        "weak-topic/explain/",
        { topic, subject },
        { signal: controller.signal },
      );

      clearTimeout(timeoutId);
      setWeakTopicDialog((prev) => ({
        ...prev,
        data: response.data.data,
        loading: false,
      }));
    } catch (error) {
      console.error("Failed to fetch weak topic explanation", error);

      let errorMessage = "Failed to load explanation. ";
      if (error.name === "AbortError" || error.code === "ECONNABORTED") {
        errorMessage =
          "Request timed out. The AI is taking too long to respond. ";
      } else if (error.response?.status === 401) {
        errorMessage = "Please log in to access this feature. ";
      } else if (error.response?.status === 500) {
        errorMessage = "Server error. Please try again. ";
      }

      setWeakTopicDialog((prev) => ({
        ...prev,
        loading: false,
        data: {
          error: errorMessage,
          canRetry: true,
        },
      }));
    }
  };

  const handleRetryExplanation = () => {
    if (weakTopicDialog.topic && weakTopicDialog.subject) {
      handleWeakTopicExplain(
        weakTopicDialog.topic,
        weakTopicDialog.subject,
        weakTopicDialog.noteId,
      );
    }
  };

  const handleStartPractice = (topic) => {
    const nid = weakTopicDialog.noteId;
    setWeakTopicDialog({
      open: false,
      topic: null,
      subject: null,
      noteId: null,
      data: null,
      loading: false,
    });
    if (nid) {
      navigate(`/quiz-mode?noteId=${nid}&n=10`);
    } else {
      // Fallback if no noteId (shouldn't happen with new backend)
      navigate("/quiz", { state: { topic } });
    }
  };

  const handlePracticeAll = () => {
    // Initialize with all weak topics selected by default
    if (!data?.weak_topics?.length) return;
    const allTopics = data.weak_topics.map((w) => w.topic);
    setPracticeAllDialog({
      open: true,
      questionCount: 10,
      selectedTopics: allTopics,
    });
  };

  const handleStartPracticeAll = () => {
    // Find note IDs for selected topics
    const selectedNoteIds = data.weak_topics
      .filter(
        (w) => practiceAllDialog.selectedTopics.includes(w.topic) && w.note_id,
      )
      .map((w) => w.note_id);

    const uniqueNoteIds = [...new Set(selectedNoteIds)];

    setPracticeAllDialog({
      open: false,
      questionCount: 10,
      selectedTopics: [],
    });

    if (uniqueNoteIds.length > 0) {
      const idsParam = uniqueNoteIds.join(",");
      const n = practiceAllDialog.questionCount || 10;
      navigate(`/quiz-mode?noteIds=${idsParam}&n=${n}`, {
        state: {
          weakTopics: practiceAllDialog.selectedTopics, // Pass for tracking/validation if needed
          isPracticeAll: true,
        },
      });
    } else {
      // Fallback to old behavior if no note IDs found (e.g. general topics)
      navigate("/quiz", {
        state: {
          weakTopics: practiceAllDialog.selectedTopics,
          questionCount: practiceAllDialog.questionCount,
          isPracticeAll: true,
        },
      });
    }
  };

  const handleToggleTopic = (topic) => {
    setPracticeAllDialog((prev) => {
      const isSelected = prev.selectedTopics.includes(topic);
      const newSelected = isSelected
        ? prev.selectedTopics.filter((t) => t !== topic)
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
    recent_activity: [],
  };

  const unreadCount = notifications.filter((n) => !n.is_read).length;



  const hour = new Date().getHours();
  const greeting =
    hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        p: { xs: 2, md: 3 },
        borderRadius: "28px",
        background:
          "linear-gradient(145deg, rgba(255,255,255,0.86) 0%, rgba(248,251,255,0.78) 100%)",
        border: "1px solid rgba(148,163,184,0.28)",
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "absolute",
          width: 260,
          height: 260,
          borderRadius: "50%",
          top: -130,
          right: -70,
          background:
            "radial-gradient(circle, rgba(14,165,233,0.22) 0%, transparent 70%)",
          pointerEvents: "none",
        },
        "&::after": {
          content: '""',
          position: "absolute",
          width: 280,
          height: 280,
          borderRadius: "50%",
          bottom: -140,
          left: -80,
          background:
            "radial-gradient(circle, rgba(124,58,237,0.18) 0%, transparent 70%)",
          pointerEvents: "none",
        },
      }}
      className="animate-fade-in-up"
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 2,
        }}
      >
        <Box>
          <Typography
            variant="body2"
            sx={{
              color: "text.disabled",
              fontWeight: 600,
              mb: 0.5,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
              fontSize: "0.7rem",
            }}
          >
            {new Date().toLocaleDateString("en-US", {
              weekday: "long",
              month: "long",
              day: "numeric",
            })}
          </Typography>
          <Typography
            variant="h2"
            sx={{
              fontWeight: 900,
              color: "text.primary",
              letterSpacing: "-0.025em",
              mb: 1,
              lineHeight: 1.15,
            }}
          >
            {greeting}, {user?.first_name || "Student"}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 0.5,
                bgcolor: "rgba(11,218,91,0.1)",
                border: "1px solid rgba(11,218,91,0.25)",
                borderRadius: "20px",
                px: 1.5,
                py: 0.5,
              }}
            >
              <WhatshotIcon sx={{ color: "#0bda5b", fontSize: 16 }} />
              <Typography
                variant="caption"
                sx={{ color: "#0bda5b", fontWeight: 800, lineHeight: 1 }}
              >
                {data.streak}-day streak
              </Typography>
            </Box>
            <Typography
              variant="body2"
              sx={{ color: "text.secondary", fontWeight: 500 }}
            >
              Keep it up!
            </Typography>
          </Box>
        </Box>
        <Box sx={{ display: "flex", gap: 1.5, mt: { xs: 2, md: 0 } }}>
          <Button
            variant="outlined"
            startIcon={<NotificationsIcon />}
            onClick={(e) => setNotificationAnchor(e.currentTarget)}
            sx={{
              borderColor: "divider",
              color: "text.primary",
              bgcolor: "background.paper",
              fontWeight: 700,
              borderRadius: "12px",
              display: { xs: "none", sm: "flex" },
              "&:hover": {
                borderColor: "primary.light",
                bgcolor: "rgba(37,99,235,0.04)",
              },
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
              borderRadius: "12px",
              fontWeight: 700,
              px: 3,
              boxShadow: "0 4px 14px rgba(37,99,235,0.25)",
            }}
          >
            Daily Goals
          </Button>
        </Box>
      </Box>

      {/* Notifications Menu */}
      <Menu
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={() => setNotificationAnchor(null)}
        PaperProps={{
          sx: { width: 360, maxHeight: 400 },
        }}
      >
        <Box
          sx={{
            p: 2,
            borderBottom: "1px solid",
            borderColor: "divider",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Notifications
          </Typography>
          {unreadCount > 0 && (
            <Button
              size="small"
              onClick={markAllNotificationsRead}
              sx={{ textTransform: "none", fontSize: "0.75rem" }}
            >
              Mark all as read
            </Button>
          )}
        </Box>
        {notifications.length === 0 ? (
          <MenuItem disabled>
            <Typography variant="body2" color="text.secondary">
              No notifications
            </Typography>
          </MenuItem>
        ) : (
          notifications.map((notif) => (
            <MenuItem
              key={notif.id}
              onClick={() => markNotificationRead(notif.id)}
              sx={{
                bgcolor: notif.is_read ? "transparent" : "action.hover",
                whiteSpace: "normal",
                py: 1.5,
              }}
            >
              <Box>
                <Typography
                  variant="body2"
                  sx={{ fontWeight: notif.is_read ? 400 : 600 }}
                >
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
      <Dialog
        open={todoOpen}
        onClose={() => setTodoOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Daily Goals & To-Do List</DialogTitle>
        <DialogContent>
          <Box sx={{ display: "flex", gap: 1, mb: 2 }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Add a new task..."
              value={newTodo}
              onChange={(e) => setNewTodo(e.target.value)}
              onKeyPress={(e) => e.key === "Enter" && addTodo()}
            />
            <Button
              variant="contained"
              onClick={addTodo}
              startIcon={<AddIcon />}
            >
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
                  bgcolor: "background.paper",
                  mb: 1,
                  borderRadius: 1,
                  border: "1px solid",
                  borderColor: "divider",
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
                    textDecoration: todo.completed ? "line-through" : "none",
                    color: todo.completed ? "text.secondary" : "text.primary",
                  }}
                />
              </ListItem>
            ))}
            {todos.length === 0 && (
              <Typography
                variant="body2"
                color="text.secondary"
                align="center"
                sx={{ py: 4 }}
              >
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
        <LoadingSkeletonPack rows={4} cardHeight={180} />
      ) : statsError ? (
        <SurfaceCard sx={{ borderColor: "rgba(244,63,94,0.22)" }}>
          <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
            Dashboard temporarily unavailable
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {statsError}
          </Typography>
          <Button variant="contained" color="error" onClick={fetchStats}>
            Retry
          </Button>
        </SurfaceCard>
      ) : (
        <>
          {/* Stats Grid */}
          <Grid container spacing={3}>
            <Grid item xs={6} md={3}>
              <StatCard
                title="Current Streak"
                value={`${data.streak || 0} Days`}
                subtext="keep it up!"
                colorKey="streak"
                icon={<WhatshotIcon sx={{ fontSize: 24 }} />}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <StatCard
                title="Questions Answered"
                value={data.questions_answered}
                subtext="total"
                colorKey="questions"
                icon={<QuizIcon sx={{ fontSize: 24 }} />}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <StatCard
                title="Topics Mastered"
                value={data.topics_mastered}
                subtext=">80% mastery"
                colorKey="mastered"
                icon={<SchoolIcon sx={{ fontSize: 24 }} />}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <StatCard
                title="Avg. Quiz Score"
                value={`${data.avg_score}%`}
                subtext="lifetime"
                colorKey="score"
                icon={<AnalyticsIcon sx={{ fontSize: 24 }} />}
              />
            </Grid>
          </Grid>

          {/* Quick Actions */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Box
                sx={{
                  width: 4,
                  height: 20,
                  borderRadius: 2,
                  background: "linear-gradient(180deg, #2563EB, #7C3AED)",
                }}
              />
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  color: "text.primary",
                  letterSpacing: "-0.01em",
                }}
              >
                Quick Actions
              </Typography>
            </Box>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <QuickActionCard
                  title="Start Adaptive Quiz"
                  subtitle="Test your knowledge on weak areas and get instant feedback."
                  icon={<PsychologyIcon />}
                  colorIdx={0}
                  onClick={() => navigate("/quiz")}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <QuickActionCard
                  title="Upload Lecture Notes"
                  subtitle="AI parses your PDFs to generate summaries and flashcards."
                  icon={<UploadFileIcon />}
                  colorIdx={1}
                  onClick={() => navigate("/lectures")}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <QuickActionCard
                  title="Generate Study Guide"
                  subtitle="Create a personalized study plan for your upcoming exams."
                  icon={<AutoStoriesIcon />}
                  colorIdx={2}
                  onClick={() => navigate("/analysis")}
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <QuickActionCard
                  title="Exam Preparation"
                  subtitle="Generate strategies and practice with past papers."
                  icon={<AssignmentIcon />}
                  colorIdx={3}
                  onClick={() => navigate("/exam-preparation")}
                />
              </Grid>
            </Grid>
          </Box>

          {/* ── Analytics Charts ─────────────────────────────────── */}
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2.5 }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Box
                sx={{
                  width: 4,
                  height: 20,
                  borderRadius: 2,
                  background: "linear-gradient(180deg,#6366F1,#EC4899)",
                }}
              />
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 800,
                  color: "text.primary",
                  letterSpacing: "-0.01em",
                }}
              >
                Analytics & Performance
              </Typography>
            </Box>

            {/* Row 1: Area + Bar */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Card
                  sx={{
                    borderRadius: "20px",
                    border: "2px solid #EDE9FE",
                    p: 0,
                  }}
                >
                  <CardContent sx={{ p: "24px !important" }}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        mb: 2,
                      }}
                    >
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: "10px",
                          background: "linear-gradient(135deg,#6366F1,#8B5CF6)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <AnalyticsIcon sx={{ color: "#fff", fontSize: 18 }} />
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 800 }}>
                        Weekly Study Activity
                      </Typography>
                    </Box>
                    <ResponsiveContainer width="100%" height={220}>
                      <AreaChart
                        data={[
                          { day: "Mon", questions: 12, time: 45 },
                          { day: "Tue", questions: 19, time: 60 },
                          { day: "Wed", questions: 8, time: 30 },
                          { day: "Thu", questions: 25, time: 90 },
                          { day: "Fri", questions: 16, time: 55 },
                          { day: "Sat", questions: 30, time: 120 },
                          { day: "Sun", questions: 22, time: 75 },
                        ]}
                        margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient
                            id="colorQ"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="5%"
                              stopColor="#6366F1"
                              stopOpacity={0.4}
                            />
                            <stop
                              offset="95%"
                              stopColor="#6366F1"
                              stopOpacity={0}
                            />
                          </linearGradient>
                          <linearGradient
                            id="colorT"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop
                              offset="5%"
                              stopColor="#EC4899"
                              stopOpacity={0.4}
                            />
                            <stop
                              offset="95%"
                              stopColor="#EC4899"
                              stopOpacity={0}
                            />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#EDE9FE" />
                        <XAxis
                          dataKey="day"
                          tick={{ fontSize: 12, fill: "#6B7280" }}
                        />
                        <YAxis tick={{ fontSize: 12, fill: "#6B7280" }} />
                        <Tooltip
                          contentStyle={{
                            borderRadius: 12,
                            border: "1px solid #EDE9FE",
                            boxShadow: "0 8px 24px rgba(99,102,241,0.15)",
                          }}
                        />
                        <Legend />
                        <Area
                          type="monotone"
                          dataKey="questions"
                          stroke="#6366F1"
                          strokeWidth={2.5}
                          fill="url(#colorQ)"
                          name="Questions"
                        />
                        <Area
                          type="monotone"
                          dataKey="time"
                          stroke="#EC4899"
                          strokeWidth={2.5}
                          fill="url(#colorT)"
                          name="Minutes"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card
                  sx={{
                    borderRadius: "20px",
                    border: "2px solid #FDE9F7",
                    height: "100%",
                    p: 0,
                  }}
                >
                  <CardContent
                    sx={{
                      p: "24px !important",
                      height: "100%",
                      display: "flex",
                      flexDirection: "column",
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        mb: 2,
                      }}
                    >
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: "10px",
                          background: "linear-gradient(135deg,#EC4899,#F43F5E)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <SchoolIcon sx={{ color: "#fff", fontSize: 18 }} />
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 800 }}>
                        Score Distribution
                      </Typography>
                    </Box>
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie
                          data={[
                            {
                              name: "Excellent (>90%)",
                              value:
                                data.mastery_data.filter(
                                  (m) => m.percentage >= 90,
                                ).length || 2,
                            },
                            {
                              name: "Good (70-90%)",
                              value:
                                data.mastery_data.filter(
                                  (m) =>
                                    m.percentage >= 70 && m.percentage < 90,
                                ).length || 3,
                            },
                            {
                              name: "Average (50-70%)",
                              value:
                                data.mastery_data.filter(
                                  (m) =>
                                    m.percentage >= 50 && m.percentage < 70,
                                ).length || 2,
                            },
                            {
                              name: "Weak (<50%)",
                              value: data.weak_topics.length || 1,
                            },
                          ]}
                          cx="50%"
                          cy="50%"
                          innerRadius={55}
                          outerRadius={85}
                          paddingAngle={4}
                          dataKey="value"
                        >
                          {["#10B981", "#6366F1", "#F59E0B", "#F43F5E"].map(
                            (c, i) => (
                              <Cell key={i} fill={c} />
                            ),
                          )}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            borderRadius: 12,
                            border: "1px solid #EDE9FE",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Row 2: Bar + Radar + Mastery */}
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Card
                  sx={{
                    borderRadius: "20px",
                    border: "2px solid #D1FAE5",
                    p: 0,
                  }}
                >
                  <CardContent sx={{ p: "24px !important" }}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        mb: 2,
                      }}
                    >
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: "10px",
                          background: "linear-gradient(135deg,#10B981,#06B6D4)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <QuizIcon sx={{ color: "#fff", fontSize: 18 }} />
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 800 }}>
                        Quiz Scores
                      </Typography>
                    </Box>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart
                        data={[
                          { week: "W1", score: 65 },
                          { week: "W2", score: 72 },
                          { week: "W3", score: 68 },
                          { week: "W4", score: 80 },
                          { week: "W5", score: parseInt(data.avg_score) || 78 },
                        ]}
                        margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
                      >
                        <defs>
                          <linearGradient
                            id="barGrad"
                            x1="0"
                            y1="0"
                            x2="0"
                            y2="1"
                          >
                            <stop offset="5%" stopColor="#10B981" />
                            <stop offset="95%" stopColor="#06B6D4" />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#D1FAE5" />
                        <XAxis
                          dataKey="week"
                          tick={{ fontSize: 12, fill: "#6B7280" }}
                        />
                        <YAxis
                          domain={[0, 100]}
                          tick={{ fontSize: 12, fill: "#6B7280" }}
                        />
                        <Tooltip
                          contentStyle={{
                            borderRadius: 12,
                            border: "1px solid #D1FAE5",
                          }}
                        />
                        <Bar
                          dataKey="score"
                          fill="url(#barGrad)"
                          radius={[6, 6, 0, 0]}
                          name="Score %"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card
                  sx={{
                    borderRadius: "20px",
                    border: "2px solid #FEF3C7",
                    p: 0,
                  }}
                >
                  <CardContent sx={{ p: "24px !important" }}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 1.5,
                        mb: 2,
                      }}
                    >
                      <Box
                        sx={{
                          width: 36,
                          height: 36,
                          borderRadius: "10px",
                          background: "linear-gradient(135deg,#F59E0B,#F97316)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        <PsychologyIcon sx={{ color: "#fff", fontSize: 18 }} />
                      </Box>
                      <Typography variant="h6" sx={{ fontWeight: 800 }}>
                        Skill Radar
                      </Typography>
                    </Box>
                    <ResponsiveContainer width="100%" height={200}>
                      <RadarChart
                        data={[
                          { skill: "Recall", score: 75 },
                          { skill: "Comprehension", score: 68 },
                          { skill: "Application", score: 80 },
                          { skill: "Analysis", score: 60 },
                          { skill: "Synthesis", score: 72 },
                          { skill: "Evaluation", score: 65 },
                        ]}
                      >
                        <PolarGrid stroke="#FDE68A" />
                        <PolarAngleAxis
                          dataKey="skill"
                          tick={{ fontSize: 11, fill: "#6B7280" }}
                        />
                        <Radar
                          name="Skills"
                          dataKey="score"
                          stroke="#F59E0B"
                          fill="#F59E0B"
                          fillOpacity={0.3}
                          strokeWidth={2}
                        />
                        <Tooltip
                          contentStyle={{
                            borderRadius: 12,
                            border: "1px solid #FEF3C7",
                          }}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card
                  sx={{
                    borderRadius: "20px",
                    border: "2px solid #EDE9FE",
                    p: 0,
                    height: "100%",
                  }}
                >
                  <CardContent
                    sx={{
                      p: "24px !important",
                      display: "flex",
                      flexDirection: "column",
                      height: "100%",
                      gap: 3,
                    }}
                  >
                    <Box
                      sx={{ display: "flex", alignItems: "center", gap: 1.5 }}
                    >
                      <Avatar
                        sx={{
                          background: "linear-gradient(135deg,#8B5CF6,#6366F1)",
                          width: 36,
                          height: 36,
                          borderRadius: "10px",
                        }}
                      >
                        <SchoolIcon sx={{ fontSize: 18 }} />
                      </Avatar>
                      <Box>
                        <Typography
                          variant="h6"
                          sx={{ fontWeight: 800, lineHeight: 1.2 }}
                        >
                          Subject Mastery
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{ color: "text.secondary" }}
                        >
                          {data.mastery_data.length > 0
                            ? Math.round(
                                data.mastery_data.reduce(
                                  (a, c) => a + c.percentage,
                                  0,
                                ) / data.mastery_data.length,
                              )
                            : 0}
                          % avg
                        </Typography>
                      </Box>
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 2.5,
                        flex: 1,
                      }}
                    >
                      {data.mastery_data.length > 0 ? (
                        data.mastery_data.map((m, idx) => (
                          <MasteryBar
                            key={idx}
                            subject={m.subject}
                            percentage={m.percentage}
                            color={SUBJECT_COLORS[idx % SUBJECT_COLORS.length]}
                          />
                        ))
                      ) : (
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          align="center"
                          sx={{ my: "auto" }}
                        >
                          No mastery data yet. Take some quizzes!
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {/* Performance Overview: Weak Topics + Activity */}
            <Grid container spacing={3}>
              {/* Weak Topics */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: "100%", minHeight: 400 }}>
                  <CardContent
                    sx={{
                      p: "24px !important",
                      display: "flex",
                      flexDirection: "column",
                      height: "100%",
                      gap: 3,
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: 2,
                      }}
                    >
                      <Box
                        sx={{ display: "flex", alignItems: "center", gap: 1.5 }}
                      >
                        <Avatar
                          sx={{
                            bgcolor: "rgba(248, 113, 113, 0.1)",
                            color: "#F87171",
                            width: 40,
                            height: 40,
                            borderRadius: "10px",
                          }}
                        >
                          <WarningIcon />
                        </Avatar>
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: 800,
                            color: "text.primary",
                            lineHeight: 1.2,
                          }}
                        >
                          Weak Topics
                        </Typography>
                      </Box>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={handlePracticeAll}
                        sx={{
                          borderRadius: "8px",
                          textTransform: "none",
                          fontWeight: 700,
                          borderColor: "primary.main",
                          color: "primary.main",
                          px: 2,
                          py: 0.5,
                        }}
                      >
                        Practice All
                      </Button>
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        flexWrap: "wrap",
                        gap: 1,
                        flex: 1,
                      }}
                    >
                      {data.weak_topics.length > 0 ? (
                        data.weak_topics.map((w, idx) => (
                          <DashboardTopicChip
                            key={idx}
                            topic={w.topic}
                            subject={w.subject}
                            accuracy={w.accuracy}
                            noteId={w.note_id}
                            onPlay={(t, s, nId) => {
                              navigate(`/concept-coach?topic=${encodeURIComponent(t)}&subject=${encodeURIComponent(s || '')}&autoExplain=true`);
                            }}
                          />
                        ))
                      ) : (
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          align="center"
                          sx={{ my: "auto", width: '100%' }}
                        >
                          Great job! No weak topics detected.
                        </Typography>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              {/* Recent Activity */}
              <Grid item xs={12} md={4}>
                <Card sx={{ height: "100%", minHeight: 400 }}>
                  <CardContent
                    sx={{
                      p: "24px !important",
                      display: "flex",
                      flexDirection: "column",
                      height: "100%",
                      gap: 3,
                    }}
                  >
                    <Box
                      sx={{ display: "flex", alignItems: "center", gap: 1.5 }}
                    >
                      <Avatar
                        sx={{
                          bgcolor: "rgba(16, 185, 129, 0.1)",
                          color: "#10B981",
                          width: 40,
                          height: 40,
                          borderRadius: "10px",
                        }}
                      >
                        <ScheduleIcon />
                      </Avatar>
                      <Typography
                        variant="h6"
                        sx={{
                          fontWeight: 800,
                          color: "text.primary",
                          lineHeight: 1.2,
                        }}
                      >
                        Recent Activity
                      </Typography>
                    </Box>
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 3,
                        flex: 1,
                        mt: 1,
                      }}
                    >
                      {data.recent_activity.length > 0 ? (
                        data.recent_activity.map((act, idx) => (
                          <Box
                            key={idx}
                            sx={{
                              display: "flex",
                              gap: 2,
                              alignItems: "flex-start",
                            }}
                          >
                            <Avatar
                              sx={{
                                width: 36,
                                height: 36,
                                bgcolor:
                                  act.type === "upload"
                                    ? "rgba(19, 127, 236, 0.1)"
                                    : "rgba(16, 185, 129, 0.1)",
                                color:
                                  act.type === "upload"
                                    ? "primary.main"
                                    : "#10B981",
                              }}
                            >
                              {act.type === "upload" ? (
                                <UploadIcon sx={{ fontSize: 18 }} />
                              ) : (
                                <CheckIcon sx={{ fontSize: 18 }} />
                              )}
                            </Avatar>
                            <Box>
                              <Typography
                                variant="body2"
                                sx={{ fontWeight: 600, color: "text.primary" }}
                              >
                                {act.text}
                              </Typography>
                              <Typography
                                variant="caption"
                                sx={{
                                  color: "text.secondary",
                                  fontWeight: 500,
                                }}
                              >
                                {act.subtext}
                              </Typography>
                            </Box>
                          </Box>
                        ))
                      ) : (
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          align="center"
                          sx={{ my: "auto" }}
                        >
                          No recent activity.
                        </Typography>
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
            onClose={() =>
              setPracticeAllDialog({ open: false, questionCount: 10 })
            }
            maxWidth="sm"
            fullWidth
          >
            <DialogTitle>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <QuizIcon sx={{ color: "primary.main" }} />
                <Typography variant="h6" sx={{ fontWeight: 700 }}>
                  Practice Weak Topics
                </Typography>
              </Box>
            </DialogTitle>
            <DialogContent>
              <Box
                sx={{ display: "flex", flexDirection: "column", gap: 3, pt: 1 }}
              >
                {/* Weak Topics Multi-Select */}
                <Box>
                  <Typography
                    variant="subtitle2"
                    sx={{ fontWeight: 600, mb: 1.5, color: "text.primary" }}
                  >
                    Select topics to practice:
                  </Typography>
                  {data.weak_topics && data.weak_topics.length > 0 ? (
                    <Box
                      sx={{ display: "flex", flexDirection: "column", gap: 1 }}
                    >
                      {data.weak_topics.map((topic, idx) => {
                        const isSelected =
                          practiceAllDialog.selectedTopics.includes(
                            topic.topic,
                          );
                        return (
                          <Box
                            key={idx}
                            onClick={() => handleToggleTopic(topic.topic)}
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 1,
                              p: 1.5,
                              borderRadius: "8px",
                              bgcolor: isSelected
                                ? (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "rgba(19, 127, 236, 0.2)"
                                      : "rgba(239, 246, 255, 1)"
                                : (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "rgba(35, 54, 72, 0.3)"
                                      : "rgba(248, 250, 252, 1)",
                              border: "2px solid",
                              borderColor: isSelected
                                ? "primary.main"
                                : "divider",
                              cursor: "pointer",
                              transition: "all 0.2s",
                              "&:hover": {
                                borderColor: "primary.main",
                                bgcolor: (theme) =>
                                  theme.palette.mode === "dark"
                                    ? "rgba(19, 127, 236, 0.15)"
                                    : "rgba(239, 246, 255, 0.8)",
                              },
                            }}
                          >
                            <Checkbox
                              checked={isSelected}
                              onChange={() => handleToggleTopic(topic.topic)}
                              sx={{ p: 0 }}
                            />
                            <Box sx={{ flex: 1 }}>
                              <Typography
                                variant="body2"
                                sx={{ fontWeight: 600, color: "text.primary" }}
                              >
                                {topic.topic}
                              </Typography>
                              <Typography
                                variant="caption"
                                sx={{ color: "text.secondary" }}
                              >
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
                  <Typography
                    variant="subtitle2"
                    sx={{ fontWeight: 600, mb: 1.5, color: "text.primary" }}
                  >
                    Number of Questions:
                  </Typography>
                  <Box sx={{ display: "flex", gap: 1.5 }}>
                    {[5, 10, 15, 20].map((count) => (
                      <Button
                        key={count}
                        variant={
                          practiceAllDialog.questionCount === count
                            ? "contained"
                            : "outlined"
                        }
                        onClick={() =>
                          setPracticeAllDialog((prev) => ({
                            ...prev,
                            questionCount: count,
                          }))
                        }
                        sx={{
                          flex: 1,
                          fontWeight: 700,
                          textTransform: "none",
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
                onClick={() =>
                  setPracticeAllDialog({ open: false, questionCount: 10 })
                }
                sx={{ textTransform: "none" }}
              >
                Cancel
              </Button>
              <Button
                variant="contained"
                onClick={handleStartPracticeAll}
                startIcon={<QuizIcon />}
                disabled={
                  !practiceAllDialog.selectedTopics ||
                  practiceAllDialog.selectedTopics.length === 0
                }
                sx={{ textTransform: "none", fontWeight: 700 }}
              >
                Start Practice
              </Button>
            </DialogActions>
          </Dialog>

          {/* Weak Topic Explanation Dialog */}
          <Dialog
            open={weakTopicDialog.open}
            onClose={() =>
              setWeakTopicDialog({
                open: false,
                topic: null,
                subject: null,
                noteId: null,
                data: null,
                loading: false,
              })
            }
            maxWidth="md"
            fullWidth
          >
            <DialogTitle sx={{ pb: 1 }}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <PlayCircleIcon sx={{ color: "primary.main" }} />
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
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 2,
                    py: 6,
                  }}
                >
                  <CircularProgress />
                  <Typography variant="body2" color="text.secondary">
                    Generating personalized explanation...
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    This may take a few moments
                  </Typography>
                </Box>
              ) : weakTopicDialog.data?.error ? (
                <Box
                  sx={{
                    py: 4,
                    textAlign: "center",
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                  }}
                >
                  <Typography color="error" sx={{ mb: 1 }}>
                    {weakTopicDialog.data.error}
                  </Typography>
                  {weakTopicDialog.data.canRetry && (
                    <Button
                      variant="outlined"
                      onClick={handleRetryExplanation}
                      sx={{ mx: "auto" }}
                    >
                      Try Again
                    </Button>
                  )}
                </Box>
              ) : weakTopicDialog.data ? (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
                  {/* Explanation */}
                  {weakTopicDialog.data.explanation && (
                    <Box>
                      <Typography
                        variant="h6"
                        sx={{ fontWeight: 700, mb: 1, color: "text.primary" }}
                      >
                        Understanding the Topic
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ color: "text.secondary", whiteSpace: "pre-line" }}
                      >
                        {weakTopicDialog.data.explanation}
                      </Typography>
                    </Box>
                  )}

                  {/* Key Concepts */}
                  {weakTopicDialog.data.key_concepts &&
                    weakTopicDialog.data.key_concepts.length > 0 && (
                      <Box>
                        <Typography
                          variant="h6"
                          sx={{ fontWeight: 700, mb: 1, color: "text.primary" }}
                        >
                          Key Concepts
                        </Typography>
                        <Box
                          sx={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 1,
                          }}
                        >
                          {weakTopicDialog.data.key_concepts.map(
                            (concept, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 1 }}>
                                <CheckCircleIcon
                                  sx={{
                                    fontSize: 20,
                                    color: "primary.main",
                                    mt: 0.2,
                                  }}
                                />
                                <Typography
                                  variant="body2"
                                  sx={{ color: "text.secondary" }}
                                >
                                  {concept}
                                </Typography>
                              </Box>
                            ),
                          )}
                        </Box>
                      </Box>
                    )}

                  {/* Common Mistakes */}
                  {weakTopicDialog.data.common_mistakes &&
                    weakTopicDialog.data.common_mistakes.length > 0 && (
                      <Box>
                        <Typography
                          variant="h6"
                          sx={{ fontWeight: 700, mb: 1, color: "text.primary" }}
                        >
                          Common Mistakes to Avoid
                        </Typography>
                        <Box
                          sx={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 1,
                          }}
                        >
                          {weakTopicDialog.data.common_mistakes.map(
                            (mistake, idx) => (
                              <Box key={idx} sx={{ display: "flex", gap: 1 }}>
                                <WarningIcon
                                  sx={{
                                    fontSize: 20,
                                    color: "#F87171",
                                    mt: 0.2,
                                  }}
                                />
                                <Typography
                                  variant="body2"
                                  sx={{ color: "text.secondary" }}
                                >
                                  {mistake}
                                </Typography>
                              </Box>
                            ),
                          )}
                        </Box>
                      </Box>
                    )}

                  {/* Practice Tips */}
                  {weakTopicDialog.data.practice_tips &&
                    weakTopicDialog.data.practice_tips.length > 0 && (
                      <Box>
                        <Typography
                          variant="h6"
                          sx={{ fontWeight: 700, mb: 1, color: "text.primary" }}
                        >
                          Practice Tips
                        </Typography>
                        <Box
                          sx={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 1,
                          }}
                        >
                          {weakTopicDialog.data.practice_tips.map(
                            (tip, idx) => (
                              <Box
                                key={idx}
                                sx={{
                                  p: 2,
                                  borderRadius: "8px",
                                  bgcolor: (theme) =>
                                    theme.palette.mode === "dark"
                                      ? "rgba(35, 54, 72, 0.5)"
                                      : "rgba(239, 246, 255, 1)",
                                  border: "1px solid",
                                  borderColor: "divider",
                                }}
                              >
                                <Typography
                                  variant="body2"
                                  sx={{
                                    color: "text.primary",
                                    fontWeight: 600,
                                  }}
                                >
                                  {idx + 1}. {tip}
                                </Typography>
                              </Box>
                            ),
                          )}
                        </Box>
                      </Box>
                    )}

                  {/* Resources */}
                  {weakTopicDialog.data.resources &&
                    weakTopicDialog.data.resources.length > 0 && (
                      <Box>
                        <Typography
                          variant="h6"
                          sx={{ fontWeight: 700, mb: 1, color: "text.primary" }}
                        >
                          Learning Resources
                        </Typography>
                        <Box
                          sx={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 1.5,
                          }}
                        >
                          {weakTopicDialog.data.resources.map(
                            (resource, idx) => (
                              <Box
                                key={idx}
                                sx={{
                                  p: 2,
                                  borderRadius: "8px",
                                  bgcolor: "background.paper",
                                  border: "1px solid",
                                  borderColor: "divider",
                                }}
                              >
                                <Box
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 1,
                                    mb: 0.5,
                                  }}
                                >
                                  {resource.type === "article" && (
                                    <AutoStoriesIcon
                                      sx={{
                                        fontSize: 18,
                                        color: "primary.main",
                                      }}
                                    />
                                  )}
                                  {resource.type === "video" && (
                                    <PlayCircleIcon
                                      sx={{
                                        fontSize: 18,
                                        color: "primary.main",
                                      }}
                                    />
                                  )}
                                  {resource.type === "practice" && (
                                    <QuizIcon
                                      sx={{
                                        fontSize: 18,
                                        color: "primary.main",
                                      }}
                                    />
                                  )}
                                  <Typography
                                    variant="body2"
                                    sx={{
                                      fontWeight: 600,
                                      color: "text.primary",
                                    }}
                                  >
                                    {resource.title}
                                  </Typography>
                                </Box>
                                <Typography
                                  variant="caption"
                                  sx={{ color: "text.secondary" }}
                                >
                                  {resource.description}
                                </Typography>
                              </Box>
                            ),
                          )}
                        </Box>
                      </Box>
                    )}

                  {/* Study Approach */}
                  {weakTopicDialog.data.study_approach && (
                    <Box
                      sx={{
                        p: 3,
                        borderRadius: "12px",
                        bgcolor: (theme) =>
                          theme.palette.mode === "dark"
                            ? "rgba(19, 127, 236, 0.1)"
                            : "rgba(239, 246, 255, 1)",
                        border: "2px solid",
                        borderColor: "primary.main",
                      }}
                    >
                      <Typography
                        variant="h6"
                        sx={{ fontWeight: 700, mb: 1, color: "primary.main" }}
                      >
                        Recommended Study Approach
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ color: "text.secondary", whiteSpace: "pre-line" }}
                      >
                        {weakTopicDialog.data.study_approach}
                      </Typography>
                    </Box>
                  )}
                </Box>
              ) : null}
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 3 }}>
              <Button
                onClick={() =>
                  setWeakTopicDialog({
                    open: false,
                    topic: null,
                    subject: null,
                    noteId: null,
                    data: null,
                    loading: false,
                  })
                }
                sx={{ textTransform: "none" }}
              >
                Close
              </Button>
              <Button
                variant="contained"
                onClick={() => handleStartPractice(weakTopicDialog.topic)}
                startIcon={<QuizIcon />}
                sx={{ textTransform: "none", fontWeight: 700 }}
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
