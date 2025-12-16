// SidebarLayout.jsx
import React from "react";
import { Outlet, Link, useLocation } from "react-router-dom";
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  AppBar,
  Typography,
  IconButton,
  Divider,
  Avatar,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import DashboardIcon from "@mui/icons-material/Dashboard";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import QuizIcon from "@mui/icons-material/Quiz";
import ArticleIcon from "@mui/icons-material/Article";
import BarChartIcon from "@mui/icons-material/BarChart";
import ListAltIcon from "@mui/icons-material/ListAlt";
import AutoStoriesIcon from "@mui/icons-material/AutoStories";
import StyleIcon from "@mui/icons-material/Style";
import SummarizeIcon from "@mui/icons-material/Summarize";
import SchoolIcon from "@mui/icons-material/School";
import Notifications from "../components/Notifications";
import { useAuth } from "../context/AuthContext";

// Generate avatar color based on username
const getAvatarColor = (username) => {
  if (!username) return '#667eea';
  const colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#30cfd0', '#a8edea', '#ff9a56'];
  const hash = username.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[hash % colors.length];
};

// Generate initials
const getInitials = (user) => {
  if (user?.first_name && user?.last_name) {
    return `${user.first_name.charAt(0)}${user.last_name.charAt(0)}`.toUpperCase();
  }
  if (user?.username) {
    const parts = user.username.split(/[\s_-]/);
    if (parts.length > 1) {
      return `${parts[0].charAt(0)}${parts[1].charAt(0)}`.toUpperCase();
    }
    return user.username.substring(0, 2).toUpperCase();
  }
  return 'U';
};

const drawerWidth = 280;
const navItems = [
  {
    key: "upload",
    label: "Upload Notes",
    icon: <UploadFileIcon />,
    to: "/upload",
  },
  {
    key: "lectures",
    label: "My Lectures",
    icon: <AutoStoriesIcon />,
    to: "/lectures",
  },
  {
    key: "flashcards",
    label: "Flashcards",
    icon: <StyleIcon />,
    to: "/flashcards",
  },
  {
    key: "summarize",
    label: "Summarize Lectures",
    icon: <SummarizeIcon />,
    to: "/summarize",
  },
  {
    key: "exam-preparation",
    label: "Exam Preparation",
    icon: <SchoolIcon />,
    to: "/exam-preparation",
  },
  {
    key: "questions",
    label: "Generate Questions",
    icon: <ArticleIcon />,
    to: "/questions",
  },
  { key: "quiz", label: "Quiz", icon: <QuizIcon />, to: "/quiz-entry" },
  {
    key: "weak",
    label: "Weak Topics",
    icon: <ListAltIcon />,
    to: "/weak-topics",
  },
  {
    key: "study",
    label: "Study Plan",
    icon: <BarChartIcon />,
    to: "/study-plan",
  },
];

export default function SidebarLayout() {
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = React.useState(true);
  const location = useLocation();
  const { user } = useAuth();
  
  const userInitials = getInitials(user);
  const avatarColor = getAvatarColor(user?.username);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const handleSidebarToggle = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const drawer = (
    <Box
      sx={{
        width: drawerWidth,
        height: "100%",
        display: "flex",
        flexDirection: "column",
        background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
      }}
    >
      {/* Sidebar Header */}
      <Toolbar sx={{ px: 2.5, py: 2.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
          <Avatar
            sx={{
              width: 40,
              height: 40,
              background: "transparent",
              color: "#fff",
              fontWeight: "bold",
              border: "2px solid rgba(255,255,255,0.5)",
            }}
          >
            L
          </Avatar>
          <Box>
            <Typography
              variant="h6"
              sx={{ color: "#fff", fontWeight: 700, fontSize: "18px" }}
            >
              LearnFlow
            </Typography>
          </Box>
        </Box>
      </Toolbar>

      <Divider sx={{ background: "rgba(255,255,255,0.2)" }} />

      {/* Navigation List */}
      <List sx={{ flex: 1, px: 1.5, py: 2 }}>
        {navItems.map((item) => {
          const isActive = location.pathname === item.to;
          return (
            <ListItem key={item.key} disablePadding sx={{ mb: 1 }}>
              <ListItemButton
                component={Link}
                to={item.to}
                sx={{
                  borderRadius: "10px",
                  color: isActive ? "#fff" : "rgba(255,255,255,0.8)",
                  background: isActive
                    ? "rgba(255,255,255,0.25)"
                    : "transparent",
                  backdropFilter: isActive ? "blur(10px)" : "none",
                  border: isActive ? "1px solid rgba(255,255,255,0.4)" : "none",
                  transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                  "&:hover": {
                    background: "rgba(255,255,255,0.15)",
                    backdropFilter: "blur(10px)",
                  },
                  py: 1.5,
                }}
              >
                <ListItemIcon sx={{ color: "inherit", minWidth: 40 }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText
                  primary={item.label}
                  sx={{
                    "& .MuiTypography-root": {
                      fontSize: "14px",
                      fontWeight: isActive ? 600 : 500,
                      letterSpacing: "0.3px",
                    },
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>

      {/* Footer */}
      <Divider sx={{ background: "rgba(255,255,255,0.2)" }} />
      <Box sx={{ p: 2.5, textAlign: "center" }}>
        <Typography
          variant="caption"
          sx={{ color: "rgba(255,255,255,0.6)", display: "block" }}
        >
          © 2025 LearnFlow
        </Typography>
      </Box>
    </Box>
  );

  return (
    <Box
      sx={{
        display: "flex",
        minHeight: "100vh",
        background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
      }}
    >
      {/* Top AppBar */}
      <AppBar
        position="fixed"
        sx={{
          width: { sm: isSidebarOpen ? `calc(100% - ${drawerWidth}px)` : "100%" },
          ml: { sm: isSidebarOpen ? `${drawerWidth}px` : 0 },
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          color: "#fff",
          boxShadow: "0 4px 20px rgba(102, 126, 234, 0.15)",
          borderBottom: "none",
          zIndex: 1200,
          transition: theme => theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar sx={{ justifyContent: "space-between" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <IconButton
              color="inherit"
              edge="start"
              onClick={() => {
                // Check if mobile or desktop
                if (window.innerWidth < 600) {
                  handleDrawerToggle();
                } else {
                  handleSidebarToggle();
                }
              }}
              sx={{
                mr: 1,
                "&:hover": { background: "rgba(255,255,255,0.1)" },
              }}
            >
              <MenuIcon />
            </IconButton>
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 40,
                  height: 40,
                  borderRadius: "12px",
                  background: "rgba(255,255,255,0.15)",
                  backdropFilter: "blur(4px)",
                }}
              >
                <AutoStoriesIcon sx={{ fontSize: 24, color: "#fff" }} />
              </Box>
              <Box>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 700, fontSize: "20px", lineHeight: 1.1 }}
                >
                  LearnFlow
                </Typography>
                <Typography
                  variant="caption"
                  sx={{
                    color: "rgba(255,255,255,0.8)",
                    fontSize: "12px",
                    display: "block",
                    mt: 0.5,
                  }}
                >
                  Personalized Education Platform
                </Typography>
              </Box>
            </Box>
          </Box>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
            <Notifications />
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <IconButton
                component={Link}
                to="/profile"
                sx={{ color: "white", "&:hover": { background: "rgba(255,255,255,0.1)" }, p: 0.5 }}
                title={user?.username || 'Profile'}
              >
                <Avatar sx={{ 
                  width: 36, 
                  height: 36, 
                  bgcolor: avatarColor,
                  fontWeight: 700,
                  fontSize: '14px',
                  border: '2px solid rgba(255,255,255,0.3)'
                }}>
                  {userInitials}
                </Avatar>
              </IconButton>
              <IconButton
                onClick={() => {
                  localStorage.removeItem('access_token');
                  localStorage.removeItem('refresh_token');
                  window.location.href = '/login';
                }}
                sx={{ color: "white", "&:hover": { background: "rgba(255,255,255,0.1)" } }}
                title="Logout"
              >
                <Typography variant="body2" sx={{ fontSize: "14px" }}>Logout</Typography>
              </IconButton>
            </Box>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { sm: isSidebarOpen ? drawerWidth : 0 }, flexShrink: { sm: 0 }, transition: 'width 0.3s' }}
        aria-label="mailbox folders"
      >
        {/* Mobile Drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: "block", sm: "none" },
            "& .MuiDrawer-paper": {
              width: drawerWidth,
              boxSizing: "border-box",
            },
          }}
        >
          {drawer}
        </Drawer>

        {/* Desktop Drawer */}
        <Drawer
          variant="persistent"
          sx={{
            display: { xs: "none", sm: "block" },
            "& .MuiDrawer-paper": {
              width: drawerWidth,
              boxSizing: "border-box",
              position: "fixed",
              height: "100vh",
              zIndex: 1100,
              borderRight: "none",
            },
          }}
          open={isSidebarOpen}
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content Area */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { sm: `calc(100% - ${isSidebarOpen ? drawerWidth : 0}px)` },
          ml: { sm: 0 }, // Margin left is handled by flex layout since drawer is persistent but in a flex container
          minHeight: "100vh",
          transition: theme => theme.transitions.create(['width', 'margin'], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
          }),
        }}
      >
        <Toolbar /> {/* Push content below AppBar */}
        <Box
          sx={{
            p: 4,
            background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
            minHeight: "calc(100vh - 64px)",
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
