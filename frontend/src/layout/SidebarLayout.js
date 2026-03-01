// SidebarLayout.jsx
import React from "react";
import { Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  ListItemIcon,
  Typography,
  Avatar,
  IconButton,
  AppBar,
  Toolbar,
  Container,
  useTheme,
  Tooltip,
  Chip,
  Menu,
  MenuItem,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import ArticleIcon from "@mui/icons-material/Article";
import BarChartIcon from "@mui/icons-material/BarChart";
import AutoStoriesIcon from "@mui/icons-material/AutoStories";
import StyleIcon from "@mui/icons-material/Style";
import SummarizeIcon from "@mui/icons-material/Summarize";
import SchoolIcon from "@mui/icons-material/School";
import PersonIcon from "@mui/icons-material/Person";
import CalendarTodayIcon from "@mui/icons-material/CalendarToday";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import GradingIcon from "@mui/icons-material/Grading";
import TranslateIcon from "@mui/icons-material/Translate";
import { useAuth } from "../context/AuthContext";
import LogoutIcon from "@mui/icons-material/Logout";
import { useTranslation } from "react-i18next";
import i18n from "../i18n";

// Generate avatar color
const getAvatarColor = (username) => {
  if (!username) return '#2563EB';
  const colors = ['#2563EB', '#1D4ED8', '#F59E0B', '#10B981', '#7C3AED'];
  const hash = username.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return colors[hash % colors.length];
};

const getInitials = (user) => {
  if (!user) return 'U';
  if (user.first_name && user.last_name) {
    return `${user.first_name.charAt(0)}${user.last_name.charAt(0)}`.toUpperCase();
  }
  return user.username ? user.username.substring(0, 2).toUpperCase() : 'U';
};

const LANGUAGES = [
  { code: 'en', label: 'English', short: 'EN' },
  { code: 'hi', label: 'हिंदी', short: 'हि' },
  { code: 'ta', label: 'தமிழ்', short: 'த' },
  { code: 'fr', label: 'Français', short: 'FR' },
];

const LanguageSwitcher = () => {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [currentLang, setCurrentLang] = React.useState(i18n.language || 'en');
  const open = Boolean(anchorEl);
  const current = LANGUAGES.find(l => l.code === currentLang) || LANGUAGES[0];

  const handleChange = (code) => {
    i18n.changeLanguage(code);
    localStorage.setItem('learnflow_lang', code);
    setCurrentLang(code);
    setAnchorEl(null);
  };

  return (
    <>
      <Tooltip title="Change language">
        <Box
          onClick={e => setAnchorEl(e.currentTarget)}
          sx={{
            display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer',
            px: 1.5, py: 0.5, borderRadius: 2,
            border: '1px solid', borderColor: 'divider',
            '&:hover': { borderColor: 'primary.main', bgcolor: 'rgba(37,99,235,0.05)' },
            transition: 'all 0.2s'
          }}
        >
          <TranslateIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
          <Typography variant="caption" fontWeight={800} color="text.secondary" sx={{ fontSize: '0.78rem' }}>
            {current.short}
          </Typography>
        </Box>
      </Tooltip>
      <Menu anchorEl={anchorEl} open={open} onClose={() => setAnchorEl(null)}
        PaperProps={{ sx: { borderRadius: 2, mt: 1, minWidth: 140, border: '1px solid', borderColor: 'divider' } }}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        {LANGUAGES.map(lang => (
          <MenuItem key={lang.code} onClick={() => handleChange(lang.code)}
            selected={lang.code === currentLang}
            sx={{ borderRadius: 1, mx: 0.5, mb: 0.3, fontWeight: lang.code === currentLang ? 800 : 500 }}
          >
            <Typography variant="body2" fontWeight={lang.code === currentLang ? 800 : 500}>
              {lang.label}
            </Typography>
          </MenuItem>
        ))}
      </Menu>
    </>
  );
};


export default function SidebarLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const theme = useTheme();
  const { user, logout } = useAuth();
  const { t } = useTranslation();
  const [open, setOpen] = React.useState(false);

  const navItems = React.useMemo(() => [
    { key: "dashboard",        label: t('nav_dashboard'),        icon: <BarChartIcon />,      to: "/dashboard" },
    { key: "lectures",         label: t('nav_lectures'),         icon: <AutoStoriesIcon />,   to: "/lectures" },
    { key: "questions",        label: t('nav_quiz'),             icon: <ArticleIcon />,       to: "/quiz" },
    { key: "study-plan",       label: t('nav_study_plan'),       icon: <CalendarTodayIcon />, to: "/study-plan" },
    { key: "exam-preparation", label: t('nav_exam_prep'),        icon: <SchoolIcon />,        to: "/exam-preparation" },
    { key: "flashcards",       label: t('nav_flashcards'),       icon: <StyleIcon />,         to: "/flashcards" },
    { key: "summarize",        label: t('nav_summarize'),        icon: <SummarizeIcon />,     to: "/summarize" },
    { key: "concept-coach",    label: t('nav_concept_coach'),    icon: <SmartToyIcon />,      to: "/concept-coach", flagship: true },
    { key: "rubric-evaluator", label: t('nav_rubric_evaluator'), icon: <GradingIcon />,       to: "/rubric-evaluator" },
    { key: "profile",          label: t('nav_profile'),          icon: <PersonIcon />,        to: "/profile" },
  ], [t]);
  
  const userInitials = getInitials(user);
  const avatarColor = getAvatarColor(user?.username);

  const toggleDrawer = (newOpen) => () => {
    setOpen(newOpen);
  };

  const handleLogout = () => {
      logout();
      navigate('/login');
  };

  const drawerContent = (
    <Box sx={{ width: 300, height: '100%', display: 'flex', flexDirection: 'column', p: 3 }} role="presentation" onClick={toggleDrawer(false)}>
      {/* Drawer Header */}
      <Box 
        component={Link} 
        to="/dashboard" 
        sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 6, px: 2, mt: 2, textDecoration: 'none' }}
      >
        <Box 
          sx={{ 
            width: 40, 
            height: 40, 
            background: 'linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)', 
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'white',
            fontWeight: 800,
            fontSize: '1.2rem',
            boxShadow: '0 8px 16px rgba(37, 99, 235, 0.3)'
          }} 
        >
          LF
        </Box>
        <Typography variant="h5" sx={{ 
            fontWeight: 800, 
            background: "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: '-0.5px'
        }}>
          LearnFlow
        </Typography>
      </Box>

      <List sx={{ px: 1 }}>
        {navItems.map((item) => {
           const isActive = location.pathname === item.to;
           return (
            <React.Fragment key={item.key}>
              {item.flagship && (
                <Box sx={{ px: 2, pt: 2, pb: 0.5 }}>
                  <Typography variant="caption" sx={{ fontWeight: 800, color: 'primary.main', letterSpacing: '0.1em', fontSize: '0.65rem' }}>{t('nav_main_feature')}</Typography>
                </Box>
              )}
              <ListItem disablePadding sx={{ mb: 1.5 }}>
                <ListItemButton 
                  component={Link} 
                  to={item.to}
                  sx={{ 
                    borderRadius: '12px !important',
                    py: item.flagship ? 1.8 : 1.5,
                    px: 2.5,
                    background: isActive 
                      ? (theme.palette.mode === 'dark' ? 'rgba(37, 99, 235, 0.15)' : 'rgba(37, 99, 235, 0.1)')
                      : item.flagship ? (theme.palette.mode === 'dark' ? 'rgba(37,99,235,0.08)' : 'rgba(37,99,235,0.05)') : 'transparent',
                    color: isActive ? 'primary.main' : item.flagship ? 'primary.main' : 'text.secondary',
                    borderLeft: isActive ? '4px solid' : item.flagship ? '4px solid' : '4px solid transparent',
                    borderColor: isActive ? 'primary.main' : item.flagship ? 'rgba(37,99,235,0.4)' : 'transparent',
                    transition: 'all 0.2s ease',
                    '&:hover': { 
                        bgcolor: isActive ? (theme.palette.mode === 'dark' ? 'rgba(37, 99, 235, 0.25)' : 'rgba(37, 99, 235, 0.2)') : 'rgba(255, 255, 255, 0.03)',
                        transform: 'translateX(2px)'
                    }
                  }}
                >
                    <ListItemIcon sx={{ 
                        minWidth: 40, 
                        color: isActive ? 'primary.main' : item.flagship ? 'primary.main' : 'inherit',
                        '& svg': { fontSize: 24 }
                    }}>
                        {item.icon}
                    </ListItemIcon>
                   <ListItemText 
                      primary={item.label} 
                      primaryTypographyProps={{ 
                          fontWeight: isActive || item.flagship ? 700 : 500,
                          fontSize: '0.95rem'
                      }} 
                   />
                   {item.flagship && !isActive && (
                     <Chip label="NEW" size="small" sx={{ height: 18, fontSize: '0.6rem', fontWeight: 800, bgcolor: 'primary.main', color: 'white', borderRadius: 1 }} />
                   )}
                </ListItemButton>
              </ListItem>
            </React.Fragment>
           );
        })}

      </List>
      
      <Box sx={{ flexGrow: 1 }} />
      
      {/* Logout Button */}
      <Box sx={{ px: 2, mb: 2 }}>
          <ListItemButton 
            onClick={handleLogout}
            sx={{ 
              borderRadius: '12px !important',
              py: 1.5,
              px: 3,
              color: 'text.secondary',
              '&:hover': { bgcolor: 'rgba(239, 68, 68, 0.1)', color: 'error.main' }
            }}
          >
              <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                  <LogoutIcon />
              </ListItemIcon>
              <ListItemText primary={t('nav_logout')} primaryTypographyProps={{ fontWeight: 600 }} />
          </ListItemButton>
      </Box>

      <Typography variant="caption" sx={{ color: 'text.secondary', textAlign: 'center', pb: 2, opacity: 0.6 }}>
          v2.5 • © 2025 LearnFlow
      </Typography>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', bgcolor: 'background.default' }}>
      
      {/* Top Navigation Bar */}
      <AppBar 
        position="sticky" 
        elevation={0}
        sx={{ 
            bgcolor: theme.palette.mode === 'dark' ? 'rgba(11, 15, 25, 0.8)' : 'rgba(255, 255, 255, 0.8)',
            backdropFilter: 'blur(16px)',
            borderBottom: '1px solid',
            borderColor: 'divider',
            color: 'text.primary',
        }}
      >
        <Toolbar sx={{ justifyContent: 'space-between', height: 72 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
                <IconButton
                    edge="start"
                    color="inherit"
                    aria-label="menu"
                    onClick={toggleDrawer(true)}
                    sx={{ 
                        mr: 2,
                        width: 44,
                        height: 44,
                        borderRadius: '12px',
                        '&:hover': { bgcolor: 'rgba(37, 99, 235, 0.1)', color: 'primary.main' }
                    }}
                >
                    <MenuIcon />
                </IconButton>
                <Typography 
                    variant="h5" 
                    component={Link} 
                    to="/dashboard"
                    sx={{ 
                        fontWeight: 800, 
                        display: { xs: 'none', sm: 'block' }, 
                        textDecoration: 'none', 
                        background: "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
                        backgroundClip: "text",
                        WebkitBackgroundClip: "text",
                        WebkitTextFillColor: "transparent",
                        letterSpacing: '-0.5px'
                    }}
                >
                    LearnFlow
                </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                 <LanguageSwitcher />
                 {/* User Profile in Navbar */}
                 <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, cursor: 'pointer', pl: 1, pr: 2, py: 0.5, borderRadius: '50px', border: '1px solid', borderColor: 'divider', '&:hover': { bgcolor: 'action.hover' } }}>
                    <Avatar 
                        src={user?.avatar_url}
                        sx={{ 
                            width: 38, 
                            height: 38, 
                            background: `linear-gradient(135deg, ${avatarColor} 0%, #172554 100%)`, 
                            fontSize: '0.95rem', 
                            fontWeight: 700,
                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                        }}
                    >
                        {userInitials}
                    </Avatar>
                    <Box sx={{ display: { xs: 'none', sm: 'block' } }}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, lineHeight: 1.2 }}>
                            {user?.first_name || user?.username || 'Student'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
                            Student Account
                        </Typography>
                    </Box>
                 </Box>
            </Box>
        </Toolbar>
      </AppBar>

      {/* Drawer */}
      <Drawer
        anchor="left"
        open={open}
        onClose={toggleDrawer(false)}
        PaperProps={{
            sx: {
                background: theme.palette.background.paper, 
                // Using theme paper background which is now #151B2B in dark mode
                width: 300,
                borderRight: '1px solid',
                borderColor: 'divider',
            }
        }}
        ModalProps={{
            keepMounted: true, 
        }}
      >
        {drawerContent}
      </Drawer>

      {/* Main Content */}
      <Box component="main" sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', bgcolor: 'background.default' }}>
        <Container
          maxWidth="lg"
          sx={{
            flex: 1,
            py: { xs: 3, md: 4 },
            px: { xs: 2, sm: 3, md: 4 },
          }}
        >
          <Outlet />
        </Container>
      </Box>
    </Box>
  );
}

