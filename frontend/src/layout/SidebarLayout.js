// SidebarLayout.js
import React from "react";
import { Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import {
  Avatar,
  IconButton,
  AppBar,
  Toolbar,
  Tooltip,
  Menu,
  MenuItem,
  Typography,
  Drawer,
  Box,
  Divider,
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
import TranslateIcon from "@mui/icons-material/Translate";
import LogoutIcon from "@mui/icons-material/Logout";
import KeyboardArrowRightIcon from "@mui/icons-material/KeyboardArrowRight";
import AddIcon from "@mui/icons-material/Add";
import RemoveIcon from "@mui/icons-material/Remove";
import { useAuth } from "../context/AuthContext";
import { useColorMode } from "../context/ThemeContext";
import Notifications from "../components/Notifications";
import { useTranslation } from "react-i18next";
import i18n from "../i18n";

const SIDEBAR_WIDTH = 260;

/* ── helpers ────────────────────────────────────────────────── */
const getAvatarColor = (username) => {
  if (!username) return "#2563EB";
  const colors = ["#2563EB", "#7C3AED", "#0891B2", "#059669", "#D97706"];
  const hash = username.split("").reduce((a, c) => a + c.charCodeAt(0), 0);
  return colors[hash % colors.length];
};

const getInitials = (user) => {
  if (!user) return "U";
  if (user.first_name && user.last_name)
    return `${user.first_name[0]}${user.last_name[0]}`.toUpperCase();
  return user.username ? user.username.substring(0, 2).toUpperCase() : "U";
};

/* ── Language switcher ──────────────────────────────────────── */
const LANGUAGES = [
  { code: "en", label: "English", short: "EN" },
  { code: "hi", label: "हिंदी", short: "हि" },
  { code: "ta", label: "தமிழ்", short: "த" },
  { code: "fr", label: "Français", short: "FR" },
];

const LanguageSwitcher = () => {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [currentLang, setCurrentLang] = React.useState(i18n.language || "en");
  const open = Boolean(anchorEl);
  const current = LANGUAGES.find((l) => l.code === currentLang) || LANGUAGES[0];

  const handleChange = (code) => {
    i18n.changeLanguage(code);
    localStorage.setItem("learnflow_lang", code);
    setCurrentLang(code);
    setAnchorEl(null);
  };

  return (
    <>
      <Tooltip title="Change language">
        <Box
          onClick={(e) => setAnchorEl(e.currentTarget)}
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 0.5,
            px: 1.5,
            py: 0.75,
            borderRadius: "8px",
            border: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
            cursor: "pointer",
            transition: "all 0.2s",
            "&:hover": {
              borderColor: "primary.main",
              bgcolor: "rgba(37,99,235,0.05)",
            },
          }}
        >
          <TranslateIcon sx={{ fontSize: 14, color: "text.secondary" }} />
          <Typography
            variant="caption"
            sx={{ fontWeight: 700, color: "text.secondary", lineHeight: 1 }}
          >
            {current.short}
          </Typography>
        </Box>
      </Tooltip>
      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={() => setAnchorEl(null)}
        PaperProps={{
          sx: {
            borderRadius: 2,
            mt: 1,
            minWidth: 140,
            border: "1px solid #e2e8f0",
            boxShadow: "0 8px 32px rgba(15,23,42,0.12)",
          },
        }}
        transformOrigin={{ horizontal: "right", vertical: "top" }}
        anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
      >
        {LANGUAGES.map((lang) => (
          <MenuItem
            key={lang.code}
            onClick={() => handleChange(lang.code)}
            selected={lang.code === currentLang}
            sx={{ borderRadius: 1.5, mx: 0.5, mb: 0.3 }}
          >
            <Typography
              variant="body2"
              fontWeight={lang.code === currentLang ? 700 : 500}
            >
              {lang.label}
            </Typography>
          </MenuItem>
        ))}
      </Menu>
    </>
  );
};

/* ── Nav items ──────────────────────────────────────────────── */
const NAV_COLORS = {
  dashboard: {
    color: "#2563EB",
    light: "rgba(37,99,235,0.12)",
    grad: "linear-gradient(135deg,#2563EB,#3B82F6)",
  },
  lectures: {
    color: "#7C3AED",
    light: "rgba(124,58,237,0.12)",
    grad: "linear-gradient(135deg,#7C3AED,#A78BFA)",
  },
  questions: {
    color: "#F43F5E",
    light: "rgba(244,63,94,0.12)",
    grad: "linear-gradient(135deg,#F43F5E,#FB7185)",
  },
  "weak-topics": {
    color: "#F59E0B",
    light: "rgba(245,158,11,0.14)",
    grad: "linear-gradient(135deg,#F59E0B,#FCD34D)",
  },
  "question-bank": {
    color: "#10B981",
    light: "rgba(16,185,129,0.12)",
    grad: "linear-gradient(135deg,#10B981,#34D399)",
  },
  "study-plan": {
    color: "#10B981",
    light: "rgba(16,185,129,0.12)",
    grad: "linear-gradient(135deg,#10B981,#34D399)",
  },
  "exam-preparation": {
    color: "#F59E0B",
    light: "rgba(245,158,11,0.12)",
    grad: "linear-gradient(135deg,#F59E0B,#FCD34D)",
  },
  flashcards: {
    color: "#06B6D4",
    light: "rgba(6,182,212,0.12)",
    grad: "linear-gradient(135deg,#06B6D4,#67E8F9)",
  },
  summarize: {
    color: "#0EA5E9",
    light: "rgba(14,165,233,0.12)",
    grad: "linear-gradient(135deg,#0EA5E9,#38BDF8)",
  },
  "concept-coach": {
    color: "#7C3AED",
    light: "rgba(124,58,237,0.15)",
    grad: "linear-gradient(135deg,#4F46E5,#7C3AED,#EC4899)",
  },
  profile: {
    color: "#6B7280",
    light: "rgba(107,114,128,0.10)",
    grad: "linear-gradient(135deg,#6B7280,#9CA3AF)",
  },
  "quiz-result": {
    color: "#06B6D4",
    light: "rgba(6,182,212,0.14)",
    grad: "linear-gradient(135deg,#06B6D4,#38BDF8)",
  },
};

const useNavItems = (t) =>
  React.useMemo(
    () => [
      {
        key: "dashboard",
        label: t("nav_dashboard"),
        icon: <BarChartIcon />,
        to: "/dashboard",
      },
      {
        key: "lectures",
        label: t("nav_lectures"),
        icon: <AutoStoriesIcon />,
        to: "/lectures",
      },
      {
        key: "summarize",
        label: t("nav_summarize"),
        icon: <SummarizeIcon />,
        to: "/summarize",
      },
      {
        key: "questions",
        label: t("nav_quiz"),
        icon: <ArticleIcon />,
        to: "/quiz",
      },
      {
        key: "weak-topics",
        label: t("nav_weak_topics", "Weak Topics"),
        icon: <SchoolIcon />,
        to: "/weak-topics",
      },
      {
        key: "question-bank",
        label: "Question Bank",
        icon: <ArticleIcon />,
        to: "/question-bank",
      },
      {
        key: "flashcards",
        label: t("nav_flashcards"),
        icon: <StyleIcon />,
        to: "/flashcards",
      },
      {
        key: "study-plan",
        label: t("nav_study_plan"),
        icon: <CalendarTodayIcon />,
        to: "/study-plan",
      },
      {
        key: "exam-preparation",
        label: t("nav_exam_prep"),
        icon: <SchoolIcon />,
        to: "/exam-preparation",
        badge: "HOT",
      },
      {
        key: "concept-coach",
        label: t("nav_concept_coach"),
        icon: <SmartToyIcon />,
        to: "/concept-coach",
        flagship: true,
        badge: "AI",
      },
      {
        key: "profile",
        label: t("nav_profile"),
        icon: <PersonIcon />,
        to: "/profile",
      },
    ],
    [t],
  );

/* ── Page title map ─────────────────────────────────────────── */
const PAGE_TITLES = {
  "/dashboard": "Dashboard",
  "/lectures": "Lecture Notes",
  "/summarize": "Summarize",
  "/quiz": "Generate Quiz",
  "/question-bank": "Question Bank",
  "/weak-topics": "Weak Topics",
  "/study-plan": "Study Plan",
  "/exam-preparation": "Exam Preparation",
  "/flashcards": "Flashcards",
  "/concept-coach": "Concept Coach",
  "/profile": "My Profile",
  "/quiz-result": "Quiz Result",
};

const PAGE_SUBTITLES = {
  "/dashboard": "Analyze your priorities and performance",
  "/lectures": "Capture and manage your learning sources",
  "/summarize": "Extract key points and formulas quickly",
  "/quiz": "Practice with adaptive questions",
  "/question-bank": "Search and practice all questions",
  "/weak-topics": "Turn weak areas into next actions",
  "/study-plan": "Plan and track your execution blocks",
  "/exam-preparation": "Prepare with step-by-step exam workflow",
  "/flashcards": "Reinforce memory with active recall",
  "/concept-coach": "Learn interactively with guided tutoring",
  "/profile": "Manage identity and study preferences",
  "/quiz-result": "Translate outcomes into immediate next steps",
};

/* ── Sidebar nav item ───────────────────────────────────────── */
const NavItem = ({ item, isActive, onClick }) => {
  const nc = NAV_COLORS[item.key] || NAV_COLORS["dashboard"];
  return (
    <Link to={item.to} onClick={onClick} style={{ textDecoration: "none" }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          px: 1.5,
          py: 1.1,
          borderRadius: "12px",
          position: "relative",
          overflow: "hidden",
          transition: "all 0.2s cubic-bezier(0.22,1,0.36,1)",
          cursor: "pointer",
          mx: 0.5,
          bgcolor: isActive ? `${nc.light}` : "transparent",
          color: isActive ? nc.color : "text.secondary",
          fontWeight: isActive ? 700 : 500,
          "&::before": isActive
            ? {
                content: '""',
                position: "absolute",
                left: 0,
                top: "15%",
                bottom: "15%",
                width: 3,
                borderRadius: "0 4px 4px 0",
                background: nc.grad,
                boxShadow: `0 0 8px ${nc.color}60`,
              }
            : {},
          "&:hover": {
            bgcolor: isActive ? nc.light : "rgba(15,23,42,0.04)",
            color: isActive ? nc.color : "text.primary",
            transform: "translateX(2px)",
            "& .nav-icon-box": {
              background: isActive ? nc.grad : `${nc.color}18`,
              transform: "scale(1.08)",
              boxShadow: `0 4px 12px ${nc.color}30`,
            },
          },
        }}
      >
        <Box
          className="nav-icon-box"
          sx={{
            width: 32,
            height: 32,
            borderRadius: "9px",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: isActive
              ? nc.grad
              : item.flagship
                ? "rgba(124,58,237,0.1)"
                : "rgba(15,23,42,0.04)",
            color: isActive ? "#fff" : item.flagship ? "#7C3AED" : nc.color,
            transition: "all 0.2s cubic-bezier(0.22,1,0.36,1)",
            boxShadow: isActive ? `0 4px 12px ${nc.color}40` : "none",
            "& svg": { fontSize: "1.05rem" },
          }}
        >
          {item.icon}
        </Box>

        <Typography
          variant="body2"
          sx={{
            flex: 1,
            fontWeight: "inherit",
            color: "inherit",
            lineHeight: 1,
            fontSize: "0.83rem",
            letterSpacing: isActive ? "-0.01em" : 0,
          }}
        >
          {item.label}
        </Typography>

        {item.badge && !isActive && (
          <Box
            sx={{
              fontSize: "0.58rem",
              fontWeight: 900,
              lineHeight: 1,
              px: 0.7,
              py: 0.35,
              borderRadius: "5px",
              background:
                item.badge === "HOT"
                  ? "linear-gradient(135deg,#F59E0B,#F97316)"
                  : "linear-gradient(135deg,#4F46E5,#7C3AED,#EC4899)",
              color: "#FFFFFF",
              letterSpacing: "0.06em",
              animation:
                item.badge === "HOT"
                  ? "pulseGlow 2.5s infinite"
                  : "pulseGlowViolet 2.5s infinite",
            }}
          >
            {item.badge}
          </Box>
        )}

        {isActive && (
          <KeyboardArrowRightIcon
            sx={{ fontSize: 15, color: nc.color, opacity: 0.8 }}
          />
        )}
      </Box>
    </Link>
  );
};

/* ── Sidebar content (shared) ───────────────────────────────── */
function SidebarContent({
  navItems,
  location,
  handleLogout,
  user,
  t,
  onNavClick,
  colorMode,
}) {
  const initials = getInitials(user);
  const avatarColor = getAvatarColor(user?.username);

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        py: 2.5,
        px: 1.25,
      }}
    >
      {/* Logo with aurora blob */}
      <Link to="/dashboard" style={{ textDecoration: "none" }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1.5,
            px: 1.25,
            mb: 2.5,
            position: "relative",
            overflow: "hidden",
            borderRadius: "14px",
            py: 1.5,
            background:
              "linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(124,58,237,0.06) 100%)",
            border: "1px solid rgba(37,99,235,0.12)",
            "&::before": {
              content: '""',
              position: "absolute",
              width: 80,
              height: 80,
              borderRadius: "50%",
              top: -30,
              right: -20,
              background:
                "radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%)",
              pointerEvents: "none",
            },
          }}
        >
          <Box
            sx={{
              width: 36,
              height: 36,
              borderRadius: "10px",
              flexShrink: 0,
              background: "linear-gradient(135deg, #2563EB 0%, #7C3AED 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 4px 14px rgba(37,99,235,0.40)",
              animation: "pulseGlow 3s infinite",
            }}
          >
            <Typography
              sx={{
                color: "#fff",
                fontWeight: 900,
                fontSize: "0.85rem",
                lineHeight: 1,
              }}
            >
              LF
            </Typography>
          </Box>
          <Box>
            <Typography
              sx={{
                fontWeight: 900,
                fontSize: "1.05rem",
                lineHeight: 1,
                background:
                  "linear-gradient(135deg, #2563EB 0%, #7C3AED 60%, #EC4899 100%)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                backgroundClip: "text",
                letterSpacing: "-0.03em",
              }}
            >
              LearnFlow
            </Typography>
            <Typography
              sx={{
                fontSize: "0.62rem",
                color: "text.disabled",
                fontWeight: 600,
                letterSpacing: "0.06em",
              }}
            >
              AI STUDY ASSISTANT
            </Typography>
          </Box>
        </Box>
      </Link>

      <Divider sx={{ mb: 1.5, opacity: 0.5 }} />

      {/* Nav */}
      <Box
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          gap: 0.2,
          overflowY: "auto",
          overflowX: "hidden",
          pb: 1,
        }}
      >
        {navItems.map((item) => {
          const isActive =
            location.pathname === item.to ||
            location.pathname.startsWith(item.to + "/");

          return (
            <React.Fragment key={item.key}>
              {item.flagship && (
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1,
                    px: 2,
                    pt: 2,
                    pb: 0.75,
                  }}
                >
                  <Box
                    sx={{
                      flex: 1,
                      height: "1px",
                      background:
                        "linear-gradient(90deg, rgba(124,58,237,0.3), transparent)",
                    }}
                  />
                  <Typography
                    variant="overline"
                    sx={{
                      color: "#7C3AED",
                      fontSize: "0.6rem",
                      fontWeight: 800,
                      letterSpacing: "0.12em",
                      background: "linear-gradient(90deg, #7C3AED, #EC4899)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                    }}
                  >
                    {t("nav_main_feature")}
                  </Typography>
                  <Box
                    sx={{
                      flex: 1,
                      height: "1px",
                      background:
                        "linear-gradient(90deg, transparent, rgba(236,72,153,0.3))",
                    }}
                  />
                </Box>
              )}
              <NavItem item={item} isActive={isActive} onClick={onNavClick} />
            </React.Fragment>
          );
        })}
      </Box>

      <Divider sx={{ my: 1.5, opacity: 0.5 }} />

      {/* User card */}
      <Box
        sx={{
          px: 1.5,
          py: 1.25,
          borderRadius: "14px",
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          background: `linear-gradient(135deg, ${avatarColor}12 0%, ${avatarColor}06 100%)`,
          border: "1px solid",
          borderColor: `${avatarColor}28`,
          transition: "all 0.2s",
          position: "relative",
          overflow: "hidden",
          "&:hover": { boxShadow: `0 4px 16px ${avatarColor}20` },
        }}
      >
        <Avatar
          src={user?.avatar_url}
          sx={{
            width: 34,
            height: 34,
            fontSize: "0.78rem",
            fontWeight: 900,
            background: `linear-gradient(135deg, ${avatarColor} 0%, #1e40af 100%)`,
            boxShadow: `0 3px 10px ${avatarColor}50`,
          }}
        >
          {initials}
        </Avatar>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 800,
              color: "text.primary",
              lineHeight: 1.2,
              fontSize: "0.8rem",
            }}
            noWrap
          >
            {user?.first_name
              ? `${user.first_name} ${user.last_name || ""}`.trim()
              : user?.username || "Student"}
          </Typography>
          <Typography
            variant="caption"
            sx={{ color: avatarColor, fontSize: "0.67rem", fontWeight: 600 }}
          >
            Student
          </Typography>
        </Box>
        <Tooltip title={t("nav_logout")}>
          <IconButton
            onClick={handleLogout}
            size="small"
            sx={{
              color: "text.disabled",
              "&:hover": {
                color: "#EF4444",
                bgcolor: "rgba(239,68,68,0.10)",
                transform: "rotate(180deg)",
              },
              borderRadius: "8px",
              transition: "all 0.3s ease",
            }}
          >
            <LogoutIcon sx={{ fontSize: 16 }} />
          </IconButton>
        </Tooltip>
      </Box>

      <Box
        sx={{
          mt: 1.5,
          px: 1,
          py: 1,
          borderRadius: "10px",
          border: "1px solid",
          borderColor: "divider",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 1,
          bgcolor: "rgba(255,255,255,0.65)",
        }}
      >
        <Typography
          sx={{ fontSize: "0.64rem", fontWeight: 700, color: "text.secondary" }}
        >
          Font size
        </Typography>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          <IconButton
            size="small"
            onClick={colorMode.decreaseFont}
            sx={{
              width: 24,
              height: 24,
              border: "1px solid",
              borderColor: "divider",
            }}
            aria-label="Decrease font size"
          >
            <RemoveIcon sx={{ fontSize: 14 }} />
          </IconButton>
          <IconButton
            size="small"
            onClick={colorMode.increaseFont}
            sx={{
              width: 24,
              height: 24,
              border: "1px solid",
              borderColor: "divider",
            }}
            aria-label="Increase font size"
          >
            <AddIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Box>
      </Box>

      <Typography
        sx={{
          textAlign: "center",
          fontSize: "0.6rem",
          color: "text.disabled",
          mt: 1.2,
          letterSpacing: "0.06em",
        }}
      >
        v2.5 · © 2026 LearnFlow
      </Typography>
    </Box>
  );
}

/* ── Main layout ─────────────────────────────────────────────── */
export default function SidebarLayout() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const colorMode = useColorMode();
  const { t } = useTranslation();
  const [mobileOpen, setMobileOpen] = React.useState(false);
  const [userMenuAnchor, setUserMenuAnchor] = React.useState(null);
  const navItems = useNavItems(t);

  const initials = getInitials(user);
  const avatarColor = getAvatarColor(user?.username);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const currentTitle =
    Object.entries(PAGE_TITLES).find(
      ([path]) =>
        location.pathname === path || location.pathname.startsWith(path + "/"),
    )?.[1] || "LearnFlow";

  const currentSubtitle =
    Object.entries(PAGE_SUBTITLES).find(
      ([path]) =>
        location.pathname === path || location.pathname.startsWith(path + "/"),
    )?.[1] || "Learning-first workspace";

  return (
    <Box
      sx={{
        display: "flex",
        minHeight: "100vh",
        bgcolor: "background.default",
        position: "relative",
        overflow: "hidden",
        "&::before": {
          content: '""',
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          zIndex: 0,
          background:
            "radial-gradient(circle at 10% 16%, rgba(14,165,233,0.18) 0%, transparent 34%), radial-gradient(circle at 84% 14%, rgba(124,58,237,0.16) 0%, transparent 32%), radial-gradient(circle at 72% 82%, rgba(236,72,153,0.15) 0%, transparent 35%), radial-gradient(circle at 18% 82%, rgba(16,185,129,0.12) 0%, transparent 30%)",
        },
      }}
    >
      {/* ── Persistent Sidebar (desktop) ── */}
      <Box
        component="aside"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          display: { xs: "none", md: "flex" },
          flexDirection: "column",
          position: "fixed",
          left: 0,
          top: 0,
          height: "100vh",
          bgcolor: "rgba(255,255,255,0.78)",
          backdropFilter: "blur(18px)",
          WebkitBackdropFilter: "blur(18px)",
          borderRight: "1px solid",
          borderColor: "rgba(148,163,184,0.30)",
          zIndex: 1200,
          overflowY: "auto",
          overflowX: "hidden",
          boxShadow:
            "8px 0 34px rgba(15,23,42,0.10), 1px 0 0 rgba(191,219,254,0.65)",
        }}
      >
        <SidebarContent
          navItems={navItems}
          location={location}
          handleLogout={handleLogout}
          user={user}
          t={t}
          onNavClick={undefined}
          colorMode={colorMode}
        />
      </Box>

      {/* ── Mobile Drawer ── */}
      <Drawer
        anchor="left"
        open={mobileOpen}
        onClose={() => setMobileOpen(false)}
        sx={{ display: { md: "none" } }}
        PaperProps={{
          sx: {
            width: SIDEBAR_WIDTH,
            border: "none",
            boxShadow: "4px 0 32px rgba(15,23,42,0.12)",
          },
        }}
        ModalProps={{ keepMounted: true }}
      >
        <SidebarContent
          navItems={navItems}
          location={location}
          handleLogout={handleLogout}
          user={user}
          t={t}
          onNavClick={() => setMobileOpen(false)}
          colorMode={colorMode}
        />
      </Drawer>

      {/* ── Main area ── */}
      <Box
        sx={{
          flex: 1,
          ml: { md: `${SIDEBAR_WIDTH}px` },
          display: "flex",
          flexDirection: "column",
          minHeight: "100vh",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* Top Bar */}
        <AppBar
          position="sticky"
          elevation={0}
          sx={{
            bgcolor: "rgba(255,255,255,0.86)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            borderBottom: "1px solid transparent",
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.95), rgba(255,255,255,0.95)), linear-gradient(90deg, #2563EB, #7C3AED, #EC4899)",
            backgroundOrigin: "border-box",
            backgroundClip: "padding-box, border-box",
            color: "#0F172A",
            zIndex: 1100,
            "&::after": {
              content: '""',
              position: "absolute",
              bottom: 0,
              left: 0,
              right: 0,
              height: "2px",
              background:
                "linear-gradient(90deg, #2563EB 0%, #7C3AED 50%, #EC4899 100%)",
              opacity: 0.35,
            },
          }}
        >
          <Toolbar
            sx={{
              justifyContent: "space-between",
              minHeight: "64px !important",
              px: { xs: 2, md: 3 },
            }}
          >
            {/* Left */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
              {/* Hamburger - mobile only */}
              <IconButton
                edge="start"
                onClick={() => setMobileOpen(true)}
                sx={{
                  display: { md: "none" },
                  borderRadius: "10px",
                  width: 38,
                  height: 38,
                  color: "text.secondary",
                  "&:hover": { bgcolor: "rgba(15,23,42,0.05)" },
                }}
              >
                <MenuIcon />
              </IconButton>

              {/* Mobile logo */}
              <Link
                to="/dashboard"
                style={{
                  textDecoration: "none",
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <Box
                  sx={{
                    width: 28,
                    height: 28,
                    borderRadius: "8px",
                    background:
                      "linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%)",
                    display: { xs: "flex", md: "none" },
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: "0 2px 8px rgba(37,99,235,0.25)",
                  }}
                >
                  <Typography
                    sx={{ color: "#fff", fontWeight: 900, fontSize: "0.7rem" }}
                  >
                    LF
                  </Typography>
                </Box>
              </Link>

              {/* Page title - desktop only */}
              <Box
                sx={{
                  display: { xs: "flex", md: "flex" },
                  alignItems: { xs: "flex-start", md: "center" },
                  flexDirection: { xs: "column", md: "row" },
                  gap: 1,
                }}
              >
                <Box
                  sx={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "linear-gradient(135deg, #2563EB, #7C3AED)",
                    boxShadow: "0 0 8px rgba(37,99,235,0.5)",
                    display: { xs: "none", md: "block" },
                  }}
                />
                <Box>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: 800,
                      color: "text.primary",
                      fontSize: "0.98rem",
                      letterSpacing: "-0.02em",
                    }}
                  >
                    {currentTitle}
                  </Typography>
                  <Typography
                    sx={{
                      display: { xs: "none", lg: "block" },
                      fontSize: "0.72rem",
                      color: "text.secondary",
                      lineHeight: 1.2,
                    }}
                  >
                    {currentSubtitle}
                  </Typography>
                </Box>
              </Box>
            </Box>

            {/* Right */}
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              <Notifications />
              <LanguageSwitcher />

              {/* User pill */}
              <Box
                onClick={(e) => setUserMenuAnchor(e.currentTarget)}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 1,
                  border: "1px solid",
                  borderColor: "divider",
                  borderRadius: "30px",
                  pl: 0.5,
                  pr: 1.5,
                  py: 0.5,
                  bgcolor: "background.paper",
                  cursor: "pointer",
                  transition: "all 0.2s",
                  "&:hover": {
                    borderColor: "primary.main",
                    boxShadow: "0 2px 8px rgba(37,99,235,0.12)",
                  },
                }}
              >
                <Avatar
                  src={user?.avatar_url}
                  sx={{
                    width: 30,
                    height: 30,
                    fontSize: "0.72rem",
                    fontWeight: 800,
                    background: `linear-gradient(135deg, ${avatarColor} 0%, #1e40af 100%)`,
                  }}
                >
                  {initials}
                </Avatar>
                <Box
                  sx={{
                    display: { xs: "none", sm: "flex" },
                    flexDirection: "column",
                    lineHeight: 1,
                  }}
                >
                  <Typography
                    sx={{
                      fontSize: "0.78rem",
                      fontWeight: 700,
                      color: "text.primary",
                      lineHeight: 1.3,
                    }}
                  >
                    {user?.first_name || user?.username || "Student"}
                  </Typography>
                  <Typography
                    sx={{
                      fontSize: "0.65rem",
                      color: "text.disabled",
                      fontWeight: 500,
                    }}
                  >
                    Student
                  </Typography>
                </Box>
              </Box>
            </Box>
          </Toolbar>
        </AppBar>

        {/* User dropdown */}
        <Menu
          anchorEl={userMenuAnchor}
          open={Boolean(userMenuAnchor)}
          onClose={() => setUserMenuAnchor(null)}
          PaperProps={{
            sx: {
              borderRadius: 2,
              mt: 1,
              minWidth: 180,
              border: "1px solid #e2e8f0",
              boxShadow: "0 8px 32px rgba(15,23,42,0.12)",
            },
          }}
          transformOrigin={{ horizontal: "right", vertical: "top" }}
          anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
        >
          <Box sx={{ px: 2, py: 1.5, borderBottom: "1px solid #E2E8F0" }}>
            <Typography variant="body2" fontWeight={700}>
              {user?.first_name || user?.username}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {user?.email}
            </Typography>
          </Box>
          <MenuItem
            onClick={() => {
              navigate("/profile");
              setUserMenuAnchor(null);
            }}
            sx={{ borderRadius: 1.5, mx: 0.5, mt: 0.5 }}
          >
            <PersonIcon
              sx={{ fontSize: 18, mr: 1.5, color: "text.secondary" }}
            />
            <Typography variant="body2" fontWeight={600}>
              My Profile
            </Typography>
          </MenuItem>
          <Divider sx={{ my: 0.5 }} />
          <MenuItem
            onClick={() => {
              handleLogout();
              setUserMenuAnchor(null);
            }}
            sx={{
              borderRadius: 1.5,
              mx: 0.5,
              mb: 0.5,
              color: "#EF4444",
              "&:hover": { bgcolor: "rgba(239,68,68,0.06)" },
            }}
          >
            <LogoutIcon sx={{ fontSize: 18, mr: 1.5 }} />
            <Typography variant="body2" fontWeight={600}>
              {t("nav_logout")}
            </Typography>
          </MenuItem>
        </Menu>

        {/* Page content */}
        <Box
          component="main"
          sx={{
            flex: 1,
            width: "100%",
            maxWidth: 1400,
            mx: "auto",
            px: { xs: 2, sm: 3, md: 4 },
            py: { xs: 3, md: 4 },
            position: "relative",
          }}
        >
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
