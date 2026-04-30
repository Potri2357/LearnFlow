import React from "react";
import { useNavigate } from "react-router-dom";
import {
  Box,
  Button,
  Chip,
  Container,
  Grid,
  Paper,
  Stack,
  Typography,
  alpha,
  useTheme,
} from "@mui/material";
import {
  ArrowForward as ArrowForwardIcon,
  AutoStories as AutoStoriesIcon,
  CheckCircleOutline as CheckCircleOutlineIcon,
  Login as LoginIcon,
  PlayCircleOutline as PlayCircleOutlineIcon,
  Psychology as PsychologyIcon,
  Quiz as QuizIcon,
  School as SchoolIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
} from "@mui/icons-material";
import { SurfaceCard } from "../components/ui";

const valueCards = [
  {
    icon: <QuizIcon />,
    title: "Adaptive quizzes",
    description:
      "Turn lecture notes into active recall sessions with visible progress.",
    iconBg: "rgba(37,99,235,0.10)",
    iconColor: "primary.main",
  },
  {
    icon: <AutoStoriesIcon />,
    title: "Lecture capture",
    description:
      "Upload PDFs, images, audio, or video and keep every source organized.",
    iconBg: "rgba(124,58,237,0.10)",
    iconColor: "secondary.main",
  },
  {
    icon: <PsychologyIcon />,
    title: "Concept coach",
    description:
      "Ask follow-up questions and get guided explanations when a topic stalls.",
    iconBg: "rgba(6,182,212,0.12)",
    iconColor: "info.main",
  },
];

const steps = [
  {
    number: "01",
    title: "Capture",
    description: "Add lecture materials and keep sources in one workspace.",
  },
  {
    number: "02",
    title: "Practice",
    description: "Generate quizzes and flashcards that target what you missed.",
  },
  {
    number: "03",
    title: "Plan",
    description: "Translate weak topics into study blocks and next actions.",
  },
];

const trustPoints = [
  {
    icon: <SecurityIcon />,
    title: "Private by default",
    text: "Keep your study data contained in a single authenticated workspace.",
    iconBg: "rgba(16,185,129,0.12)",
    iconColor: "success.main",
  },
  {
    icon: <SpeedIcon />,
    title: "Fast feedback",
    text: "Progressive rendering and lightweight cards make the app feel immediate.",
    iconBg: "rgba(37,99,235,0.10)",
    iconColor: "primary.main",
  },
  {
    icon: <TimelineIcon />,
    title: "Clear progression",
    text: "Every flow ends with an obvious next step, not a dead end.",
    iconBg: "rgba(124,58,237,0.10)",
    iconColor: "secondary.main",
  },
];

const featureGrid = [
  {
    title: "Dashboard",
    description:
      "Study time, streaks, mastery, and weak topics at a glance.",
  },
  {
    title: "Study plan",
    description:
      "Make generated plans skimmable, printable, and easy to execute.",
  },
  {
    title: "Exam prep",
    description:
      "Preserve workflow progress while you upload syllabi and papers.",
  },
  {
    title: "Quiz result",
    description:
      "Translate the outcome into the next practice session immediately.",
  },
];

function AccentIcon({ children, background, color }) {
  return (
    <Box
      sx={{
        width: 52,
        height: 52,
        borderRadius: 2,
        display: "grid",
        placeItems: "center",
        color,
        bgcolor: background,
      }}
    >
      {children}
    </Box>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const theme = useTheme();

  return (
    <Box
      sx={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        background:
          "radial-gradient(circle at top left, rgba(37,99,235,0.14) 0, transparent 28%), radial-gradient(circle at 85% 12%, rgba(124,58,237,0.13) 0, transparent 26%), linear-gradient(180deg, #F7FAFF 0%, #EEF4FF 100%)",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          opacity: 0.9,
          backgroundImage:
            "linear-gradient(rgba(37,99,235,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(37,99,235,0.03) 1px, transparent 1px)",
          backgroundSize: "42px 42px",
          maskImage:
            "linear-gradient(180deg, rgba(0,0,0,0.7), rgba(0,0,0,0.02))",
        }}
      />

      <Container
        maxWidth="lg"
        sx={{ position: "relative", zIndex: 1, py: { xs: 4, md: 6 } }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 2,
            mb: { xs: 5, md: 7 },
          }}
        >
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
            <Box
              sx={{
                width: 44,
                height: 44,
                borderRadius: 2,
                display: "grid",
                placeItems: "center",
                color: "#fff",
                fontWeight: 900,
                letterSpacing: "0.04em",
                background: "linear-gradient(135deg, #2563EB 0%, #7C3AED 100%)",
                boxShadow: "0 14px 30px rgba(37,99,235,0.22)",
              }}
            >
              LF
            </Box>
            <Box>
              <Typography sx={{ fontWeight: 900, letterSpacing: "-0.03em", lineHeight: 1 }}>
                LearnFlow
              </Typography>
              <Typography variant="caption" sx={{ color: "text.secondary", fontWeight: 700 }}>
                Premium light-first study workspace
              </Typography>
            </Box>
          </Box>

          <Stack direction="row" spacing={1} sx={{ display: { xs: "none", sm: "flex" } }}>
            <Button variant="text" onClick={() => navigate("/login")}>
              Login
            </Button>
            <Button
              variant="contained"
              onClick={() => navigate("/register")}
              endIcon={<ArrowForwardIcon />}
            >
              Get Started
            </Button>
          </Stack>
        </Box>

        <Grid container spacing={{ xs: 4, md: 6 }} alignItems="center">
          <Grid item xs={12} md={7}>
            <Stack spacing={3.5}>
              <Chip
                label="Phase 1 redesign"
                sx={{
                  alignSelf: "flex-start",
                  fontWeight: 800,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: theme.palette.primary.main,
                  border: "1px solid",
                  borderColor: alpha(theme.palette.primary.main, 0.14),
                }}
              />

              <Box>
                <Typography
                  variant="h1"
                  sx={{
                    fontSize: { xs: "2.75rem", md: "4.8rem" },
                    lineHeight: 1.02,
                    fontWeight: 900,
                    letterSpacing: "-0.05em",
                    maxWidth: 800,
                  }}
                >
                  Study with a calmer, faster workspace.
                </Typography>
                <Typography
                  variant="h5"
                  sx={{
                    mt: 2.5,
                    maxWidth: 760,
                    color: "text.secondary",
                    fontWeight: 400,
                    lineHeight: 1.65,
                  }}
                >
                  LearnFlow turns lectures into quizzes, flashcards, summaries,
                  and study plans without making the interface feel heavy or noisy.
                </Typography>
              </Box>

              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={() => navigate("/register")}
                  endIcon={<ArrowForwardIcon />}
                  sx={{ minWidth: 170 }}
                >
                  Get Started
                </Button>
                <Button
                  variant="outlined"
                  size="large"
                  onClick={() => navigate("/login")}
                  startIcon={<LoginIcon />}
                  sx={{ minWidth: 170 }}
                >
                  Login
                </Button>
              </Stack>

              <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
                {valueCards.map((card) => (
                  <SurfaceCard key={card.title} sx={{ flex: 1, minWidth: 0 }} contentSx={{ p: 2.5 }}>
                    <Stack spacing={1.5}>
                      <AccentIcon
                        background={card.iconBg}
                        color={theme.palette[card.iconColor.split(".")[0]].main}
                      >
                        {card.icon}
                      </AccentIcon>
                      <Box>
                        <Typography sx={{ fontWeight: 800, mb: 0.5 }}>
                          {card.title}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {card.description}
                        </Typography>
                      </Box>
                    </Stack>
                  </SurfaceCard>
                ))}
              </Stack>
            </Stack>
          </Grid>

          <Grid item xs={12} md={5}>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 2.5, md: 3.5 },
                borderRadius: 4,
                border: "1px solid",
                borderColor: "divider",
                background:
                  "linear-gradient(160deg, rgba(255,255,255,0.96) 0%, rgba(247,251,255,0.98) 100%)",
                boxShadow: "0 20px 60px rgba(19,32,58,0.10)",
              }}
            >
              <Stack spacing={3}>
                <Box>
                  <Typography
                    variant="overline"
                    sx={{ color: "primary.main", fontWeight: 800 }}
                  >
                    Why it feels different
                  </Typography>
                  <Typography
                    variant="h4"
                    sx={{ mt: 0.5, fontWeight: 800, letterSpacing: "-0.03em" }}
                  >
                    Everything is organized around the next action.
                  </Typography>
                </Box>

                <Stack spacing={1.5}>
                  {steps.map((step) => (
                    <Box
                      key={step.number}
                      sx={{ display: "flex", gap: 1.5, alignItems: "flex-start" }}
                    >
                      <Box
                        sx={{
                          width: 38,
                          height: 38,
                          borderRadius: 2,
                          bgcolor: "rgba(37,99,235,0.10)",
                          color: "primary.main",
                          display: "grid",
                          placeItems: "center",
                          fontWeight: 800,
                          flexShrink: 0,
                        }}
                      >
                        {step.number}
                      </Box>
                      <Box>
                        <Typography sx={{ fontWeight: 800 }}>{step.title}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {step.description}
                        </Typography>
                      </Box>
                    </Box>
                  ))}
                </Stack>

                <SurfaceCard contentSx={{ p: 2.5 }}>
                  <Stack direction="row" spacing={1.5} alignItems="center">
                    <Box
                      sx={{
                        width: 50,
                        height: 50,
                        borderRadius: 2,
                        display: "grid",
                        placeItems: "center",
                        bgcolor: "rgba(16,185,129,0.12)",
                        color: "success.main",
                      }}
                    >
                      <PlayCircleOutlineIcon />
                    </Box>
                    <Box>
                      <Typography sx={{ fontWeight: 800 }}>Quick start</Typography>
                      <Typography variant="body2" color="text.secondary">
                        Upload a lecture, generate practice, and review weak topics in one flow.
                      </Typography>
                    </Box>
                  </Stack>
                </SurfaceCard>
              </Stack>
            </Paper>
          </Grid>
        </Grid>

        <Stack spacing={3.5} sx={{ mt: { xs: 6, md: 10 } }}>
          <Box>
            <Typography variant="h3" sx={{ fontWeight: 900, letterSpacing: "-0.04em" }}>
              How it works
            </Typography>
            <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 700 }}>
              A simple loop: capture content, practice what matters, then plan the next session.
            </Typography>
          </Box>

          <Grid container spacing={2.5}>
            {steps.map((step) => (
              <Grid item xs={12} md={4} key={step.number}>
                <SurfaceCard sx={{ height: "100%" }}>
                  <Stack spacing={2}>
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        borderRadius: 2,
                        bgcolor: "rgba(124,58,237,0.10)",
                        color: "secondary.main",
                        display: "grid",
                        placeItems: "center",
                        fontWeight: 900,
                      }}
                    >
                      {step.number}
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 800 }}>
                      {step.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {step.description}
                    </Typography>
                  </Stack>
                </SurfaceCard>
              </Grid>
            ))}
          </Grid>
        </Stack>

        <Stack spacing={3.5} sx={{ mt: { xs: 6, md: 10 } }}>
          <Box>
            <Typography variant="h3" sx={{ fontWeight: 900, letterSpacing: "-0.04em" }}>
              Built for the core learning flows
            </Typography>
            <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 740 }}>
              The first release focuses on the surfaces people visit every day:
              dashboard, lectures, practice, planning, and review.
            </Typography>
          </Box>

          <Grid container spacing={2.5}>
            {featureGrid.map((feature) => (
              <Grid item xs={12} sm={6} key={feature.title}>
                <SurfaceCard>
                  <Stack spacing={1.25}>
                    <CheckCircleOutlineIcon color="primary" />
                    <Typography variant="h6" sx={{ fontWeight: 800 }}>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {feature.description}
                    </Typography>
                  </Stack>
                </SurfaceCard>
              </Grid>
            ))}
          </Grid>
        </Stack>

        <Stack spacing={3.5} sx={{ mt: { xs: 6, md: 10 } }}>
          <Box>
            <Typography variant="h3" sx={{ fontWeight: 900, letterSpacing: "-0.04em" }}>
              Trust and clarity built in
            </Typography>
            <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 700 }}>
              The interface stays readable, consistent, and focused even when the
              content gets dense.
            </Typography>
          </Box>

          <Grid container spacing={2.5}>
            {trustPoints.map((item) => (
              <Grid item xs={12} md={4} key={item.title}>
                <SurfaceCard sx={{ height: "100%" }}>
                  <Stack spacing={1.5}>
                    <AccentIcon
                      background={item.iconBg}
                      color={theme.palette[item.iconColor.split(".")[0]].main}
                    >
                      {item.icon}
                    </AccentIcon>
                    <Typography variant="h6" sx={{ fontWeight: 800 }}>
                      {item.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {item.text}
                    </Typography>
                  </Stack>
                </SurfaceCard>
              </Grid>
            ))}
          </Grid>
        </Stack>

        <Paper
          elevation={0}
          sx={{
            mt: { xs: 6, md: 10 },
            p: { xs: 3, md: 4 },
            borderRadius: 4,
            color: "#fff",
            background:
              "linear-gradient(135deg, #2563EB 0%, #7C3AED 55%, #EC4899 100%)",
            boxShadow: "0 18px 50px rgba(124,58,237,0.22)",
          }}
        >
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={2}
            alignItems={{ xs: "flex-start", md: "center" }}
            justifyContent="space-between"
          >
            <Box>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 900,
                  letterSpacing: "-0.04em",
                  color: "inherit",
                }}
              >
                Start with a cleaner learning loop.
              </Typography>
              <Typography sx={{ mt: 1, maxWidth: 720, color: "rgba(255,255,255,0.88)" }}>
                Create an account, add your first lecture, and move directly into
                practice or planning.
              </Typography>
            </Box>
            <Stack direction={{ xs: "column", sm: "row" }} spacing={1.5}>
              <Button
                variant="contained"
                onClick={() => navigate("/register")}
                sx={{ bgcolor: "#fff", color: "primary.main", '&:hover': { bgcolor: 'rgba(255,255,255,0.92)' } }}
              >
                Get Started
              </Button>
              <Button
                variant="outlined"
                onClick={() => navigate("/login")}
                sx={{
                  borderColor: "rgba(255,255,255,0.55)",
                  color: "#fff",
                  '&:hover': { borderColor: '#fff', bgcolor: 'rgba(255,255,255,0.10)' },
                }}
              >
                Login
              </Button>
            </Stack>
          </Stack>
        </Paper>
      </Container>
    </Box>
  );
}
