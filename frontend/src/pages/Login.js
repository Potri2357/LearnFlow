import React, { useState } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Container,
  Divider,
  Grid,
  Link,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import {
  ArrowForward as ArrowForwardIcon,
  Login as LoginIcon,
  Quiz as QuizIcon,
  School as SchoolIcon,
  Star as StarIcon,
} from "@mui/icons-material";
import { useAuth } from "../context/AuthContext";

const benefits = [
  "Adaptive quizzes from your lecture notes",
  "One dashboard for priorities and weak topics",
  "Study plans and flashcards in the same flow",
];

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);

    const result = await login(username, password);
    if (result.success) {
      navigate("/dashboard");
    } else {
      setError(result.error || "Sign in failed. Please check your credentials.");
    }

    setLoading(false);
  };

  const handleGoogleLogin = () => {
    navigate("/google-login");
  };

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "radial-gradient(circle at top left, rgba(37,99,235,0.12) 0, transparent 28%), radial-gradient(circle at 85% 18%, rgba(124,58,237,0.12) 0, transparent 24%), linear-gradient(180deg, #F7FAFF 0%, #EEF4FF 100%)",
        py: { xs: 3, md: 5 },
      }}
    >
      <Container maxWidth="lg">
        <Grid container spacing={3.5} alignItems="stretch">
          <Grid item xs={12} md={5} sx={{ display: "flex" }}>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 3, md: 4 },
                borderRadius: 4,
                color: "#fff",
                background:
                  "linear-gradient(160deg, #1D4ED8 0%, #5B21B6 58%, #C026D3 100%)",
                boxShadow: "0 24px 70px rgba(37,99,235,0.28)",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                minHeight: { md: 720 },
                width: "100%",
              }}
            >
              <Stack spacing={3}>
                <Box sx={{ display: "inline-flex", alignItems: "center", gap: 1.2 }}>
                  <Box
                    sx={{
                      width: 44,
                      height: 44,
                      borderRadius: 2,
                      bgcolor: "rgba(255,255,255,0.18)",
                      display: "grid",
                      placeItems: "center",
                      fontWeight: 900,
                    }}
                  >
                    LF
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 900, color: "inherit" }}>
                      LearnFlow
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{ color: "rgba(255,255,255,0.8)" }}
                    >
                      Premium light-first study workspace
                    </Typography>
                  </Box>
                </Box>

                <Box>
                  <Typography
                    variant="overline"
                    sx={{ color: "rgba(255,255,255,0.82)", letterSpacing: "0.14em" }}
                  >
                    Welcome back
                  </Typography>
                  <Typography
                    variant="h3"
                    sx={{
                      mt: 1,
                      fontWeight: 900,
                      letterSpacing: "-0.04em",
                      color: "inherit",
                    }}
                  >
                    Sign in to continue your learning loop.
                  </Typography>
                  <Typography
                    sx={{
                      mt: 2,
                      color: "rgba(255,255,255,0.9)",
                      maxWidth: 440,
                      lineHeight: 1.7,
                    }}
                  >
                    Keep your dashboard, lecture notes, quizzes, flashcards, and
                    study plan in one place.
                  </Typography>
                </Box>

                <Stack spacing={1.5}>
                  {benefits.map((benefit) => (
                    <Box
                      key={benefit}
                      sx={{ display: "flex", gap: 1.2, alignItems: "flex-start" }}
                    >
                      <Box
                        sx={{
                          width: 26,
                          height: 26,
                          borderRadius: "50%",
                          bgcolor: "rgba(255,255,255,0.16)",
                          display: "grid",
                          placeItems: "center",
                          flexShrink: 0,
                        }}
                      >
                        <StarIcon sx={{ fontSize: 14, color: "#fff" }} />
                      </Box>
                      <Typography sx={{ color: "rgba(255,255,255,0.92)" }}>
                        {benefit}
                      </Typography>
                    </Box>
                  ))}
                </Stack>

                <Stack direction="row" spacing={1.5} sx={{ flexWrap: "wrap" }}>
                  <Box
                    sx={{
                      flex: 1,
                      minWidth: 140,
                      p: 2,
                      borderRadius: 3,
                      bgcolor: "rgba(255,255,255,0.12)",
                      border: "1px solid rgba(255,255,255,0.16)",
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "rgba(255,255,255,0.78)" }}
                    >
                      Practice mode
                    </Typography>
                    <Typography
                      variant="h5"
                      sx={{ fontWeight: 900, color: "inherit", mt: 0.3 }}
                    >
                      Faster
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      flex: 1,
                      minWidth: 140,
                      p: 2,
                      borderRadius: 3,
                      bgcolor: "rgba(255,255,255,0.12)",
                      border: "1px solid rgba(255,255,255,0.16)",
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "rgba(255,255,255,0.78)" }}
                    >
                      Study flow
                    </Typography>
                    <Typography
                      variant="h5"
                      sx={{ fontWeight: 900, color: "inherit", mt: 0.3 }}
                    >
                      Clearer
                    </Typography>
                  </Box>
                </Stack>
              </Stack>

              <Box sx={{ mt: 4, display: { xs: "none", md: "block" } }}>
                <Box
                  sx={{
                    display: "flex",
                    gap: 1.5,
                    alignItems: "center",
                    p: 2,
                    borderRadius: 3,
                    bgcolor: "rgba(255,255,255,0.12)",
                    border: "1px solid rgba(255,255,255,0.14)",
                  }}
                >
                  <Box
                    sx={{
                      width: 42,
                      height: 42,
                      borderRadius: 2,
                      display: "grid",
                      placeItems: "center",
                      bgcolor: "rgba(255,255,255,0.16)",
                    }}
                  >
                    <SchoolIcon />
                  </Box>
                  <Box>
                    <Typography sx={{ fontWeight: 800, color: "inherit" }}>
                      Focused learning
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{ color: "rgba(255,255,255,0.84)" }}
                    >
                      Every screen keeps the next step visible.
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </Paper>
          </Grid>

          <Grid item xs={12} md={7} sx={{ display: "flex" }}>
            <Paper
              elevation={0}
              sx={{
                p: { xs: 3, md: 4 },
                borderRadius: 4,
                border: "1px solid",
                borderColor: "divider",
                bgcolor: "rgba(255,255,255,0.92)",
                boxShadow: "0 20px 54px rgba(19,32,58,0.10)",
                width: "100%",
              }}
            >
              <Stack spacing={3}>
                <Box>
                  <Typography
                    variant="overline"
                    sx={{ color: "primary.main", fontWeight: 800 }}
                  >
                    Sign in
                  </Typography>
                  <Typography
                    variant="h4"
                    sx={{ mt: 0.5, fontWeight: 900, letterSpacing: "-0.04em" }}
                  >
                    Welcome back to LearnFlow
                  </Typography>
                  <Typography color="text.secondary" sx={{ mt: 1 }}>
                    Use your existing account or continue with Google.
                  </Typography>
                </Box>

                {error ? <Alert severity="error">{error}</Alert> : null}

                <Button
                  type="button"
                  variant="outlined"
                  onClick={handleGoogleLogin}
                  startIcon={<LoginIcon />}
                  sx={{ py: 1.4, borderRadius: 3 }}
                >
                  Continue with Google
                </Button>

                <Divider>
                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", px: 1 }}
                  >
                    or sign in with username
                  </Typography>
                </Divider>

                <Box component="form" onSubmit={handleSubmit}>
                  <Stack spacing={2}>
                    <TextField
                      label="Username"
                      value={username}
                      onChange={(event) => setUsername(event.target.value)}
                      autoComplete="username"
                      required
                      fullWidth
                    />
                    <TextField
                      label="Password"
                      type="password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      autoComplete="current-password"
                      required
                      fullWidth
                      helperText="Use the same password you created during registration."
                    />
                    <Button
                      type="submit"
                      variant="contained"
                      size="large"
                      disabled={loading}
                      endIcon={<ArrowForwardIcon />}
                    >
                      {loading ? "Signing in..." : "Sign in"}
                    </Button>
                  </Stack>
                </Box>

                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ textAlign: "center" }}
                >
                  Don&apos;t have an account? {" "}
                  <Link
                    component={RouterLink}
                    to="/register"
                    underline="hover"
                    sx={{ fontWeight: 800 }}
                  >
                    Create one
                  </Link>
                </Typography>
              </Stack>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
