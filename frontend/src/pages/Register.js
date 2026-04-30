import React, { useMemo, useState } from "react";
import { Link as RouterLink, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Container,
  Divider,
  Grid,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import {
  ArrowForward as ArrowForwardIcon,
  CheckCircleOutline as CheckCircleOutlineIcon,
  PersonAdd as PersonAddIcon,
  School as SchoolIcon,
  Star as StarIcon,
} from "@mui/icons-material";
import { useAuth } from "../context/AuthContext";

const promises = [
  "Organize lectures, quizzes, and flashcards in one workspace",
  "Keep study plans and weak-topic reviews easy to revisit",
  "Get a dashboard that highlights the next best action",
];

const passwordRules = [
  "Use at least 8 characters.",
  "Include a mix of letters and numbers when possible.",
  "Make sure both password fields match before submitting.",
];

export default function Register() {
  const [formData, setFormData] = useState({
    username: "",
    email: "",
    password: "",
    password2: "",
    first_name: "",
    last_name: "",
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const passwordsMatch = useMemo(
    () => formData.password.length === 0 || formData.password === formData.password2,
    [formData.password, formData.password2],
  );

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((current) => ({ ...current, [name]: value }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError("");

    if (formData.password !== formData.password2) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    const result = await register(formData);

    if (result.success) {
      navigate("/dashboard");
      return;
    }

    let errorMessage = "Registration failed. Please try again.";
    if (typeof result.error === "string") {
      errorMessage = result.error;
    } else if (result.error && typeof result.error === "object") {
      const values = Object.values(result.error).flat();
      if (values.length > 0) {
        errorMessage = values[0];
      }
    }

    setError(errorMessage);
    setLoading(false);
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
                    Create account
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
                    Start with a calmer study workspace.
                  </Typography>
                  <Typography
                    sx={{
                      mt: 2,
                      color: "rgba(255,255,255,0.9)",
                      maxWidth: 440,
                      lineHeight: 1.7,
                    }}
                  >
                    Join LearnFlow to capture lectures, practice actively, and plan
                    the next session without switching tools.
                  </Typography>
                </Box>

                <Stack spacing={1.5}>
                  {promises.map((promise) => (
                    <Box
                      key={promise}
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
                        {promise}
                      </Typography>
                    </Box>
                  ))}
                </Stack>

                <Box
                  sx={{
                    p: 2,
                    borderRadius: 3,
                    bgcolor: "rgba(255,255,255,0.12)",
                    border: "1px solid rgba(255,255,255,0.16)",
                  }}
                >
                  <Typography sx={{ fontWeight: 800, color: "inherit" }}>
                    What you get first
                  </Typography>
                  <Typography
                    variant="body2"
                    sx={{ color: "rgba(255,255,255,0.84)", mt: 0.5 }}
                  >
                    A dashboard, lectures library, quiz flow, flashcards, and study
                    plan in one place.
                  </Typography>
                </Box>
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
                      Built for momentum
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{ color: "rgba(255,255,255,0.84)" }}
                    >
                      The first screen points you to the next step.
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
              <Stack spacing={3} component="form" onSubmit={handleSubmit}>
                <Box>
                  <Typography
                    variant="overline"
                    sx={{ color: "secondary.main", fontWeight: 800 }}
                  >
                    Sign up
                  </Typography>
                  <Typography
                    variant="h4"
                    sx={{ mt: 0.5, fontWeight: 900, letterSpacing: "-0.04em" }}
                  >
                    Create your LearnFlow account
                  </Typography>
                  <Typography color="text.secondary" sx={{ mt: 1 }}>
                    Set up your learning profile in less than a minute.
                  </Typography>
                </Box>

                {error ? <Alert severity="error">{error}</Alert> : null}

                <Grid container spacing={2}>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      label="First name"
                      name="first_name"
                      value={formData.first_name}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <TextField
                      label="Last name"
                      name="last_name"
                      value={formData.last_name}
                      onChange={handleChange}
                      fullWidth
                    />
                  </Grid>
                </Grid>

                <TextField
                  label="Username"
                  name="username"
                  value={formData.username}
                  onChange={handleChange}
                  autoComplete="username"
                  required
                  fullWidth
                />
                <TextField
                  label="Email"
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleChange}
                  autoComplete="email"
                  required
                  fullWidth
                />
                <TextField
                  label="Password"
                  name="password"
                  type="password"
                  value={formData.password}
                  onChange={handleChange}
                  autoComplete="new-password"
                  required
                  fullWidth
                  helperText="Choose a password you can re-enter on another device."
                />
                <TextField
                  label="Confirm password"
                  name="password2"
                  type="password"
                  value={formData.password2}
                  onChange={handleChange}
                  autoComplete="new-password"
                  required
                  error={!passwordsMatch}
                  helperText={!passwordsMatch ? "Passwords must match." : " "}
                  fullWidth
                />

                <Box
                  sx={{
                    p: 2.25,
                    borderRadius: 3,
                    bgcolor: "rgba(37,99,235,0.06)",
                    border: "1px solid rgba(37,99,235,0.14)",
                  }}
                >
                  <Typography sx={{ fontWeight: 800, mb: 1 }}>
                    Password guidance
                  </Typography>
                  <Stack spacing={0.8}>
                    {passwordRules.map((rule) => (
                      <Box key={rule} sx={{ display: "flex", gap: 1, alignItems: "flex-start" }}>
                        <CheckCircleOutlineIcon
                          sx={{ fontSize: 18, color: "primary.main", mt: 0.2 }}
                        />
                        <Typography variant="body2" color="text.secondary">
                          {rule}
                        </Typography>
                      </Box>
                    ))}
                  </Stack>
                </Box>

                <Button
                  type="submit"
                  variant="contained"
                  size="large"
                  disabled={loading}
                  endIcon={<ArrowForwardIcon />}
                >
                  {loading ? "Creating account..." : "Create account"}
                </Button>

                <Divider>
                  <Typography variant="caption" sx={{ color: "text.secondary", px: 1 }}>
                    Already have an account?
                  </Typography>
                </Divider>

                <Button
                  component={RouterLink}
                  to="/login"
                  variant="text"
                  startIcon={<PersonAddIcon />}
                >
                  Sign in instead
                </Button>
              </Stack>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
