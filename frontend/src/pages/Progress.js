import React, { useEffect, useState } from "react";
import API from "../api/api";
import {
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  CircularProgress,
  LinearProgress,
} from "@mui/material";

function Progress() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    API.get("progress/")
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error("Error fetching progress:", err);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <Container
        maxWidth="md"
        sx={{
          mt: 5,
          mb: 5,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "400px",
        }}
      >
        <CircularProgress />
      </Container>
    );
  }

  if (!data) {
    return (
      <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
        <Typography sx={{ textAlign: "center", color: "#999" }}>
          Unable to load progress data.
        </Typography>
      </Container>
    );
  }

  const accuracyPercent = Math.min(
    Math.max(parseFloat(data.accuracy ?? 0), 0),
    100
  );
  const correctPercent = data.total_questions
    ? Math.round((data.correct_answers / data.total_questions) * 100)
    : 0;

  return (
    <Container maxWidth="md" sx={{ mt: 5, mb: 5 }}>
      <Typography
        variant="h3"
        gutterBottom
        sx={{
          fontWeight: "bold",
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          mb: 4,
        }}
      >
        📊 Your Learning Progress
      </Typography>

      <Grid container spacing={3}>
        {/* Accuracy Card */}
        <Grid item xs={12} sm={6}>
          <Card
            sx={{
              p: 3,
              background: "linear-gradient(135deg, #e0f2fe 0%, #cffafe 100%)",
              boxShadow: "0 6px 20px rgba(0,0,0,0.1)",
              borderRadius: 2,
              borderLeft: "6px solid #0ea5e9",
              transition: "all 0.3s ease",
              "&:hover": {
                boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
                transform: "translateY(-4px)",
              },
            }}
          >
            <CardContent sx={{ p: 0 }}>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: "bold",
                  color: "#333",
                  mb: 2,
                }}
              >
                📈 Accuracy Rate
              </Typography>
              <Typography
                variant="h3"
                sx={{
                  fontWeight: "bold",
                  color: "#0ea5e9",
                  mb: 2,
                }}
              >
                {accuracyPercent.toFixed(2)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={accuracyPercent}
                sx={{
                  height: 12,
                  borderRadius: 10,
                  backgroundColor: "rgba(255,255,255,0.5)",
                  "& .MuiLinearProgress-bar": {
                    borderRadius: 10,
                    background:
                      "linear-gradient(90deg, #06b6d4 0%, #0ea5e9 100%)",
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Correct Answers Card */}
        <Grid item xs={12} sm={6}>
          <Card
            sx={{
              p: 3,
              background: "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)",
              boxShadow: "0 6px 20px rgba(0,0,0,0.1)",
              borderRadius: 2,
              borderLeft: "6px solid #22c55e",
              transition: "all 0.3s ease",
              "&:hover": {
                boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
                transform: "translateY(-4px)",
              },
            }}
          >
            <CardContent sx={{ p: 0 }}>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: "bold",
                  color: "#333",
                  mb: 2,
                }}
              >
                ✅ Correct Answers
              </Typography>
              <Box sx={{ display: "flex", alignItems: "center", gap: 2 }}>
                <Box>
                  <Typography
                    variant="h3"
                    sx={{
                      fontWeight: "bold",
                      color: "#22c55e",
                    }}
                  >
                    {data.correct_answers}
                  </Typography>
                  <Typography
                    sx={{
                      fontSize: "0.9rem",
                      color: "#555",
                      fontWeight: 500,
                    }}
                  >
                    of {data.total_questions}
                  </Typography>
                </Box>
                <Box
                  sx={{
                    flex: 1,
                    textAlign: "right",
                  }}
                >
                  <Typography
                    sx={{
                      fontSize: "1.3rem",
                      fontWeight: "bold",
                      color: "#22c55e",
                    }}
                  >
                    {correctPercent}%
                  </Typography>
                </Box>
              </Box>
              <LinearProgress
                variant="determinate"
                value={correctPercent}
                sx={{
                  height: 12,
                  borderRadius: 10,
                  backgroundColor: "rgba(255,255,255,0.5)",
                  mt: 1.5,
                  "& .MuiLinearProgress-bar": {
                    borderRadius: 10,
                    background:
                      "linear-gradient(90deg, #16a34a 0%, #22c55e 100%)",
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Total Questions Card */}
        <Grid item xs={12}>
          <Card
            sx={{
              p: 3,
              background: "linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%)",
              boxShadow: "0 6px 20px rgba(0,0,0,0.1)",
              borderRadius: 2,
              borderLeft: "6px solid #a855f7",
              transition: "all 0.3s ease",
              "&:hover": {
                boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
                transform: "translateY(-4px)",
              },
            }}
          >
            <CardContent sx={{ p: 0 }}>
              <Box
                sx={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <Box>
                  <Typography
                    variant="h6"
                    sx={{
                      fontWeight: "bold",
                      color: "#333",
                      mb: 1,
                    }}
                  >
                    📝 Total Questions Answered
                  </Typography>
                  <Typography sx={{ fontSize: "0.9rem", color: "#666" }}>
                    Complete your learning journey
                  </Typography>
                </Box>
                <Typography
                  variant="h3"
                  sx={{
                    fontWeight: "bold",
                    color: "#a855f7",
                    minWidth: "100px",
                    textAlign: "right",
                  }}
                >
                  {data.total_questions}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
}

export default Progress;
