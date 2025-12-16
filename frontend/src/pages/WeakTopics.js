import React, { useState, useEffect } from "react";
import API from "../api/api";
import {
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  LinearProgress,
  CircularProgress,
} from "@mui/material";
import { useParams } from "react-router-dom";

function WeakTopics() {
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(true);
  const { noteId } = useParams();

  useEffect(() => {
    setLoading(true);
    API.get(`weak-topics/?note_id=${noteId}`)
      .then((res) => setTopics(res.data.weak_topics || []))
      .catch((err) => {
        console.error("Error fetching weak topics:", err);
        setTopics([]);
      })
      .finally(() => setLoading(false));
  }, [noteId]);

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
        📚 Weak Topics for Lecture {noteId}
      </Typography>

      {loading ? (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            minHeight: "300px",
          }}
        >
          <CircularProgress />
        </Box>
      ) : topics.length === 0 ? (
        <Card
          sx={{
            p: 4,
            background: "linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
            borderRadius: 3,
            textAlign: "center",
          }}
        >
          <Typography sx={{ color: "#999", fontStyle: "italic" }}>
            ℹ️ No weak topics found for this lecture note yet.
          </Typography>
        </Card>
      ) : (
        <Grid container spacing={3}>
          {(() => {
            // Calculate max score for comparative visualization
            const maxScore = Math.max(
              ...topics.map((t) => parseFloat(t.score ?? 0)),
              1
            );

            return topics.map((t, i) => {
              const score = parseFloat(t.score ?? 0);
              // Calculate percentage relative to max score (comparative)
              const percentage = Math.round((score / maxScore) * 100);
              const isHigh = percentage >= 70;
              const isMedium = percentage >= 40 && percentage < 70;

              return (
                <Grid item xs={12} sm={6} key={i}>
                  <Card
                    sx={{
                      p: 3,
                      background: isHigh
                        ? "linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)"
                        : isMedium
                        ? "linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)"
                        : "linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)",
                      boxShadow: "0 6px 20px rgba(0,0,0,0.1)",
                      borderRadius: 2,
                      borderLeft: `6px solid ${
                        isHigh ? "#dc3545" : isMedium ? "#ffc107" : "#28a745"
                      }`,
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
                          mb: 2,
                        }}
                      >
                        <Typography
                          variant="h6"
                          sx={{
                            fontWeight: "bold",
                            color: "#333",
                            flex: 1,
                          }}
                        >
                          {isHigh && "🔴"}
                          {isMedium && "🟡"}
                          {!isHigh && !isMedium && "🟢"} {t.topic}
                        </Typography>
                        <Typography
                          sx={{
                            fontWeight: "bold",
                            color: isHigh
                              ? "#dc3545"
                              : isMedium
                              ? "#ffc107"
                              : "#28a745",
                            fontSize: "1.1rem",
                            ml: 2,
                            minWidth: "50px",
                            textAlign: "right",
                          }}
                        >
                          {percentage}%
                        </Typography>
                      </Box>
                      <LinearProgress
                        variant="determinate"
                        value={percentage}
                        sx={{
                          height: 12,
                          borderRadius: 10,
                          backgroundColor: "rgba(255,255,255,0.5)",
                          mb: 2,
                          "& .MuiLinearProgress-bar": {
                            borderRadius: 10,
                            background: isHigh
                              ? "linear-gradient(90deg, #dc3545 0%, #c82333 100%)"
                              : isMedium
                              ? "linear-gradient(90deg, #ffc107 0%, #e0a800 100%)"
                              : "linear-gradient(90deg, #28a745 0%, #218838 100%)",
                          },
                        }}
                      />
                      <Box
                        sx={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                        }}
                      >
                        <Typography
                          sx={{
                            fontSize: "0.9rem",
                            color: "#555",
                            fontWeight: 500,
                          }}
                        >
                          Weakness Score:
                        </Typography>
                        <Typography
                          sx={{
                            fontSize: "0.9rem",
                            color: isHigh
                              ? "#dc3545"
                              : isMedium
                              ? "#ffc107"
                              : "#28a745",
                            fontWeight: 700,
                          }}
                        >
                          {score.toFixed(2)}
                        </Typography>
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            });
          })()}
        </Grid>
      )}
    </Container>
  );
}

export default WeakTopics;
