import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import API from "../api/api";
import {
  Container,
  TextField,
  Button,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Box,
} from "@mui/material";

export default function StudyPlan() {
  const [noteId, setNoteId] = useState("");
  const [plan, setPlan] = useState("");
  const [strengths, setStrengths] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const location = useLocation();

  // Auto-load noteId from URL or state and generate plan
  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const noteIdParam = searchParams.get("noteId") || location.state?.noteId;
    
    if (noteIdParam) {
      setNoteId(noteIdParam);
      // Auto-generate plan
      generatePlan(noteIdParam);
    }
  }, [location.search, location.state]);

  const generatePlan = async (id) => {
    const targetId = id || noteId;
    if (!targetId) return;

    setLoading(true);
    try {
      const res = await API.post("study-plan/", { note_id: targetId });

      setPlan(res.data.plan || "");
      setStrengths(res.data.strengths || {});
    } catch (err) {
      console.error(err);
      setError("Failed to generate study plan. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const generate = () => generatePlan();



  if (loading && !plan) {
    return (
      <Container maxWidth="md" style={{ marginTop: 40, textAlign: "center" }}>
        <CircularProgress />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Generating Study Plan for Note #{noteId}...
        </Typography>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" style={{ marginTop: 40, marginBottom: 40 }}>
      <Card>
        <CardContent>
          <Typography
            variant="h4"
            sx={{ fontWeight: "bold", mb: 2, textAlign: "center" }}
          >
            Study Plan
          </Typography>

          {error && (
            <Typography color="error" sx={{ mb: 2, textAlign: "center" }}>
              {error}
            </Typography>
          )}

          {/* Input */}
          <TextField
            label="Lecture Note ID"
            value={noteId}
            onChange={(e) => setNoteId(e.target.value)}
            fullWidth
            margin="normal"
          />

          {/* Button */}
          <Button
            variant="contained"
            onClick={generate}
            disabled={loading}
            fullWidth
            sx={{
              background: "linear-gradient(135deg, #4f46e5, #6366f1)",
              borderRadius: "10px",
              py: 1.2,
              px: 4,
              fontWeight: "bold",
              mt: 1,
            }}
          >
            {loading ? (
              <CircularProgress size={18} color="inherit" />
            ) : (
              "Generate Study Plan"
            )}
          </Button>

          {/* Strength Topics Card */}
          {Object.keys(strengths).length > 0 && (
            <Card
              sx={{
                mt: 4,
                p: 3,
                border: "2px solid rgba(0,0,0,0.15)",
                borderRadius: "14px",
                boxShadow: "0px 4px 15px rgba(0,0,0,0.08)",
                background: "#fff",
              }}
            >
              <Typography variant="h5" sx={{ mb: 2, fontWeight: "bold" }}>
                Strong Topics
              </Typography>

              {Object.entries(strengths)
                .sort((a, b) => b[1] - a[1]) // highest strength first
                .map(([topic, score], idx) => {
                  const percent = Math.round(score * 100);

                  return (
                    <Box key={idx} sx={{ mb: 3 }}>
                      <Typography sx={{ fontWeight: "bold" }}>
                        {topic} — {percent}%
                      </Typography>

                      <Box
                        sx={{
                          mt: 1,
                          height: 10,
                          background: "#E0F2FE",
                          borderRadius: 6,
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          sx={{
                            height: "100%",
                            width: `${percent}%`,
                            background: "#38BDF8",
                            borderRadius: 6,
                          }}
                        />
                      </Box>
                    </Box>
                  );
                })}
            </Card>
          )}

          {/* Study Plan Text */}
          {plan && (
            <Card
              style={{
                marginTop: 30,
                padding: 20,
                background: "#fafafa",
                borderRadius: "12px",
              }}
            >
              <pre style={{ whiteSpace: "pre-wrap", fontSize: "15px" }}>
                {plan}
              </pre>
            </Card>
          )}
        </CardContent>
      </Card>
    </Container>
  );
}
