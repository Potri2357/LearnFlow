// src/pages/Dashboard.js
import React, { useState, useEffect, useCallback } from "react";
import API from "../api/api";
import {
  Container,
  Typography,
  TextField,
  Button,
  Card,
  CardContent,
  Grid,
  Box,
  LinearProgress,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
} from "@mui/material";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"];

export default function Dashboard() {
  const [noteId, setNoteId] = useState("");
  const [lectures, setLectures] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [previousAccuracy, setPreviousAccuracy] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch lectures on mount
  useEffect(() => {
    const fetchLectures = async () => {
      try {
        const response = await API.get('lectures/');
        setLectures(response.data);
      } catch (err) {
        console.error('Failed to fetch lectures', err);
      }
    };
    fetchLectures();
  }, []);

  // Load analytics function
  const loadAnalytics = useCallback(async (id) => {
    const targetId = id || noteId;
    
    console.log('📊 loadAnalytics called with ID:', targetId);
    
    if (!targetId) {
      setError("Please select a Lecture Note.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      console.log('🌐 Fetching analytics from API...');
      const res = await API.get(`analytics/${targetId}/`);
      const data = res.data;
      
      console.log('✅ Analytics received:', {
        mastery_score: data.mastery_score,
        weak_topics_count: data.top_weak_topics?.length || 0,
        has_trend_data: !!data.accuracy_trend_last7
      });

      if (analytics?.mastery_score != null) {
        setPreviousAccuracy(analytics.mastery_score);
        console.log('Previous accuracy saved:', analytics.mastery_score);
      }

      setAnalytics(data);
      localStorage.setItem('lastSelectedNoteId', String(targetId));
      console.log('Analytics state updated successfully');
    } catch (err) {
      console.error('❌ Analytics error:', err);
      setError(err.response?.data?.error || "Failed to load analytics.");
    } finally {
      setLoading(false);
    }
  }, [noteId, analytics]);

  // Auto-load on mount and refresh - Check EVERY time
  useEffect(() => {
    if (lectures.length === 0) return;

    console.log('=== Dashboard Auto-Load Check ===');
    const needsRefresh = localStorage.getItem('dashboardNeedsRefresh');
    const lastQuizNoteId = localStorage.getItem('lastQuizNoteId');
    const lastSelectedNoteId = localStorage.getItem('lastSelectedNoteId');

    console.log('needsRefresh:', needsRefresh);
    console.log('lastQuizNoteId:', lastQuizNoteId);
    console.log('lastSelectedNoteId:', lastSelectedNoteId);
    console.log('current noteId:', noteId);
    console.log('has analytics:', !!analytics);

    // ALWAYS check for refresh flag, even if we have analytics
    if (needsRefresh === 'true' && lastQuizNoteId) {
      console.log('🔄 REFRESHING after quiz completion');
      localStorage.removeItem('dashboardNeedsRefresh');
      const lecture = lectures.find(l => String(l.id) === String(lastQuizNoteId));
      if (lecture) {
        console.log('Found lecture:', lecture.title);
        setNoteId(String(lecture.id));
        loadAnalytics(String(lecture.id));
        return;
      } else {
        console.log('❌ Lecture not found for ID:', lastQuizNoteId);
      }
    }

    // Only auto-load if we don't have analytics yet
    if (!noteId && !analytics && lastSelectedNoteId) {
      console.log('📂 Loading last selected note');
      const lecture = lectures.find(l => String(l.id) === String(lastSelectedNoteId));
      if (lecture) {
        console.log('Found lecture:', lecture.title);
        setNoteId(String(lecture.id));
        loadAnalytics(String(lecture.id));
      } else {
        console.log('❌ Lecture not found for ID:', lastSelectedNoteId);
      }
    }
  }, [lectures, loadAnalytics]); // Removed noteId and analytics from dependencies

  // Listen for window focus
  useEffect(() => {
    const handleFocus = () => {
      if (localStorage.getItem('dashboardNeedsRefresh') === 'true') {
        const lastQuizNoteId = localStorage.getItem('lastQuizNoteId');
        if (lastQuizNoteId && lectures.length > 0) {
          localStorage.removeItem('dashboardNeedsRefresh');
          const lecture = lectures.find(l => String(l.id) === String(lastQuizNoteId));
          if (lecture) {
            setNoteId(String(lecture.id));
            loadAnalytics(String(lecture.id));
          }
        }
      }
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [lectures, loadAnalytics]);

  const weakTopics = analytics?.top_weak_topics || [];
  const maxWeakness = weakTopics.length > 0 ? Math.max(...weakTopics.map(t => t.weakness_score)) : 1;

  return (
    <Container maxWidth="xl" sx={{ mt: 5, mb: 5 }}>
      <Typography
        variant="h2"
        sx={{
          textAlign: "center",
          fontWeight: "bold",
          mb: 4,
          background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
          backgroundClip: "text",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
        }}
      >
        📊 LearnFlow Dashboard
      </Typography>

      {/* Input Section */}
      <Box sx={{ display: "flex", justifyContent: "center", gap: 2, mb: 4, flexWrap: { xs: "wrap", sm: "nowrap" } }}>
        <FormControl 
          variant="outlined" 
          sx={{ 
            flex: 1,
            minWidth: "200px",
            maxWidth: { xs: "100%", sm: "350px" },
          }}
        >
          <InputLabel>Select Lecture Note</InputLabel>
          <Select
            value={noteId}
            onChange={(e) => setNoteId(String(e.target.value))}
            label="Select Lecture Note"
          >
            {lectures.map((l) => (
              <MenuItem key={l.id} value={l.id}>{l.title}</MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button
          variant="contained"
          onClick={() => loadAnalytics()}
          disabled={loading || !noteId}
          sx={{
            height: "56px",
            minWidth: "120px",
            background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            fontWeight: "bold",
          }}
        >
          {loading ? "LOADING..." : "LOAD"}
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>}
      {loading && <LinearProgress sx={{ mb: 3 }} />}

      {/* Refresh Available Indicator */}
      {localStorage.getItem('dashboardNeedsRefresh') === 'true' && (
        <Alert severity="info" sx={{ mb: 3 }}>
          📊 New quiz data available! 
          <Button 
            onClick={() => {
              const lastQuizNoteId = localStorage.getItem('lastQuizNoteId');
              if (lastQuizNoteId) {
                localStorage.removeItem('dashboardNeedsRefresh');
                setNoteId(lastQuizNoteId);
                loadAnalytics(lastQuizNoteId);
              }
            }}
            sx={{ ml: 2 }}
            variant="contained"
            size="small"
          >
            Refresh Now
          </Button>
        </Alert>
      )}

          {analytics && (
        <>
          {/* Mastery Score Card */}
          <Card sx={{ mb: 4, background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)", color: "white" }}>
            <CardContent>
              <Typography variant="h4" sx={{ fontWeight: "bold", textAlign: "center" }}>
                Overall Mastery Score
              </Typography>
              <Typography variant="h2" sx={{ fontWeight: "bold", textAlign: "center", mt: 2 }}>
                {analytics.mastery_score?.toFixed(1) || 0}%
              </Typography>
              {previousAccuracy != null && (
                <Typography variant="body1" sx={{ textAlign: "center", mt: 1 }}>
                  Previous: {previousAccuracy.toFixed(1)}%
                  {analytics.mastery_score > previousAccuracy && " 📈"}
                  {analytics.mastery_score < previousAccuracy && " 📉"}
                </Typography>
              )}
            </CardContent>
          </Card>

          <Grid container spacing={3}>
            {/* Weak Topics */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h5" sx={{ fontWeight: "bold", mb: 2 }}>
                    🎯 Top Weak Topics
                  </Typography>
                  {weakTopics.length === 0 ? (
                    <Box sx={{ textAlign: "center", py: 4 }}>
                      <Typography color="text.secondary">
                        No weak topics identified yet. Complete more quizzes to see your weak areas!
                      </Typography>
                    </Box>
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={weakTopics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="topic" angle={-45} textAnchor="end" height={100} />
                        <YAxis domain={[0, maxWeakness + 1]} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="weakness_score" fill="#FF8042" name="Weakness Score" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Difficulty Accuracy */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h5" sx={{ fontWeight: "bold", mb: 2 }}>
                    📈 Difficulty Accuracy
                  </Typography>
                  {!analytics.difficulty_accuracy?.easy && !analytics.difficulty_accuracy?.medium && !analytics.difficulty_accuracy?.hard ? (
                    <Box sx={{ textAlign: "center", py: 4 }}>
                      <Typography color="text.secondary">
                        No quiz data available. Complete quizzes to see your performance by difficulty!
                      </Typography>
                    </Box>
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart
                        data={[
                          { 
                            difficulty: "Easy", 
                            accuracy: analytics.difficulty_accuracy?.easy?.accuracy || 0,
                            total: analytics.difficulty_accuracy?.easy?.total || 0
                          },
                          { 
                            difficulty: "Medium", 
                            accuracy: analytics.difficulty_accuracy?.medium?.accuracy || 0,
                            total: analytics.difficulty_accuracy?.medium?.total || 0
                          },
                          { 
                            difficulty: "Hard", 
                            accuracy: analytics.difficulty_accuracy?.hard?.accuracy || 0,
                            total: analytics.difficulty_accuracy?.hard?.total || 0
                          },
                        ]}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="difficulty" />
                        <YAxis domain={[0, 100]} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="accuracy" fill="#00C49F" name="Accuracy %" />
                      </BarChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Accuracy Trend */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h5" sx={{ fontWeight: "bold", mb: 2 }}>
                    📊 7-Day Accuracy Trend
                  </Typography>
                  {!analytics.accuracy_trend_last7 || analytics.accuracy_trend_last7.every(d => d.accuracy === 0) ? (
                    <Box sx={{ textAlign: "center", py: 4 }}>
                      <Typography color="text.secondary">
                        No quiz activity in the last 7 days. Complete quizzes to see your trend!
                      </Typography>
                    </Box>
                  ) : (
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={analytics.accuracy_trend_last7}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis domain={[0, 100]} label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="accuracy" 
                          stroke="#667eea" 
                          strokeWidth={3} 
                          dot={{ r: 5 }}
                          activeDot={{ r: 8 }}
                          name="Accuracy %"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Topic Mastery */}
            {analytics.topic_mastery && analytics.topic_mastery.length > 0 && (
              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h5" sx={{ fontWeight: "bold", mb: 2 }}>
                      🎓 Topic Mastery
                    </Typography>
                    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1 }}>
                      {analytics.topic_mastery.map((t, idx) => (
                        <Chip
                          key={idx}
                          label={`${t.topic}: ${(t.mastery * 100).toFixed(0)}%`}
                          color={t.mastery > 0.7 ? "success" : t.mastery > 0.4 ? "warning" : "error"}
                          sx={{ fontWeight: "bold" }}
                        />
                      ))}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            )}
          </Grid>
        </>
      )}
    </Container>
  );
}
