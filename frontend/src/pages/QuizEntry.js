import React, { useState } from "react";
import {
  TextField,
  Button,
  Container,
  Card,
  CardContent,
  Typography,
  Box,
  InputAdornment,
  CircularProgress,
  Checkbox,
  FormControlLabel,
  FormGroup,
  Chip,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import QuizIcon from "@mui/icons-material/Quiz";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import TimerIcon from "@mui/icons-material/Timer";
import AllInclusiveIcon from "@mui/icons-material/AllInclusive";

export default function QuizEntry() {
  const { api: API } = useAuth();
  const [selectedNoteIds, setSelectedNoteIds] = useState([]);
  const [numQuestions, setNumQuestions] = useState(10);
  const [timerDuration, setTimerDuration] = useState(30); // seconds; 0 = no timer
  const [loading, setLoading] = useState(false);
  const [lectures, setLectures] = useState([]);
  const [fetchingLectures, setFetchingLectures] = useState(true);
  const navigate = useNavigate();

  const TIMER_OPTIONS = [
    { label: '15s', value: 15 },
    { label: '30s', value: 30 },
    { label: '45s', value: 45 },
    { label: '60s', value: 60 },
    { label: '90s', value: 90 },
    { label: '∞', value: 0 },
  ];

  // Fetch lectures on mount
  React.useEffect(() => {
    const fetchLectures = async () => {
      try {
        setFetchingLectures(true);
        console.log('Fetching lectures...');
        const response = await API.get('lectures/');
        console.log('Lectures API response:', response);
        console.log('Response data:', response.data);
        
        // Handle both array and object responses
        const lectureData = Array.isArray(response.data) ? response.data : response.data.results || [];
        console.log('Processed lecture data:', lectureData);
        console.log('Number of lectures:', lectureData.length);
        
        setLectures(lectureData);
      } catch (error) {
        console.error('Failed to fetch lectures:', error);
        console.error('Error details:', error.response);
        setLectures([]);
      } finally {
        setFetchingLectures(false);
      }
    };
    fetchLectures();
  }, [API]);

  const handleToggleLecture = (lectureId) => {
    setSelectedNoteIds(prev => {
      if (prev.includes(lectureId)) {
        return prev.filter(id => id !== lectureId);
      } else {
        return [...prev, lectureId];
      }
    });
  };

  const startQuiz = () => {
    if (selectedNoteIds.length === 0) return;
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      const noteIdsParam = selectedNoteIds.join(',');
      navigate(`/quiz-mode?noteIds=${noteIdsParam}&n=${numQuestions}`, {
        state: { timerDuration },
      });
    }, 500);
  };

  return (
    <Container maxWidth="md" sx={{ mt: 5, mb: 5, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography
            variant="overline"
            sx={{ 
                color: 'primary.main', 
                fontWeight: 700, 
                letterSpacing: 2,
                mb: 1,
                display: 'block'
            }}
        >
            Practice Arena
        </Typography>
        <Typography
            variant="h3"
            gutterBottom
            sx={{
            fontWeight: 900,
            letterSpacing: '-0.02em',
            mb: 2,
            background: "linear-gradient(135deg, #137fec 0%, #10b981 100%)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            }}
        >
            Test Your Knowledge
        </Typography>
        <Typography
            variant="body1"
            color="text.secondary"
            sx={{ maxWidth: 600, mx: 'auto', fontSize: '1.1rem' }}
        >
            Challenge yourself with AI-generated quizzes based on your lecture notes.
        </Typography>
      </Box>

      <Card
        sx={{
          p: 4,
          width: '100%',
          maxWidth: 600,
          background: "background.paper", // Use theme background
          boxShadow: "0 20px 40px rgba(0,0,0,0.2)",
          borderRadius: 4,
          border: "1px solid",
          borderColor: "divider",
        }}
      >
        <CardContent sx={{ p: 0 }}>
          <Stack spacing={4}>
            <Box>
              <Typography
                variant="subtitle2"
                fontWeight="700"
                color="text.secondary"
                sx={{ mb: 1.5, ml: 1, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.05em' }}
              >
                Select Lectures ({selectedNoteIds.length} selected)
              </Typography>
              <Box sx={{ 
                maxHeight: 300, 
                overflowY: 'auto', 
                border: '1px solid', 
                borderColor: 'divider',
                borderRadius: 3,
                p: 2
              }}>
                {fetchingLectures ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                    <CircularProgress size={24} />
                  </Box>
                ) : lectures.length > 0 ? (
                  <FormGroup>
                    {lectures.map((lecture) => (
                      <FormControlLabel
                        key={lecture.id}
                        control={
                          <Checkbox
                            checked={selectedNoteIds.includes(lecture.id)}
                            onChange={() => handleToggleLecture(lecture.id)}
                          />
                        }
                        label={lecture.title || `Lecture ${lecture.id}`}
                        sx={{ mb: 1 }}
                      />
                    ))}
                  </FormGroup>
                ) : (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
                    No lectures available. Please upload some lectures first.
                  </Typography>
                )}
              </Box>
              {selectedNoteIds.length > 0 && (
                <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {selectedNoteIds.map(id => {
                    const lecture = lectures.find(l => l.id === id);
                    return (
                      <Chip
                        key={id}
                        label={lecture?.title || `Lecture ${id}`}
                        onDelete={() => handleToggleLecture(id)}
                        color="primary"
                        variant="outlined"
                      />
                    );
                  })}
                </Box>
              )}
            </Box>

            <Box>
              <Typography
                variant="subtitle2"
                fontWeight="700"
                color="text.secondary"
                sx={{ mb: 1.5, ml: 1, textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.05em' }}
              >
                Configuration
              </Typography>
              <TextField
                type="number"
                label="Number of Questions"
                value={numQuestions}
                onChange={(e) => setNumQuestions(e.target.value)}
                fullWidth
                sx={{
                  mb: 3,
                  "& .MuiOutlinedInput-root": { height: '56px', borderRadius: 3 },
                }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <QuizIcon color="primary" />
                    </InputAdornment>
                  ),
                  inputProps: { min: 1, max: 50 },
                }}
              />

              {/* Timer selector */}
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5, ml: 1 }}>
                  <TimerIcon sx={{ fontSize: 16, color: 'text.secondary' }} />
                  <Typography variant="subtitle2" fontWeight={700} color="text.secondary"
                    sx={{ textTransform: 'uppercase', fontSize: '0.75rem', letterSpacing: '0.05em' }}
                  >
                    Time Per Question
                  </Typography>
                  {timerDuration === 0 && (
                    <Chip label="No Timer" size="small" color="default"
                      sx={{ fontWeight: 700, fontSize: '0.7rem', height: 20 }} />
                  )}
                </Box>
                <ToggleButtonGroup
                  value={timerDuration}
                  exclusive
                  onChange={(_, val) => { if (val !== null) setTimerDuration(val); }}
                  fullWidth
                  size="small"
                  sx={{
                    '& .MuiToggleButton-root': {
                      fontWeight: 700, fontSize: '0.82rem', py: 1.2, borderRadius: '10px !important',
                      border: '1px solid', borderColor: 'divider',
                      flex: 1,
                    },
                    '& .Mui-selected': {
                      background: 'linear-gradient(135deg, #137fec 0%, #10b981 100%) !important',
                      color: 'white !important',
                      borderColor: 'transparent !important',
                    },
                    gap: 0.75,
                  }}
                >
                  {TIMER_OPTIONS.map((opt) => (
                    <ToggleButton key={opt.value} value={opt.value}>
                      {opt.value === 0 ? <AllInclusiveIcon sx={{ fontSize: 16 }} /> : opt.label}
                    </ToggleButton>
                  ))}
                </ToggleButtonGroup>
                <Typography variant="caption" color="text.disabled" sx={{ mt: 0.75, ml: 1, display: 'block' }}>
                  {timerDuration === 0 ? 'Unlimited time — answer at your own pace' : `${timerDuration} seconds per question — auto-submits on timeout`}
                </Typography>
              </Box>
            </Box>

            <Button
              variant="contained"
              fullWidth
              size="large"
              onClick={startQuiz}
              disabled={loading || selectedNoteIds.length === 0 || !numQuestions}
              endIcon={
                loading ? (
                  <CircularProgress size={20} color="inherit" />
                ) : (
                  <ArrowForwardIcon />
                )
              }
              sx={{
                height: "56px",
                fontSize: "1.1rem",
                borderRadius: 3,
                mt: 2,
                fontWeight: 700,
                boxShadow: '0 8px 20px -4px rgba(19, 127, 236, 0.4)'
              }}
            >
              Start Quiz Session
            </Button>
          </Stack>
        </CardContent>
      </Card>
    </Container>
  );
}
