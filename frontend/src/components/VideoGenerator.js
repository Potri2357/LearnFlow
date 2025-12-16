
import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  CircularProgress, 
  Typography, 
  Card, 
  CardMedia,
  IconButton,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper
} from '@mui/material';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import ImageSearchIcon from '@mui/icons-material/ImageSearch';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import { useAuth } from '../context/AuthContext';

const VideoGenerator = ({ questionId, text }) => {
  const { api } = useAuth();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [activeStep, setActiveStep] = useState(0);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = questionId ? { question_id: questionId } : { text: text };
      const res = await api.post('/video/generate/', payload);
      setData(res.data);
      if (res.data.script && res.data.script.scenes) {
          setActiveStep(0);
      }
    } catch (err) {
      console.error("Video generation failed:", err);
      setError("Failed to generate video. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  if (!data && !loading) {
    return (
      <Button 
        variant="contained" 
        color="secondary" 
        startIcon={<PlayCircleOutlineIcon />}
        onClick={handleGenerate}
        sx={{ mt: 2, borderRadius: '20px', textTransform: 'none' }}
      >
        Generate Video Explanation 🎬
      </Button>
    );
  }

  if (loading) {
    return (
      <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <CircularProgress size={20} />
        <Typography variant="body2" color="text.secondary">
          AI is writing the script & drawing diagrams... (approx 20s)
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ mt: 2 }}>
        <Typography color="error" variant="body2">{error}</Typography>
        <Button size="small" onClick={handleGenerate}>Retry</Button>
      </Box>
    );
  }

  const scenes = data.script.scenes || [];
  const videoUrl = data.video_url ? `http://127.0.0.1:8000${data.video_url}` : null;

  return (
    <Box sx={{ mt: 3, p: 2, border: '1px solid #e0e0e0', borderRadius: '12px', bgcolor: '#fafafa' }}>
      <Typography variant="h6" sx={{ mb: 2, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
        🎬 AI Video Explanation
      </Typography>

      {/* Video Player Priority */}
      {videoUrl && (
        <Card sx={{ mb: 3, overflow: 'hidden', borderRadius: '12px', boxShadow: 3 }}>
            <CardMedia
                component="video"
                controls
                src={videoUrl}
                sx={{ width: '100%', maxHeight: 500, bgcolor: 'black' }}
            />
            {data.video_error && (
                 <Typography color="error" variant="caption" sx={{ p: 1, display: 'block' }}>
                    Note: {data.video_error}
                 </Typography>
            )}
        </Card>
      )}

      {/* Fallback or Transcript View */}
      {(!videoUrl || data.force_transcript) && (
          <Stepper activeStep={activeStep} orientation="vertical">
            {scenes.map((scene, index) => {
              // Find assets for this scene
              const diagram = data.assets.find(a => a.scene === index && a.type === 'image');
              
              return (
                <Step key={index} expanded={true}>
                  <StepLabel 
                    optional={<Typography variant="caption">{scene.duration}s</Typography>}
                  >
                    {scene.title_text || (index === 0 ? "Intro" : (index === scenes.length - 1 ? "Hint" : "Core Concept"))}
                  </StepLabel>
                  <StepContent>
                    <Paper elevation={0} sx={{ p: 2, bgcolor: 'white', border: '1px solid #eee' }}>
                        <Typography variant="body1" sx={{ fontStyle: 'italic', mb: 2, color: '#555' }}>
                            " {scene.narration_text} "
                        </Typography>
                        
                        {diagram && (
                            <Box sx={{ my: 2, textAlign: 'center' }}>
                                <img 
                                    src={diagram.url} 
                                    alt="Diagram" 
                                    style={{ maxWidth: '100%', borderRadius: '8px', border: '1px solid #ddd' }} 
                                />
                                <Typography variant="caption" display="block" color="text.secondary">
                                    AI Generated Diagram (Kroki)
                                </Typography>
                            </Box>
                        )}
                        
                        {!videoUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Button
                                    variant="outlined"
                                    onClick={handleNext}
                                    disabled={index === scenes.length - 1}
                                    sx={{ mt: 1, mr: 1 }}
                                    size="small"
                                >
                                    Next
                                </Button>
                                <Button
                                    disabled={index === 0}
                                    onClick={handleBack}
                                    sx={{ mt: 1, mr: 1 }}
                                    size="small"
                                >
                                    Back
                                </Button>
                            </Box>
                        )}
                    </Paper>
                  </StepContent>
                </Step>
              );
            })}
          </Stepper>
      )}
    </Box>
  );
};

export default VideoGenerator;
