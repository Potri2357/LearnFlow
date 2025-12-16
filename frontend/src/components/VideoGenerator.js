import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  CircularProgress,
  LinearProgress,
  Typography, 
  Card, 
  CardMedia,
  IconButton,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Collapse,
  Alert
} from '@mui/material';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';
import SettingsIcon from '@mui/icons-material/Settings';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { useAuth } from '../context/AuthContext';

const VideoGenerator = ({ questionId, text }) => {
  const { api } = useAuth();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [activeStep, setActiveStep] = useState(0);
  
  // New Props / Options
  const [visualStyle, setVisualStyle] = useState("cinematic");
  const [showSettings, setShowSettings] = useState(false);
  const [progressPercent, setProgressPercent] = useState(0);

  // Loading Messages Rotation
  const loadingMessages = [
    "Analyzing topic & writing script...",
    "Generating diagrams & visual prompts...",
    "Fetching AI visuals & illustrations...",
    "Synthesizing voiceovers (Host & Expert)...",
    "Assembling final video..."
  ];

  useEffect(() => {
    let interval;
    if (loading) {
      setProgressPercent(0);
      interval = setInterval(() => {
        setProgressPercent((prev) => {
          if (prev >= 95) return 95; // Stall at 95% until done
          return prev + 2; // Increment 2% every ~800ms -> ~40s total
        });
      }, 800);
    }
    return () => clearInterval(interval);
  }, [loading]);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = questionId 
        ? { question_id: questionId, style: visualStyle } 
        : { text: text, style: visualStyle };
      
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

  if (loading) {
    return (
      <Box sx={{ mt: 3, textAlign: 'center', p: 4, bgcolor: '#f8f9fa', borderRadius: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 600, color: '#333', mb: 2 }}>
          Creating Your Explainer Video
        </Typography>
        
        <Box sx={{ width: '100%', mb: 2 }}>
            <LinearProgress variant="determinate" value={progressPercent} sx={{ height: 10, borderRadius: 5, bgcolor: '#e0e0e0', '& .MuiLinearProgress-bar': { borderRadius: 5, background: 'linear-gradient(90deg, #6200ea, #b388ff)' } }} />
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ fontWeight: 500 }}>
          {loadingMessages[Math.floor((progressPercent / 100) * loadingMessages.length)] || "Finalizing..."} ({progressPercent}%)
        </Typography>
        
        <Typography variant="caption" display="block" sx={{ mt: 2, color: '#666', fontStyle: 'italic' }}>
          "Speed depends on complexity. Generating visuals and voiceovers in parallel..."
        </Typography>
      </Box>
    );
  }

  // Not generated state
  if (!data) {
     return (
       <Box>
         <Button 
            variant="contained" 
            color="primary" // Changed to primary for better visibility
            startIcon={<AutoAwesomeIcon />}
            onClick={handleGenerate}
            sx={{ 
                mt: 2, 
                borderRadius: '20px', 
                textTransform: 'none',
                background: 'linear-gradient(45deg, #6200ea 30%, #b388ff 90%)',
                boxShadow: '0 3px 5px 2px rgba(100, 105, 255, .3)',
                fontWeight: 'bold',
                px: 3
            }}
          >
            Generate Deep Dive Video 🎬
          </Button>

          {/* Simple Settings Toggle */}
          <IconButton 
            onClick={() => setShowSettings(!showSettings)} 
            size="small" 
            sx={{ mt: 2, ml: 1, border: '1px solid #eee' }}
            title="Video Settings"
          >
            <SettingsIcon fontSize="small" />
          </IconButton>

          <Collapse in={showSettings}>
             <Box sx={{ mt: 2, p: 2, bgcolor: '#f5f5f5', borderRadius: 2, border: '1px solid #eee' }}>
                <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>Video Settings</Typography>
                
                <FormControl size="small" fullWidth sx={{ bgcolor: 'white' }}>
                    <InputLabel>Visual Style</InputLabel>
                    <Select
                        value={visualStyle}
                        label="Visual Style"
                        onChange={(e) => setVisualStyle(e.target.value)}
                    >
                        <MenuItem value="cinematic">Star Wars / Cinematic (Default)</MenuItem>
                        <MenuItem value="cartoon">Cartoon / Vibrant</MenuItem>
                        <MenuItem value="sketch">Blueprint / Sketch</MenuItem>
                    </Select>
                </FormControl>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                    Affects the style of AI background images & tone.
                </Typography>
             </Box>
          </Collapse>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
                {error} <Button size="small" onClick={handleGenerate}>Retry</Button>
            </Alert>
          )}
       </Box>
     );
  }

  const scenes = data.script.scenes || [];
  const videoUrl = data.video_url ? `http://127.0.0.1:8000${data.video_url}` : null;

  return (
    <Box sx={{ mt: 3, p: 3, border: '1px solid #e0e0e0', borderRadius: '16px', bgcolor: '#fff', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', gap: 1 }}>
          🎬 {data.script.title || "Deep Dive Explanation"}
        </Typography>
        <Button size="small" onClick={() => setData(null)}>New Video</Button>
      </Box>

      {/* Video Player */}
      {videoUrl && (
        <Card sx={{ mb: 3, overflow: 'hidden', borderRadius: '12px', boxShadow: 4, bgcolor: 'black' }}>
            <CardMedia
                component="video"
                controls
                autoPlay
                src={videoUrl}
                sx={{ width: '100%', maxHeight: 500 }}
            />
            {data.video_error && (
                 <Typography color="error" variant="caption" sx={{ p: 1, display: 'block' }}>
                    Warning: {data.video_error}. Showing transcript mode.
                 </Typography>
            )}
        </Card>
      )}

      {/* Transcript / Slideshow View */}
      {(!videoUrl || data.force_transcript) && (
          <Stepper activeStep={activeStep} orientation="vertical">
            {scenes.map((scene, index) => {
              // Find assets for this scene
              const asset = data.assets && data.assets.find(a => a.scene === index);
              
              return (
                <Step key={index} expanded={true}>
                  <StepLabel 
                    StepIconProps={{ sx: { color: scene.speaker === 'Expert' ? '#009688' : '#6200ea' } }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                         <Typography variant="subtitle2" sx={{ fontWeight: 600, color: scene.speaker === 'Expert' ? '#00796b' : '#4527a0' }}>
                            {scene.speaker.toUpperCase()}
                         </Typography>
                         <Typography variant="caption" color="text.secondary">{scene.duration}s</Typography>
                    </Box>
                  </StepLabel>
                  <StepContent>
                    <Paper elevation={0} sx={{ p: 2, bgcolor: scene.speaker==='Expert' ? '#e0f2f1' : '#f3e5f5', borderRadius: 2 }}>
                        <Typography variant="body1" sx={{ fontStyle: 'italic', mb: 2, color: '#333' }}>
                            "{scene.text}"
                        </Typography>
                        
                        {asset && asset.type === 'image' && (
                            <Box sx={{ my: 2, textAlign: 'center' }}>
                                <img 
                                    src={`http://127.0.0.1:8000${asset.url}`} 
                                    alt="Visual" 
                                    style={{ 
                                        maxWidth: '100%', 
                                        maxHeight: '300px', 
                                        borderRadius: '8px', 
                                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)' 
                                    }} 
                                />
                                <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
                                    {asset.url.includes("diagram") ? "📊 Generated Diagram" : "🎨 AI Interaction Visual"}
                                </Typography>
                            </Box>
                        )}
                        
                        {!videoUrl && (
                            <Box sx={{ mb: 2 }}>
                                <Button
                                    variant="contained"
                                    onClick={handleNext}
                                    disabled={index === scenes.length - 1}
                                    sx={{ mt: 1, mr: 1, bgcolor: scene.speaker==='Expert'?'#009688':'#6200ea' }}
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
