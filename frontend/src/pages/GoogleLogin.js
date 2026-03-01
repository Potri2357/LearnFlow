import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    Box, 
    Typography, 
    Button, 
    Container, 
    Paper, 
    useTheme, 
    alpha,
    CircularProgress,
    Stack
} from '@mui/material';
import { 
    Google as GoogleIcon, 
    ArrowForward as ArrowForwardIcon,
    Security as SecurityIcon,
    NavigateNext as NavigateNextIcon
} from '@mui/icons-material';

export default function GoogleLogin() {
  const navigate = useNavigate();
  const theme = useTheme();

  useEffect(() => {
    // Auto-redirect to Google OAuth after 2 seconds
    const timer = setTimeout(() => {
      window.location.href = 'http://localhost:8000/accounts/google/login/';
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const handleManualRedirect = () => {
    window.location.href = 'http://localhost:8000/accounts/google/login/';
  };

  return (
    <Box sx={{ 
        minHeight: '100vh', 
        bgcolor: 'background.default', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        background: `radial-gradient(circle at 50% 50%, ${alpha(theme.palette.primary.main, 0.1)} 0%, ${theme.palette.background.default} 70%)`
    }}>
      <Container maxWidth="xs">
        <Paper 
            elevation={24}
            sx={{ 
                p: 5, 
                borderRadius: 4, 
                bgcolor: 'background.paper',
                border: '1px solid',
                borderColor: 'divider',
                textAlign: 'center',
                position: 'relative',
                overflow: 'hidden'
            }}
        >
            {/* Top decorative line */}
            <Box sx={{ 
                position: 'absolute', 
                top: 0, 
                left: 0, 
                right: 0, 
                height: 4, 
                background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || '#667eea'})` 
            }} />

            {/* Logo */}
            <Box sx={{ mb: 4, display: 'inline-flex', p: 2, borderRadius: '50%', bgcolor: alpha(theme.palette.primary.main, 0.1) }}>
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                    <rect width="40" height="40" rx="10" fill={theme.palette.primary.main} />
                    <path d="M20 10L26.6 16.6L20 23.3L13.3 16.6L20 10Z" fill="white" opacity="0.9" />
                    <path d="M20 16.6L26.6 23.3L20 30L13.3 23.3L20 16.6Z" fill="white" opacity="0.7" />
                </svg>
            </Box>

            <Typography variant="h4" fontWeight={800} gutterBottom>
                LearnFlow
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 5 }}>
                Secure Login Integration
            </Typography>

            {/* Spinner / Status */}
            <Box sx={{ mb: 5, position: 'relative', height: 60, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CircularProgress size={60} thickness={4} sx={{ color: 'primary.main', position: 'absolute' }} />
                <GoogleIcon sx={{ fontSize: 24, color: 'text.primary' }} />
            </Box>

            <Typography variant="h6" fontWeight={600} gutterBottom>
                Connecting to Google
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
                You will be redirected to the secure login page momentarily...
            </Typography>


            <Stack spacing={2}>
                 <Button 
                    variant="outlined" 
                    fullWidth 
                    onClick={handleManualRedirect}
                    endIcon={<NavigateNextIcon />}
                    sx={{ 
                        py: 1.5, 
                        borderRadius: 2,
                        textTransform: 'none',
                        fontWeight: 600
                    }}
                >
                    Continue Manually
                </Button>
                
                <Button 
                    color="inherit" 
                    onClick={() => navigate('/login')}
                    sx={{ textTransform: 'none', color: 'text.secondary', fontSize: '0.85rem' }}
                >
                    Cancel and return to login
                </Button>
            </Stack>

            <Box sx={{ mt: 4, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, color: 'text.disabled' }}>
                <SecurityIcon fontSize="small" />
                <Typography variant="caption" fontWeight={500}>
                    256-bit Secure Encryption
                </Typography>
            </Box>

        </Paper>
      </Container>
    </Box>
  );
}
