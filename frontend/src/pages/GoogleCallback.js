import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
    Box, 
    Container, 
    Paper, 
    Typography, 
    CircularProgress, 
    Button, 
    useTheme,
    alpha
} from '@mui/material';
import { 
    CheckCircle as CheckCircleIcon, 
    Error as ErrorIcon 
} from '@mui/icons-material';

export default function GoogleCallback() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { fetchCurrentUser } = useAuth();
  const [status, setStatus] = useState('processing'); // processing, success, error
  const theme = useTheme();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Check if there's an error from Google
        const error = searchParams.get('error');
        if (error) {
          setStatus('error');
          setTimeout(() => navigate('/login'), 3000);
          return;
        }

        // Get JWT tokens from URL params
        const accessToken = searchParams.get('access_token');
        const refreshToken = searchParams.get('refresh_token');

        if (accessToken && refreshToken) {
          // Store tokens in localStorage
          localStorage.setItem('access_token', accessToken);
          localStorage.setItem('refresh_token', refreshToken);
          
          // Fetch the current user to update auth state
          await fetchCurrentUser();
          setStatus('success');
          
          // Redirect to dashboard page after 2 seconds
          setTimeout(() => navigate('/dashboard'), 2000);
        } else {
          setStatus('error');
          setTimeout(() => navigate('/login'), 3000);
        }
      } catch (err) {
        console.error('OAuth callback error:', err);
        setStatus('error');
        setTimeout(() => navigate('/login'), 3000);
      }
    };

    handleCallback();
  }, [searchParams, navigate, fetchCurrentUser]);

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
                 minHeight: 400,
                 display: 'flex',
                 flexDirection: 'column',
                 alignItems: 'center',
                 justifyContent: 'center'
             }}
        >
            {status === 'processing' && (
                <>
                    <CircularProgress size={64} thickness={4} sx={{ mb: 4, color: 'primary.main' }} />
                    <Typography variant="h5" fontWeight={700} gutterBottom>
                        Finalizing Setup
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Please wait while we verify your credentials...
                    </Typography>
                </>
            )}

            {status === 'success' && (
                <>
                    <Box sx={{ 
                        mb: 4, 
                        p: 2, 
                        borderRadius: '50%', 
                        bgcolor: alpha(theme.palette.success.main, 0.1),
                        color: 'success.main',
                        display: 'flex'
                    }}>
                        <CheckCircleIcon sx={{ fontSize: 64 }} />
                    </Box>
                    <Typography variant="h4" fontWeight={800} gutterBottom color="success.main">
                        Success!
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                        Welcome to LearnFlow
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                        Redirecting you to the dashboard...
                    </Typography>
                </>
            )}

            {status === 'error' && (
                <>
                    <Box sx={{ 
                        mb: 4, 
                        p: 2, 
                        borderRadius: '50%', 
                        bgcolor: alpha(theme.palette.error.main, 0.1),
                        color: 'error.main',
                        display: 'flex'
                    }}>
                        <ErrorIcon sx={{ fontSize: 64 }} />
                    </Box>
                    <Typography variant="h5" fontWeight={700} gutterBottom color="error.main">
                        Connection Failed
                    </Typography>
                    <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                        We couldn't verify your Google account. Please try again.
                    </Typography>
                    <Button 
                        variant="contained" 
                        onClick={() => navigate('/login')}
                        sx={{ borderRadius: 2 }}
                    >
                        Return to Login
                    </Button>
                </>
            )}
        </Paper>
      </Container>
    </Box>
  );
}
