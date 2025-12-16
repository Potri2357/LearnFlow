import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Stack,
  Chip,
} from '@mui/material';
import {
  AutoStories as BooksIcon,
  Psychology as BrainIcon,
  Quiz as QuizIcon,
  TrendingUp as TrendingIcon,
  Lightbulb as LightbulbIcon,
  School as SchoolIcon,
  ArrowForward as ArrowIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const FeatureCard = ({ icon, title, description }) => (
  <motion.div
    whileHover={{ y: -8, scale: 1.02 }}
    transition={{ duration: 0.3 }}
    style={{ height: '100%' }}
  >
    <Card
      sx={{
        height: '100%',
        background: 'rgba(255, 255, 255, 0.9)',
        backdropFilter: 'blur(10px)',
        borderRadius: 4,
        border: '1px solid rgba(102, 126, 234, 0.1)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
        transition: 'all 0.3s ease',
        '&:hover': {
          boxShadow: '0 12px 48px rgba(102, 126, 234, 0.15)',
          border: '1px solid rgba(102, 126, 234, 0.3)',
        },
      }}
    >
      <CardContent sx={{ p: 4, textAlign: 'center' }}>
        <Box
          sx={{
            width: 80,
            height: 80,
            borderRadius: '20px',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 24px',
            boxShadow: '0 8px 24px rgba(102, 126, 234, 0.3)',
          }}
        >
          {icon}
        </Box>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 2, color: '#1e293b' }}>
          {title}
        </Typography>
        <Typography variant="body2" sx={{ color: '#64748b', lineHeight: 1.7 }}>
          {description}
        </Typography>
      </CardContent>
    </Card>
  </motion.div>
);

const StepCard = ({ number, title, description }) => (
  <Box sx={{ textAlign: 'center' }}>
    <Box
      sx={{
        width: 60,
        height: 60,
        borderRadius: '50%',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        color: 'white',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        margin: '0 auto 16px',
        fontSize: '24px',
        fontWeight: 700,
        boxShadow: '0 4px 16px rgba(102, 126, 234, 0.4)',
      }}
    >
      {number}
    </Box>
    <Typography variant="h6" sx={{ fontWeight: 600, mb: 1, color: '#1e293b' }}>
      {title}
    </Typography>
    <Typography variant="body2" sx={{ color: '#64748b' }}>
      {description}
    </Typography>
  </Box>
);

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <Box sx={{ minHeight: '100vh', background: '#f8fafc' }}>
      {/* Hero Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          pt: { xs: 8, md: 12 },
          pb: { xs: 12, md: 16 },
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {/* Decorative circles */}
        <Box
          sx={{
            position: 'absolute',
            top: -100,
            right: -100,
            width: 400,
            height: 400,
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.1)',
            filter: 'blur(60px)',
          }}
        />
        <Box
          sx={{
            position: 'absolute',
            bottom: -150,
            left: -150,
            width: 500,
            height: 500,
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.1)',
            filter: 'blur(80px)',
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={6} alignItems="center" justifyContent="center">
            <Grid item xs={12} md={10} sx={{ textAlign: 'center' }}>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <Chip
                  label="AI-Powered Learning Platform"
                  sx={{
                    mb: 3,
                    bgcolor: 'rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    fontWeight: 600,
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                  }}
                />
                <Typography
                  variant="h2"
                  sx={{
                    fontWeight: 800,
                    color: 'white',
                    mb: 3,
                    fontSize: { xs: '2.5rem', md: '3.5rem' },
                    lineHeight: 1.2,
                  }}
                >
                  Master Any Subject with{' '}
                  <Box component="span" sx={{ color: '#fbbf24' }}>
                    Adaptive Learning
                  </Box>
                </Typography>
                <Typography
                  variant="h6"
                  sx={{
                    color: 'rgba(255, 255, 255, 0.9)',
                    mb: 4,
                    lineHeight: 1.6,
                    fontWeight: 400,
                  }}
                >
                  Transform your lecture notes into personalized quizzes, flashcards, and study plans.
                  Our AI adapts to your learning pace and identifies weak areas automatically.
                </Typography>
                <Stack 
                  direction={{ xs: 'column', sm: 'row' }} 
                  spacing={2} 
                  justifyContent="center"
                >
                  <Button
                    variant="contained"
                    size="large"
                    onClick={() => navigate('/register')}
                    endIcon={<ArrowIcon />}
                    sx={{
                      bgcolor: 'rgba(255, 255, 255, 0.2)',
                      color: 'white',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(255, 255, 255, 0.3)',
                      px: 4,
                      py: 1.5,
                      fontSize: '1.1rem',
                      fontWeight: 700,
                      borderRadius: 3,
                      boxShadow: '0 8px 24px rgba(0, 0, 0, 0.15)',
                      '&:hover': {
                        bgcolor: 'rgba(255, 255, 255, 0.3)',
                        transform: 'translateY(-2px)',
                        boxShadow: '0 12px 32px rgba(0, 0, 0, 0.2)',
                      },
                    }}
                  >
                    Get Started Free
                  </Button>
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => navigate('/login')}
                    sx={{
                      borderColor: 'white',
                      color: 'white',
                      px: 4,
                      py: 1.5,
                      fontSize: '1.1rem',
                      fontWeight: 700,
                      borderRadius: 3,
                      borderWidth: 2,
                      '&:hover': {
                        borderWidth: 2,
                        borderColor: 'white',
                        bgcolor: 'rgba(255, 255, 255, 0.1)',
                      },
                    }}
                  >
                    Sign In
                  </Button>
                </Stack>
              </motion.div>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 12 } }}>
        <Box sx={{ textAlign: 'center', mb: 8 }}>
          <Typography
            variant="h3"
            sx={{
              fontWeight: 800,
              mb: 2,
              background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Powerful Features
          </Typography>
          <Typography variant="h6" sx={{ color: '#64748b', maxWidth: 600, mx: 'auto' }}>
            Everything you need to accelerate your learning journey
          </Typography>
        </Box>

        <Grid container spacing={4} alignItems="stretch" justifyContent="center">
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<BrainIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="AI-Powered Questions"
              description="Automatically generate high-quality MCQs from your lecture notes using advanced AI technology."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<TrendingIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="Adaptive Learning"
              description="Our algorithm adapts to your performance, focusing on topics where you need the most practice."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<QuizIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="Smart Flashcards"
              description="Create and study with AI-generated flashcards that help reinforce key concepts effectively."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<LightbulbIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="Personalized Study Plans"
              description="Get customized study recommendations based on your strengths and weaknesses."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<SchoolIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="Progress Analytics"
              description="Track your learning progress with detailed analytics and performance insights."
            />
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <FeatureCard
              icon={<BooksIcon sx={{ fontSize: 40, color: 'white' }} />}
              title="Multi-Format Support"
              description="Upload lecture notes in various formats including PDF, text, and more."
            />
          </Grid>
        </Grid>
      </Container>

      {/* How It Works Section */}
      <Box sx={{ bgcolor: '#f1f5f9', py: { xs: 8, md: 12 } }}>
        <Container maxWidth="lg">
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 800,
                mb: 2,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              How It Works
            </Typography>
            <Typography variant="h6" sx={{ color: '#64748b' }}>
              Get started in four simple steps
            </Typography>
          </Box>

          <Grid container spacing={6}>
            <Grid item xs={12} sm={6} md={3}>
              <StepCard
                number="1"
                title="Upload Notes"
                description="Upload your lecture notes or study materials in any format"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StepCard
                number="2"
                title="Generate Content"
                description="AI creates questions, flashcards, and study materials instantly"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StepCard
                number="3"
                title="Take Quizzes"
                description="Practice with adaptive quizzes that match your skill level"
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <StepCard
                number="4"
                title="Track Progress"
                description="Monitor your improvement and focus on weak areas"
              />
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* CTA Section */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          py: { xs: 8, md: 10 },
          textAlign: 'center',
        }}
      >
        <Container maxWidth="md">
          <Typography
            variant="h3"
            sx={{
              fontWeight: 800,
              color: 'white',
              mb: 3,
              fontSize: { xs: '2rem', md: '3rem' },
            }}
          >
            Ready to Transform Your Learning?
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.9)', mb: 4 }}>
            Join thousands of students already using LearnFlow to achieve their academic goals
          </Typography>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/register')}
            endIcon={<ArrowIcon />}
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.2)',
              color: 'white',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.3)',
              px: 5,
              py: 2,
              fontSize: '1.2rem',
              fontWeight: 700,
              borderRadius: 3,
              boxShadow: '0 8px 24px rgba(0, 0, 0, 0.2)',
              '&:hover': {
                bgcolor: 'rgba(255, 255, 255, 0.3)',
                transform: 'translateY(-2px)',
                boxShadow: '0 12px 32px rgba(0, 0, 0, 0.3)',
              },
            }}
          >
            Start Learning Now
          </Button>
        </Container>
      </Box>

      {/* Footer */}
      <Box sx={{ bgcolor: '#1e293b', py: 6 }}>
        <Container maxWidth="lg">
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" sx={{ color: 'white', fontWeight: 700, mb: 2 }}>
                LearnFlow
              </Typography>
              <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                Empowering students with AI-driven adaptive learning technology
              </Typography>
            </Grid>
            <Grid item xs={12} md={6} sx={{ textAlign: { xs: 'left', md: 'right' } }}>
              <Typography variant="body2" sx={{ color: '#94a3b8' }}>
                © 2025 LearnFlow. All rights reserved.
              </Typography>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </Box>
  );
}
