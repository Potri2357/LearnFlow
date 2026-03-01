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
  useTheme,
  alpha,
  Paper,
  IconButton
} from '@mui/material';
import {
  AutoStories as BooksIcon,
  Psychology as BrainIcon,
  Quiz as QuizIcon,
  TrendingUp as TrendingIcon,
  Lightbulb as LightbulbIcon,
  School as SchoolIcon,
  ArrowForward as ArrowIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Code as CodeIcon,
  Brightness4 as Brightness4Icon,
  Brightness7 as Brightness7Icon
} from '@mui/icons-material';
import { useColorMode } from '../context/ThemeContext';
import { motion } from 'framer-motion';

const FeatureCard = ({ icon, title, description, color = '#2563EB' }) => {
    const theme = useTheme();
    const { mode } = useColorMode();
    
    // Create lighter/darker shades based on the input color
    const gradientStart = mode === 'dark' ? alpha(color, 0.2) : alpha(color, 0.1);
    const gradientEnd = mode === 'dark' ? alpha(color, 0.4) : alpha(color, 0.2);
    const iconColor = color;
    
    return (
      <motion.div
        whileHover={{ y: -5, scale: 1.02 }}
        transition={{ duration: 0.3 }}
        style={{ height: '100%', width: '100%' }}
      >
        <Paper
          elevation={0}
          sx={{
            height: 260, // Fixed height for perfect uniformity
            width: '100%',
            maxWidth: 280, // Restrict max width for consistent "card" look
            mx: 'auto', // Center in grid cell
            background: mode === 'dark' ? alpha(theme.palette.background.paper, 0.6) : 'rgba(255, 255, 255, 0.9)',
            backdropFilter: 'blur(20px)',
            borderRadius: 3,
            border: '1px solid',
            borderColor: mode === 'dark' ? 'divider' : alpha(color, 0.2), 
            overflow: 'hidden',
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.3s ease',
            '&:hover': {
              boxShadow: `0 10px 40px ${alpha(color, 0.25)}`,
              borderColor: alpha(color, 0.5),
              transform: 'translateY(-2px)'
            }
          }}
        >
          {/* Top colored accent line */}
          <Box 
            sx={{ 
                position: 'absolute', 
                top: 0, 
                left: 0, 
                right: 0, 
                height: 4, 
                background: `linear-gradient(90deg, ${alpha(color, 0.8)}, ${color})`,
                opacity: 0.9,
            }} 
          />
          
          <CardContent sx={{ p: 2.5, textAlign: 'center', width: '100%' }}>
            <Box
              sx={{
                width: 56,
                height: 56,
                borderRadius: '16px',
                background: `linear-gradient(135deg, ${gradientStart} 0%, ${gradientEnd} 100%)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
                color: iconColor,
                boxShadow: `0 8px 20px ${alpha(color, 0.2)}`,
                border: `1px solid ${alpha(color, 0.1)}`,
              }}
            >
              {React.cloneElement(icon, { sx: { fontSize: 28 } })}
            </Box>
            <Typography variant="subtitle1" sx={{ fontWeight: 800, mb: 1, color: 'text.primary', fontSize: '1.05rem', lineHeight: 1.2 }}>
              {title}
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.6, fontSize: '0.875rem', px: 1 }}>
              {description}
            </Typography>
          </CardContent>
        </Paper>
      </motion.div>
    );
};

const StepCard = ({ number, title, description, color = '#3B82F6' }) => {
    const theme = useTheme();
    return (
      <Box sx={{ textAlign: 'center', position: 'relative' }}>
        <Box
          sx={{
            width: 64,
            height: 64,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${color} 0%, ${alpha(color, 0.8)} 100%)`,
            color: 'white',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 20px',
            fontSize: '24px',
            fontWeight: 800,
            boxShadow: `0 10px 20px ${alpha(color, 0.3)}`,
            border: `4px solid ${theme.palette.background.default}`,
            position: 'relative',
            zIndex: 1
          }}
        >
          {number}
        </Box>
        <Typography variant="h6" sx={{ fontWeight: 700, mb: 1.5, color: 'text.primary' }}>
          {title}
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 240, mx: 'auto' }}>
          {description}
        </Typography>
      </Box>
    );
};

const FloatingIcon = ({ icon, top, left, delay, duration = 5, color }) => {
    return (
      <Box
        component={motion.div}
        animate={{ 
          y: [0, -20, 0],
          rotate: [0, 10, -10, 0],
          opacity: [0.6, 1, 0.6]
        }}
        transition={{ 
          duration: duration, 
          repeat: Infinity, 
          repeatType: "reverse", 
          delay: delay,
          ease: "easeInOut" 
        }}
        sx={{
          position: 'absolute',
          top: top,
          left: left,
          zIndex: 0,
          color: color,
          filter: 'drop-shadow(0 4px 10px rgba(0,0,0,0.1))'
        }}
      >
        {React.cloneElement(icon, { sx: { fontSize: { xs: 40, md: 60 }, opacity: 0.8 } })}
      </Box>
    );
};

export default function LandingPage() {
  const navigate = useNavigate();
  const theme = useTheme();
  const { mode, toggleColorMode } = useColorMode();

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default', overflowX: 'hidden', position: 'relative' }}>
      {/* Theme Toggle Button */}
      <Box sx={{ position: 'fixed', top: 24, right: 24, zIndex: 10 }}>
        <IconButton 
          onClick={toggleColorMode} 
          sx={{ 
            color: mode === 'dark' ? '#F59E0B' : '#64748B',
            bgcolor: mode === 'dark' ? 'rgba(245, 158, 11, 0.1)' : 'rgba(255, 255, 255, 0.5)',
            backdropFilter: 'blur(10px)',
            width: 48,
            height: 48,
            borderRadius: '16px',
            border: '1px solid',
            borderColor: 'divider',
            transition: 'all 0.2s',
            '&:hover': { transform: 'rotate(15deg)', bgcolor: mode === 'dark' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(255, 255, 255, 0.8)' }
          }}
        >
          {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
        </IconButton>
      </Box>

      {/* Rich Animated Background */}
      <Box
        sx={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          zIndex: 0,
          pointerEvents: 'none',
          background: mode === 'dark' 
            ? `radial-gradient(circle at 50% 50%, ${alpha('#1e293b', 0.5)} 0%, ${theme.palette.background.default} 100%)`
            : `linear-gradient(135deg, #ecfdf5 0%, #eff6ff 100%)`, // Mint-Blue Tint for Light Mode
        }}
      >
        {/* Colorful Mesh Gradients for Light Mode */}
        {mode === 'light' && (
          <>
            <Box
              component={motion.div}
              animate={{ 
                x: [0, 100, 0],
                y: [0, -50, 0],
                scale: [1, 1.2, 1] 
              }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              sx={{
                position: 'absolute',
                top: '-10%',
                right: '-5%',
                width: '60vw',
                height: '60vw',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(59,130,246,0.15) 0%, rgba(59,130,246,0) 70%)',
                filter: 'blur(60px)',
              }}
            />
            <Box
              component={motion.div}
              animate={{ 
                x: [0, -70, 0],
                y: [0, 100, 0],
                scale: [1, 1.1, 1] 
              }}
              transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
              sx={{
                position: 'absolute',
                bottom: '10%',
                left: '-10%',
                width: '50vw',
                height: '50vw',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(13,148,136,0.15) 0%, rgba(13,148,136,0) 70%)',
                filter: 'blur(60px)',
              }}
            />
            <Box
              component={motion.div}
              animate={{ 
                x: [0, 50, 0],
                y: [0, 50, 0],
                opacity: [0.5, 0.8, 0.5] 
              }}
              transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
              sx={{
                position: 'absolute',
                top: '40%',
                left: '20%',
                width: '35vw',
                height: '35vw',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(52,211,153,0.15) 0%, rgba(52,211,153,0) 70%)', // Mint Green
                filter: 'blur(50px)',
              }}
            />
             <Box
              component={motion.div}
              animate={{ 
                x: [0, -40, 0],
                y: [0, -30, 0],
                opacity: [0.4, 0.7, 0.4] 
              }}
              transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
              sx={{
                position: 'absolute',
                top: '20%',
                right: '25%',
                width: '25vw',
                height: '25vw',
                borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0) 70%)', // Purple
                filter: 'blur(50px)',
              }}
            />

          </>
        )}
      </Box>

      {/* Scrollable Floating Icons Container (Scrolls with page) */}
      <Box sx={{ position: 'absolute', top: 0, left: 0, right: 0, height: '100vh', overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
        <FloatingIcon icon={<QuizIcon />} top="15%" left="15%" delay={0} color="#10B981" />
        <FloatingIcon icon={<BrainIcon />} top="25%" left="85%" delay={2} color="#8B5CF6" />
        <FloatingIcon icon={<LightbulbIcon />} top="65%" left="10%" delay={4} color="#F59E0B" />
        <FloatingIcon icon={<CodeIcon />} top="60%" left="90%" delay={1} color="#3B82F6" />
        <FloatingIcon icon={<BooksIcon />} top="85%" left="20%" delay={3} color="#F43F5E" />
      </Box>

      {/* Hero Section */}
      <Box
        sx={{
          pt: { xs: 12, md: 20 },
          pb: { xs: 12, md: 20 },
          position: 'relative',
          zIndex: 1
        }}
      >
        <Container maxWidth="lg">
          <Stack alignItems="center" spacing={4} sx={{ textAlign: 'center', maxWidth: 1000, mx: 'auto' }}>
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <Chip
                label="🚀 AI-Powered Adaptive Learning"
                sx={{
                  mb: 4,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  color: theme.palette.primary.main,
                  fontWeight: 700,
                  border: '1px solid',
                  borderColor: alpha(theme.palette.primary.main, 0.2),
                  px: 1.5,
                  height: 36,
                  fontSize: '0.9rem'
                }}
              />
              
              <Typography
                variant="h1"
                sx={{
                  fontWeight: 800,
                  color: 'text.primary',
                  mb: 3,
                  fontSize: { xs: '3rem', md: '5rem' },
                  lineHeight: 1.1,
                  letterSpacing: '-0.02em',
                  textShadow: theme.palette.mode === 'dark' ? '0 0 40px rgba(19, 127, 236, 0.1)' : 'none'
                }}
              >
                Master Any Subject with <br />
                <Box 
                  component="span" 
                  sx={{ 
                    background: mode === 'dark'
                        ? `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.info.main} 100%)`
                        : `linear-gradient(135deg, #2563eb 0%, #0d9488 100%)`, // Vibrant Blue-Teal for Light Mode
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    display: 'inline-block',
                    pb: 1,
                    filter: mode === 'light' ? 'drop-shadow(0 2px 10px rgba(37,99,235,0.2))' : 'none'
                  }}
                >
                  Intelligent Learning
                </Box>
              </Typography>
              
              <Typography
                variant="h5"
                sx={{
                  color: 'text.secondary',
                  mb: 6,
                  lineHeight: 1.6,
                  fontWeight: 400,
                  maxWidth: 800,
                  mx: 'auto',
                  fontSize: { xs: '1.1rem', md: '1.35rem' }
                }}
              >
                Transform your lecture notes into interactive quizzes, flashcards, and personalized study plans using advanced AI.
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
                    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                    px: 6,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    borderRadius: '50px',
                    boxShadow: `0 10px 30px ${alpha(theme.palette.primary.main, 0.4)}`,
                    textTransform: 'none',
                    '&:hover': {
                      background: `linear-gradient(135deg, ${theme.palette.primary.dark} 0%, ${theme.palette.primary.dark} 100%)`,
                      transform: 'translateY(-2px)',
                      boxShadow: `0 15px 35px ${alpha(theme.palette.primary.main, 0.5)}`,
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
                    borderColor: 'divider',
                    color: 'text.primary',
                    px: 6,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 700,
                    borderRadius: '50px',
                    borderWidth: 2,
                    textTransform: 'none',
                    backdropFilter: 'blur(10px)',
                    '&:hover': {
                      borderWidth: 2,
                      borderColor: theme.palette.primary.main,
                      color: theme.palette.primary.main,
                      bgcolor: alpha(theme.palette.primary.main, 0.05)
                    },
                  }}
                >
                  Sign In
                </Button>
              </Stack>
            </motion.div>
          </Stack>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: { xs: 8, md: 16 }, position: 'relative', zIndex: 1 }}>
        <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
        >
        <Box sx={{ textAlign: 'center', mb: 10 }}>
          <Typography
            variant="overline"
            sx={{
              color: 'primary.main',
              fontWeight: 700,
              letterSpacing: 1.5,
              mb: 1,
              display: 'block'
            }}
          >
            WHY LEARNFLOW?
          </Typography>
          <Typography
            variant="h2"
            sx={{
              fontWeight: 800,
              color: 'text.primary',
              mb: 2,
            }}
          >
            Supercharge Your Study Sessions
          </Typography>
          <Typography variant="h6" sx={{ color: 'text.secondary', maxWidth: 600, mx: 'auto' }}>
            Our AI analyzes your content to create the most effective learning materials for you.
          </Typography>
        </Box>

        <Grid container spacing={3} justifyContent="center">
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<BrainIcon fontSize="large" />}
              title="AI Question Generation"
              description="Automatically generate high-quality MCQs from your lecture notes with one click."
              color="#8B5CF6" // Violet
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<TrendingIcon fontSize="large" />}
              title="Adaptive Difficulty"
              description="Questions get harder as you improve, ensuring you're always challenged just right."
              color="#F43F5E" // Rose
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<QuizIcon fontSize="large" />}
              title="Smart Flashcards"
              description="Review key concepts with spaced repetition flashcards created from your notes."
              color="#10B981" // Emerald
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<LightbulbIcon fontSize="large" />}
              title="Personalized Plans"
              description="Get a daily study schedule tailored to your exam dates and weak topics."
              color="#F59E0B" // Amber
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<SchoolIcon fontSize="large" />}
              title="Performance Analytics"
              description="Visualize your progress with detailed charts and identify areas needing improvement."
              color="#3B82F6" // Blue
            />
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <FeatureCard
              icon={<BooksIcon fontSize="large" />}
              title="Multi-Format Support"
              description="Upload PDFs, paste text, or use raw notes. We handle it all seamlessly."
              color="#06B6D4" // Cyan
            />
          </Grid>
        </Grid>
        </motion.div>
      </Container>

      {/* How It Works Section */}
      <Box sx={{ bgcolor: alpha(theme.palette.background.paper, 0.4), py: { xs: 8, md: 16 }, position: 'relative', zIndex: 1 }}>
        <Container maxWidth="lg">
        <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <Box sx={{ textAlign: 'center', mb: 10 }}>
             <Typography
                variant="overline"
                sx={{
                  color: 'primary.main',
                  fontWeight: 700,
                  letterSpacing: 1.5,
                  mb: 1,
                  display: 'block'
                }}
              >
                SIMPLE WORKFLOW
              </Typography>
            <Typography
              variant="h3"
              sx={{
                fontWeight: 800,
                color: 'text.primary',
                mb: 2,
              }}
            >
              From Notes to Mastery in Minutes
            </Typography>
          </Box>

          <Box sx={{ position: 'relative' }}>
             {/* Connector Line (Desktop) */}
            <Box 
                sx={{ 
                    position: 'absolute', 
                    top: 32, 
                    left: '10%', 
                    right: '10%', 
                    height: 2, 
                    background: `linear-gradient(90deg, ${alpha(theme.palette.primary.main, 0.0)} 0%, ${alpha(theme.palette.primary.main, 0.3)} 50%, ${alpha(theme.palette.primary.main, 0.0)} 100%)`,
                    display: { xs: 'none', md: 'block' },
                    zIndex: 0
                }} 
            />

            <Grid container spacing={6}>
              <Grid item xs={12} sm={6} md={3}>
                <StepCard
                  number="1"
                  title="Upload"
                  description="Upload your PDF lectures or paste your notes directly."
                  color="#8B5CF6" // Violet
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <StepCard
                  number="2"
                  title="Process"
                  description="AI instantly analyzes the content and extracts key concepts."
                  color="#F43F5E" // Pink
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <StepCard
                  number="3"
                  title="Practice"
                  description="Take generated quizzes and review flashcards."
                  color="#F59E0B" // Amber
                />
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <StepCard
                  number="4"
                  title="Master"
                  description="Track your mastery and ace your exams."
                  color="#10B981" // Emerald
                />
              </Grid>
            </Grid>
          </Box>
          </motion.div>
        </Container>
      </Box>

      {/* Footer */}
      {/* Footer */}
      <Box sx={{ 
        bgcolor: mode === 'dark' ? 'background.paper' : '#f8fafc',
        pt: 8, 
        pb: 4, 
        borderTop: '1px solid', 
        borderColor: 'divider',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Background Decorative Mesh for Footer */}
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            opacity: 0.05,
            zIndex: 0,
            background: `radial-gradient(circle at 80% 0%, ${theme.palette.primary.main} 0%, transparent 40%),
                         radial-gradient(circle at 20% 100%, ${theme.palette.secondary.main || '#10B981'} 0%, transparent 40%)`
          }}
        />

        <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1 }}>
          <Grid container spacing={8}>
            <Grid item xs={12} md={4}>
              <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 2 }}>
                  <Box sx={{ 
                      width: 40, 
                      height: 40, 
                      borderRadius: 1.5, 
                      background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main || '#0d9488'})`, 
                      display: 'flex', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      color: 'white',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)'
                  }}>
                      <BrainIcon />
                  </Box>
                  <Typography variant="h5" sx={{ 
                    fontWeight: 800,
                    background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.secondary.main || '#0d9488'} 100%)`,
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}>
                    LearnFlow
                  </Typography>
              </Stack>
              <Typography variant="body2" sx={{ color: 'text.secondary', maxWidth: 300, mb: 3, lineHeight: 1.6 }}>
                Empowering students worldwide with state-of-the-art AI learning tools. 
                Make every study session count with LearnFlow.
              </Typography>
              <Stack direction="row" spacing={1}>
                 {[
                   { icon: 'twitter', color: '#1DA1F2' },
                   { icon: 'github', color: mode === 'dark' ? '#ffffff' : '#333' },
                   { icon: 'linkedin', color: '#0077B5' }
                 ].map((social) => (
                   <IconButton 
                    key={social.icon}
                    size="small" 
                    sx={{ 
                      border: '1px solid', 
                      borderColor: 'divider',
                      transition: 'all 0.2s',
                      '&:hover': { 
                        borderColor: social.color, 
                        color: social.color,
                        transform: 'translateY(-2px)'
                      } 
                    }}
                   >
                     <Box component="span" className={`fab fa-${social.icon}`} />
                   </IconButton>
                 ))}
              </Stack>
            </Grid>
            
            <Grid item xs={6} md={2}>
               <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
                 Product
               </Typography>
               <Stack spacing={1.5}>
                 {['Features', 'Pricing', 'Testimonials', 'FAQ'].map((item) => (
                   <Typography 
                     key={item} 
                     variant="body2" 
                     sx={{ 
                       color: 'text.secondary', 
                       cursor: 'pointer', 
                       '&:hover': { color: 'primary.main', transform: 'translateX(4px)' },
                       transition: 'all 0.2s'
                     }}
                   >
                     {item}
                   </Typography>
                 ))}
               </Stack>
            </Grid>

            <Grid item xs={6} md={2}>
               <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
                 Resources
               </Typography>
               <Stack spacing={1.5}>
                 {['Blog', 'Study Guides', 'Tutorials', 'Community'].map((item) => (
                   <Typography 
                     key={item} 
                     variant="body2" 
                     sx={{ 
                       color: 'text.secondary', 
                       cursor: 'pointer', 
                       '&:hover': { color: 'primary.main', transform: 'translateX(4px)' },
                       transition: 'all 0.2s'
                     }}
                   >
                     {item}
                   </Typography>
                 ))}
               </Stack>
            </Grid>

            <Grid item xs={6} md={2}>
               <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
                 Company
               </Typography>
               <Stack spacing={1.5}>
                 {['About Us', 'Careers', 'Privacy Policy', 'Terms of Service'].map((item) => (
                   <Typography 
                     key={item} 
                     variant="body2" 
                     sx={{ 
                       color: 'text.secondary', 
                       cursor: 'pointer', 
                       '&:hover': { color: 'primary.main', transform: 'translateX(4px)' },
                       transition: 'all 0.2s'
                     }}
                   >
                     {item}
                   </Typography>
                 ))}
               </Stack>
            </Grid>
             <Grid item xs={6} md={2}>
               <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 3, color: 'text.primary' }}>
                 Support
               </Typography>
               <Stack spacing={1.5}>
                 {['Help Center', 'Contact Us', 'Status', 'Feedback'].map((item) => (
                   <Typography 
                     key={item} 
                     variant="body2" 
                     sx={{ 
                       color: 'text.secondary', 
                       cursor: 'pointer', 
                       '&:hover': { color: 'primary.main', transform: 'translateX(4px)' },
                       transition: 'all 0.2s'
                     }}
                   >
                     {item}
                   </Typography>
                 ))}
               </Stack>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 8, pt: 4, borderTop: '1px solid', borderColor: 'divider', textAlign: 'center' }}>
            <Typography variant="body2" sx={{ color: 'text.disabled' }}>
                © {new Date().getFullYear()} LearnFlow. All rights reserved.
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}
