import React, { useState } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Fade,
  Zoom,
  Alert,
  Snackbar,
  Chip,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Settings as SettingsIcon,
  PlayArrow as ProcessIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import VideoUpload from './VideoUpload';
import BackgroundSettings from './BackgroundSettings';
import ProcessingControls from './ProcessingControls';
import DualVideoPreview from './DualVideoPreview';
import ProcessingProgress from './ProcessingProgress';
import HardwareStatus from './HardwareStatus';
import SystemAlert from './SystemAlert';
import { useVideoProcessor } from '../context/VideoProcessorContext';

const steps = [
  { label: 'Upload Video', icon: <UploadIcon /> },
  { label: 'Configure Background', icon: <SettingsIcon /> },
  { label: 'Process Video', icon: <ProcessIcon /> },
  { label: 'Download Result', icon: <DownloadIcon /> },
];

export default function VideoProcessor() {
  const {
    uploadedVideo,
    processingStatus,
    connectionError,
  } = useVideoProcessor();
  
  const [showSuccessAlert, setShowSuccessAlert] = useState(false);
  
  React.useEffect(() => {
    if (processingStatus === 'completed') {
      setShowSuccessAlert(true);
    }
  }, [processingStatus]);

  // Determine active step based on state
  const getActiveStep = () => {
    if (!uploadedVideo) return 0;
    if (processingStatus === 'idle') return 1;
    if (processingStatus === 'processing' || processingStatus === 'started') return 2;
    if (processingStatus === 'completed') return 3;
    return 1;
  };

  const activeStep = getActiveStep();

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      {/* Progress Stepper */}
      <Card
        sx={{
          mb: 4,
          background: 'rgba(26, 31, 58, 0.6)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(0, 180, 216, 0.2)',
        }}
      >
        <CardContent>
          <Stepper activeStep={activeStep} alternativeLabel>
            {steps.map((step, index) => (
              <Step key={step.label}>
                <StepLabel
                  StepIconComponent={() => (
                    <Box
                      sx={{
                        width: 40,
                        height: 40,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: index <= activeStep
                          ? 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)'
                          : 'rgba(255, 255, 255, 0.1)',
                        color: index <= activeStep ? 'white' : 'text.secondary',
                        border: index === activeStep
                          ? '2px solid #00f5ff'
                          : '2px solid transparent',
                        boxShadow: index === activeStep
                          ? '0 0 20px rgba(0, 245, 255, 0.5)'
                          : 'none',
                        transition: 'all 0.3s ease',
                      }}
                    >
                      {step.icon}
                    </Box>
                  )}
                >
                  <Typography
                    sx={{
                      color: index <= activeStep ? 'text.primary' : 'text.secondary',
                      fontWeight: index === activeStep ? 600 : 400,
                    }}
                  >
                    {step.label}
                  </Typography>
                </StepLabel>
              </Step>
            ))}
          </Stepper>
        </CardContent>
      </Card>

      {/* Hardware Status Indicator */}
      <Box sx={{ mb: 3 }}>
        <HardwareStatus />
      </Box>

      <Grid container spacing={3}>
        {/* Left Panel - Upload and Settings */}
        <Grid item xs={12} md={5} lg={4}>
          <Fade in timeout={600}>
            <Box>
              {/* Upload Section */}
              <Card
                sx={{
                  mb: 3,
                  background: 'rgba(26, 31, 58, 0.6)',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(0, 180, 216, 0.2)',
                  transition: 'all 0.3s ease',
                  '&:hover': {
                    border: '1px solid rgba(0, 180, 216, 0.4)',
                    boxShadow: '0 8px 32px rgba(0, 180, 216, 0.15)',
                  },
                }}
              >
                <CardContent>
                  <Typography
                    variant="h6"
                    sx={{
                      mb: 2,
                      fontWeight: 600,
                      display: 'flex',
                      alignItems: 'center',
                      gap: 1,
                    }}
                  >
                    <UploadIcon sx={{ color: 'primary.main' }} />
                    Upload Video
                  </Typography>
                  <VideoUpload />
                </CardContent>
              </Card>

              {/* Background Settings */}
              <Zoom in={!!uploadedVideo} timeout={400}>
                <Card
                  sx={{
                    mb: 3,
                    background: 'rgba(26, 31, 58, 0.6)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(0, 180, 216, 0.2)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      border: '1px solid rgba(0, 180, 216, 0.4)',
                      boxShadow: '0 8px 32px rgba(0, 180, 216, 0.15)',
                    },
                  }}
                >
                  <CardContent>
                    <Typography
                      variant="h6"
                      sx={{
                        mb: 2,
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                      }}
                    >
                      <SettingsIcon sx={{ color: 'primary.main' }} />
                      Background Settings
                    </Typography>
                    <BackgroundSettings />
                  </CardContent>
                </Card>
              </Zoom>

              {/* Processing Controls */}
              {uploadedVideo && (
                <Zoom in timeout={500}>
                  <Card
                    sx={{
                      background: 'rgba(26, 31, 58, 0.6)',
                      backdropFilter: 'blur(10px)',
                      border: '1px solid rgba(0, 180, 216, 0.2)',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        border: '1px solid rgba(0, 180, 216, 0.4)',
                        boxShadow: '0 8px 32px rgba(0, 180, 216, 0.15)',
                      },
                    }}
                  >
                    <CardContent>
                      <ProcessingControls />
                    </CardContent>
                  </Card>
                </Zoom>
              )}
            </Box>
          </Fade>
        </Grid>

        {/* Right Panel - Preview and Progress */}
        <Grid item xs={12} md={7} lg={8}>
          <Fade in timeout={800}>
            <Card
              sx={{
                height: '100%',
                minHeight: 600,
                background: 'rgba(26, 31, 58, 0.6)',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(0, 180, 216, 0.2)',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              <CardContent sx={{ height: '100%', p: 0 }}>
                <Box
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 0,
                  }}
                >
                  {/* Preview Header */}
                  <Box
                    sx={{
                      p: 2,
                      borderBottom: '1px solid rgba(0, 180, 216, 0.2)',
                      background: 'rgba(26, 31, 58, 0.4)',
                    }}
                  >
                    <Typography
                      variant="h6"
                      sx={{
                        fontWeight: 600,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                      }}
                    >
                      Video Preview
                      {processingStatus === 'processing' && (
                        <Chip
                          label="Processing..."
                          size="small"
                          sx={{
                            ml: 1,
                            background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                            animation: 'pulse 2s ease-in-out infinite',
                            '@keyframes pulse': {
                              '0%, 100%': { opacity: 1 },
                              '50%': { opacity: 0.7 },
                            },
                          }}
                        />
                      )}
                      {processingStatus === 'completed' && (
                        <Chip
                          label="Completed!"
                          size="small"
                          color="success"
                          sx={{ ml: 1 }}
                        />
                      )}
                    </Typography>
                  </Box>

                  {/* Video Preview Content */}
                  <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                    <Box sx={{ flex: 1, minHeight: 0 }}>
                      <DualVideoPreview />
                    </Box>
                    {(processingStatus === 'processing' || processingStatus === 'started') && (
                      <Box sx={{ p: 2 }}>
                        <ProcessingProgress />
                      </Box>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Fade>
        </Grid>
      </Grid>

      {/* Connection Error Alert */}
      {connectionError && (
        <Alert
          severity={connectionError.includes('Waiting for Python server to start') ? 'info' : 'error'}
          sx={{
            position: 'fixed',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            maxWidth: 600,
            boxShadow: connectionError.includes('Waiting for Python server to start')
              ? '0 8px 32px rgba(59, 130, 246, 0.25)'
              : '0 8px 32px rgba(239, 68, 68, 0.3)',
            border: connectionError.includes('Waiting for Python server to start')
              ? '1px solid rgba(59, 130, 246, 0.5)'
              : '1px solid rgba(239, 68, 68, 0.5)',
            background: 'rgba(26, 31, 58, 0.95)',
          }}
        >
          {connectionError}
        </Alert>
      )}

      {/* Success Alert */}
      <Snackbar
        open={processingStatus === 'completed' && showSuccessAlert}
        autoHideDuration={6000}
        onClose={() => setShowSuccessAlert(false)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          severity="success"
          sx={{
            boxShadow: '0 8px 32px rgba(74, 222, 128, 0.3)',
            border: '1px solid rgba(74, 222, 128, 0.5)',
            background: 'rgba(26, 31, 58, 0.95)',
          }}
        >
          Video processed successfully! Click the download button to save your video.
        </Alert>
      </Snackbar>

      {/* System Alerts (Memory warnings, etc.) */}
      <SystemAlert />
    </Container>
  );
}

