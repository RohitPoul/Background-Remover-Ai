import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  Alert,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Settings as SettingsIcon,
  PlayArrow as ProcessIcon,
  Download as DownloadIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  Speed as SpeedIcon,
  HighQuality as QualityIcon,
} from '@mui/icons-material';

interface HelpDialogProps {
  open: boolean;
  onClose: () => void;
}

const steps = [
  {
    label: 'Upload Your Video',
    icon: <UploadIcon />,
    content: (
      <Box>
        <Typography variant="body2" paragraph>
          Click the upload area or drag and drop your video file. Supported formats:
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip label="MP4" size="small" />
          <Chip label="MOV" size="small" />
          <Chip label="AVI" size="small" />
          <Chip label="WebM" size="small" />
        </Box>
        <Alert severity="info" sx={{ mb: 2 }}>
          <Typography variant="body2">
            <strong>Tip:</strong> For best results, use videos with clear subject separation from the background.
            Videos up to 500MB with ~200 frames work best.
          </Typography>
        </Alert>
      </Box>
    ),
  },
  {
    label: 'Configure Background',
    icon: <SettingsIcon />,
    content: (
      <Box>
        <Typography variant="body2" paragraph>
          Choose what to replace the background with:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemIcon><CheckIcon color="success" /></ListItemIcon>
            <ListItemText 
              primary="Transparent" 
              secondary="Remove background completely (alpha channel)" 
            />
          </ListItem>
          <ListItem>
            <ListItemIcon><CheckIcon color="success" /></ListItemIcon>
            <ListItemText 
              primary="Solid Color" 
              secondary="Replace with any color (green screen, blue, etc.)" 
            />
          </ListItem>
          <ListItem>
            <ListItemIcon><CheckIcon color="success" /></ListItemIcon>
            <ListItemText 
              primary="Custom Image" 
              secondary="Use your own background image" 
            />
          </ListItem>
          <ListItem>
            <ListItemIcon><CheckIcon color="success" /></ListItemIcon>
            <ListItemText 
              primary="Video Background" 
              secondary="Animated background from another video" 
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    label: 'Process Video',
    icon: <ProcessIcon />,
    content: (
      <Box>
        <Typography variant="body2" paragraph>
          Configure processing options and start:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemIcon><SpeedIcon color="primary" /></ListItemIcon>
            <ListItemText 
              primary="Fast Mode" 
              secondary="Use BiRefNet_lite for faster processing (slightly lower quality)" 
            />
          </ListItem>
          <ListItem>
            <ListItemIcon><QualityIcon color="primary" /></ListItemIcon>
            <ListItemText 
              primary="High Quality" 
              secondary="Use full BiRefNet model for best results (slower)" 
            />
          </ListItem>
        </List>
        <Alert severity="warning" sx={{ mt: 2 }}>
          <Typography variant="body2">
            <strong>Processing Time:</strong> Depends on video length and quality settings. 
            A 10-second video typically takes 2-5 minutes.
          </Typography>
        </Alert>
      </Box>
    ),
  },
  {
    label: 'Download Result',
    icon: <DownloadIcon />,
    content: (
      <Box>
        <Typography variant="body2" paragraph>
          Once processing completes, download your video in various formats:
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip label="MP4 (Universal)" size="small" color="primary" />
          <Chip label="WebM (Web optimized)" size="small" />
          <Chip label="MOV (Professional)" size="small" />
        </Box>
        <Typography variant="body2">
          The processed video will maintain the original quality and frame rate while 
          replacing the background according to your settings.
        </Typography>
      </Box>
    ),
  },
];

const tips = [
  {
    icon: <InfoIcon color="info" />,
    title: "Best Video Quality",
    content: "Use well-lit videos with clear subject-background separation for optimal AI detection."
  },
  {
    icon: <SpeedIcon color="warning" />,
    title: "Performance Tips",
    content: "Enable Fast Mode for quicker results, or use High Quality for professional output."
  },
  {
    icon: <WarningIcon color="error" />,
    title: "File Size Limits",
    content: "Keep videos under 500MB and ~200 frames for best performance and stability."
  },
];

export default function HelpDialog({ open, onClose }: HelpDialogProps) {
  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="md" 
      fullWidth
      PaperProps={{
        sx: {
          background: 'rgba(26, 31, 58, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(0, 180, 216, 0.3)',
        }
      }}
    >
      <DialogTitle
        sx={{
          background: 'linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 245, 255, 0.05) 100%)',
          borderBottom: '1px solid rgba(0, 180, 216, 0.2)',
        }}
      >
        <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
          How to Use Video Background Remover
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
          AI-powered background removal and replacement made easy
        </Typography>
      </DialogTitle>
      
      <DialogContent sx={{ p: 3 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2, color: 'primary.main' }}>
            Step-by-Step Guide
          </Typography>
          <Stepper orientation="vertical">
            {steps.map((step, index) => (
              <Step key={step.label} active>
                <StepLabel
                  StepIconComponent={() => (
                    <Box
                      sx={{
                        width: 32,
                        height: 32,
                        borderRadius: '50%',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                        color: 'white',
                        fontSize: '1.2rem',
                      }}
                    >
                      {step.icon}
                    </Box>
                  )}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                    {step.label}
                  </Typography>
                </StepLabel>
                <StepContent>
                  <Box sx={{ ml: 1, pb: 2 }}>
                    {step.content}
                  </Box>
                </StepContent>
              </Step>
            ))}
          </Stepper>
        </Box>

        <Divider sx={{ my: 3, borderColor: 'rgba(0, 180, 216, 0.2)' }} />

        <Box>
          <Typography variant="h6" sx={{ mb: 2, color: 'primary.main' }}>
            Pro Tips & Best Practices
          </Typography>
          {tips.map((tip, index) => (
            <Box key={index} sx={{ display: 'flex', gap: 2, mb: 2 }}>
              {tip.icon}
              <Box>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  {tip.title}
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                  {tip.content}
                </Typography>
              </Box>
            </Box>
          ))}
        </Box>

        <Divider sx={{ my: 3, borderColor: 'rgba(0, 180, 216, 0.2)' }} />

        <Box>
          <Typography variant="h6" sx={{ mb: 2, color: 'primary.main' }}>
            🧠 AI Model Credits
          </Typography>
          <Alert severity="info" sx={{ background: 'rgba(0, 180, 216, 0.1)', mb: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Powered by BiRefNet AI Model</strong>
            </Typography>
            <Typography variant="body2">
              This application uses the BiRefNet (Bilateral Reference Network) model for high-quality 
              background segmentation, developed by <strong>Zheng Peng, Jianbo Jiao, and colleagues</strong> 
              at the University of Birmingham.
            </Typography>
            <Box sx={{ mt: 1 }}>
              <Button
                size="small"
                onClick={() => window.open('https://github.com/ZhengPeng7/BiRefNet', '_blank')}
                sx={{ 
                  color: 'primary.main',
                  textTransform: 'none',
                  p: 0,
                  minWidth: 'auto',
                  '&:hover': { background: 'transparent' }
                }}
              >
                → View Original Research & Model
              </Button>
            </Box>
          </Alert>
        </Box>

        <Divider sx={{ my: 3, borderColor: 'rgba(0, 180, 216, 0.2)' }} />

        <Alert severity="success" sx={{ background: 'rgba(76, 175, 80, 0.1)' }}>
          <Typography variant="body2">
            <strong>Need Help?</strong> Check the example presets above to see different background 
            removal options in action. Each preset demonstrates a different use case.
          </Typography>
        </Alert>
      </DialogContent>
      
      <DialogActions sx={{ p: 3, borderTop: '1px solid rgba(0, 180, 216, 0.2)' }}>
        <Button 
          onClick={onClose}
          variant="contained"
          sx={{
            background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
            '&:hover': {
              background: 'linear-gradient(135deg, #0096c7 0%, #005577 100%)',
            },
          }}
        >
          Got It!
        </Button>
      </DialogActions>
    </Dialog>
  );
}
