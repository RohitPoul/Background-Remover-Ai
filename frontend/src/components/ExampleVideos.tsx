import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
} from '@mui/material';
import {
  PlayCircle as PlayIcon,
  Image as ImageIcon,
  VideoLibrary as VideoIcon,
  Palette as ColorIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

interface Example {
  video: string;
  backgroundType: 'Color' | 'Image' | 'Video' | 'Transparent';
  backgroundFile?: string;
  backgroundColor?: string;
  label: string;
  description: string;
}

const examples: Example[] = [
  {
    video: 'rickroll-2sec.mp4',
    backgroundType: 'Transparent',
    label: 'Transparent Background',
    description: 'Remove background with alpha channel',
  },
  {
    video: 'rickroll-2sec.mp4',
    backgroundType: 'Color',
    backgroundColor: '#00FF00',
    label: 'Green Screen',
    description: 'Replace with solid color',
  },
  {
    video: 'rickroll-2sec.mp4',
    backgroundType: 'Image',
    backgroundFile: 'images.webp',
    label: 'Image Background',
    description: 'Replace with custom image',
  },
  {
    video: 'rickroll-2sec.mp4',
    backgroundType: 'Video',
    backgroundFile: 'background.mp4',
    label: 'Video Background',
    description: 'Replace with animated background',
  },
];

export default function ExampleVideos() {
  const {
    setUploadedVideo,
    setBackgroundType,
    setBackgroundColor,
    setBackgroundImage,
    setBackgroundVideo,
  } = useVideoProcessor();

  const handleExampleClick = async (example: Example) => {
    try {
      // Load the example video
      const videoResponse = await fetch(`/${example.video}`);
      const videoBlob = await videoResponse.blob();
      const videoFile = new File([videoBlob], example.video, { type: 'video/mp4' });
      setUploadedVideo(videoFile);

      // Set background type
      setBackgroundType(example.backgroundType);

      // Set background based on type
      if (example.backgroundType === 'Color' && example.backgroundColor) {
        setBackgroundColor(example.backgroundColor);
      } else if (example.backgroundType === 'Image' && example.backgroundFile) {
        const imgResponse = await fetch(`/${example.backgroundFile}`);
        const imgBlob = await imgResponse.blob();
        const imgFile = new File([imgBlob], example.backgroundFile, { type: 'image/webp' });
        setBackgroundImage(imgFile);
      } else if (example.backgroundType === 'Video' && example.backgroundFile) {
        const bgResponse = await fetch(`/${example.backgroundFile}`);
        const bgBlob = await bgResponse.blob();
        const bgFile = new File([bgBlob], example.backgroundFile, { type: 'video/mp4' });
        setBackgroundVideo(bgFile);
      }
    } catch (error) {
      console.error('Error loading example:', error);
    }
  };

  const getIcon = (type: string) => {
    switch (type) {
      case 'Transparent':
        return <PlayIcon sx={{ color: 'secondary.main' }} />;
      case 'Color':
        return <ColorIcon sx={{ color: 'success.main' }} />;
      case 'Image':
        return <ImageIcon sx={{ color: 'warning.main' }} />;
      case 'Video':
        return <VideoIcon sx={{ color: 'info.main' }} />;
      default:
        return <PlayIcon />;
    }
  };

  return (
    <Box>
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
        <PlayIcon sx={{ color: 'primary.main' }} />
        Example Presets
      </Typography>
      
      <Typography variant="body2" sx={{ color: 'text.secondary', mb: 3 }}>
        Try these example configurations to see the different background removal options in action.
      </Typography>

      <Grid container spacing={2}>
        {examples.map((example, index) => (
          <Grid item xs={12} sm={6} key={index}>
            <Paper
              sx={{
                p: 2,
                cursor: 'pointer',
                border: '1px solid rgba(0, 180, 216, 0.2)',
                background: 'rgba(0, 180, 216, 0.03)',
                transition: 'all 0.3s ease',
                '&:hover': {
                  border: '1px solid rgba(0, 180, 216, 0.5)',
                  background: 'rgba(0, 180, 216, 0.1)',
                  transform: 'translateY(-2px)',
                  boxShadow: '0 4px 20px rgba(0, 180, 216, 0.2)',
                },
              }}
              onClick={() => handleExampleClick(example)}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,
                    background: 'rgba(0, 180, 216, 0.1)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {getIcon(example.backgroundType)}
                </Box>
                
                <Box sx={{ flexGrow: 1 }}>
                  <Typography
                    variant="subtitle2"
                    sx={{ fontWeight: 600, mb: 0.5 }}
                  >
                    {example.label}
                  </Typography>
                  <Typography
                    variant="caption"
                    sx={{ color: 'text.secondary', display: 'block' }}
                  >
                    {example.description}
                  </Typography>
                </Box>
                
                <Chip
                  label="Load"
                  size="small"
                  color="primary"
                  variant="outlined"
                  sx={{
                    fontWeight: 600,
                    borderWidth: 2,
                    '&:hover': {
                      borderColor: 'primary.main',
                      background: 'rgba(0, 180, 216, 0.1)',
                    },
                  }}
                />
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Box
        sx={{
          mt: 3,
          p: 2,
          background: 'rgba(0, 180, 216, 0.05)',
          borderRadius: 1,
          border: '1px solid rgba(0, 180, 216, 0.2)',
        }}
      >
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          <strong>Note:</strong> Example files (rickroll-2sec.mp4, background.mp4, images.webp) should be placed in the project root.
          The app can process videos up to 500MB with approximately 200 frames at once.
        </Typography>
      </Box>
    </Box>
  );
}
