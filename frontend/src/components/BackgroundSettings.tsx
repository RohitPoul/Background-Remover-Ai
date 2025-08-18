import React, { useState } from 'react';
import {
  Box,
  ToggleButtonGroup,
  ToggleButton,
  TextField,
  Button,
  Typography,
  Paper,
  Fade,
  Tooltip,
  Chip,
} from '@mui/material';
import {
  Palette as ColorIcon,
  Image as ImageIcon,
  VideoLibrary as VideoIcon,
  Upload as UploadIcon,
  AutoAwesome as AutoAwesomeIcon,

} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

const backgroundOptions = [
  {
    value: 'Transparent',
    label: 'Transparent',
    icon: <AutoAwesomeIcon />,
    description: 'Remove background (Alpha channel)',
  },
  {
    value: 'Color',
    label: 'Solid Color',
    icon: <ColorIcon />,
    description: 'Use a solid color background',
  },
  {
    value: 'Image',
    label: 'Background Image',
    icon: <ImageIcon />,
    description: 'Upload a custom image',
  },
  {
    value: 'Video',
    label: 'Background Video',
    icon: <VideoIcon />,
    description: 'Use a video background',
  },
];

const colorPresets = [
  { color: '#00FF00', label: 'Green Screen' },
  { color: '#FFFFFF', label: 'White' },
  { color: '#000000', label: 'Black' },
  { color: '#FF0000', label: 'Red' },
  { color: '#0000FF', label: 'Blue' },
  { color: '#00B4D8', label: 'Cyan' },
  { color: '#9333EA', label: 'Purple' },
  { color: '#F59E0B', label: 'Amber' },
];

export default function BackgroundSettings() {
  const {
    backgroundType,
    setBackgroundType,
    backgroundColor,
    setBackgroundColor,
    backgroundImage,
    setBackgroundImage,
    backgroundVideo,
    setBackgroundVideo,
    videoHandling,
    setVideoHandling,
  } = useVideoProcessor();



  const handleBackgroundFileUpload = (event: React.ChangeEvent<HTMLInputElement>, type: 'image' | 'video') => {
    const file = event.target.files?.[0];
    if (file) {
      if (type === 'image') {
        setBackgroundImage(file);
      } else {
        setBackgroundVideo(file);
      }
    }
  };

  return (
    <Box>
      {/* Background Type Selection */}
      <Typography
        variant="subtitle2"
        sx={{
          mb: 2,
          color: 'text.secondary',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: 0.5,
        }}
      >
        Background Type
      </Typography>
      
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mb: 3 }}>
        {backgroundOptions.map((option) => (
          <Tooltip key={option.value} title={option.description} placement="top">
            <Paper
              onClick={() => setBackgroundType(option.value as any)}
              sx={{
                p: 1.5,
                cursor: 'pointer',
                textAlign: 'center',
                border: backgroundType === option.value 
                  ? '2px solid #00b4d8' 
                  : '1px solid rgba(0, 180, 216, 0.3)',
                background: backgroundType === option.value
                  ? 'linear-gradient(135deg, rgba(0, 180, 216, 0.2) 0%, rgba(0, 119, 182, 0.1) 100%)'
                  : 'transparent',
                transition: 'all 0.3s ease',
                '&:hover': {
                  borderColor: 'rgba(0, 180, 216, 0.5)',
                  background: 'rgba(0, 180, 216, 0.05)',
                },
              }}
            >
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0.5 }}>
                {React.cloneElement(option.icon, { 
                  sx: { 
                    fontSize: 24, 
                    color: backgroundType === option.value ? 'primary.main' : 'text.secondary'
                  } 
                })}
                <Typography 
                  variant="caption" 
                  sx={{ 
                    fontWeight: backgroundType === option.value ? 600 : 400,
                    color: backgroundType === option.value ? 'primary.main' : 'text.secondary'
                  }}
                >
                  {option.label}
                </Typography>
              </Box>
            </Paper>
          </Tooltip>
        ))}
      </Box>

      {/* Transparent Background Notice */}
      <Fade in={backgroundType === 'Transparent'} timeout={300}>
        <Box sx={{ display: backgroundType === 'Transparent' ? 'block' : 'none' }}>
          <Paper
            sx={{
              p: 2,
              border: '1px solid rgba(0, 180, 216, 0.3)',
              borderRadius: 1,
              background: 'rgba(0, 180, 216, 0.05)',
              textAlign: 'center',
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
            <Typography variant="body1" sx={{ mb: 1, fontWeight: 600 }}>
              Transparent Background Mode
            </Typography>
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
              Background will be removed with alpha channel preserved.
              Output will be in WebM or MOV format for transparency support.
            </Typography>
            <Chip
              label="Recommended: WebM format"
              color="primary"
              size="small"
              sx={{ mr: 1 }}
            />
            <Chip
              label="Alternative: MOV with ProRes"
              color="default"
              size="small"
            />
          </Paper>
        </Box>
      </Fade>

      {/* Color Background Options */}
      <Fade in={backgroundType === 'Color'} timeout={300}>
        <Box sx={{ display: backgroundType === 'Color' ? 'block' : 'none' }}>
          <Typography
            variant="subtitle2"
            sx={{
              mb: 2,
              color: 'text.secondary',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: 0.5,
            }}
          >
            Select Color
          </Typography>
          
          {/* Color Presets */}
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
            {colorPresets.map((preset) => (
              <Tooltip key={preset.color} title={preset.label}>
                <Paper
                  onClick={() => setBackgroundColor(preset.color)}
                  sx={{
                    width: 40,
                    height: 40,
                    bgcolor: preset.color,
                    cursor: 'pointer',
                    border: backgroundColor === preset.color
                      ? '3px solid #00b4d8'
                      : '2px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: 1,
                    transition: 'all 0.2s ease',
                    '&:hover': {
                      transform: 'scale(1.1)',
                      boxShadow: `0 4px 20px ${preset.color}40`,
                    },
                  }}
                />
              </Tooltip>
            ))}
          </Box>
          
          {/* Custom Color Input */}
          <TextField
            fullWidth
            label="Custom Color"
            value={backgroundColor}
            onChange={(e) => setBackgroundColor(e.target.value)}
            variant="outlined"
            InputProps={{
              startAdornment: (
                <Box
                  sx={{
                    width: 24,
                    height: 24,
                    bgcolor: backgroundColor,
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    borderRadius: 0.5,
                    mr: 1,
                  }}
                />
              ),
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  borderColor: 'rgba(0, 180, 216, 0.3)',
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(0, 180, 216, 0.5)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: 'primary.main',
                },
              },
            }}
          />
        </Box>
      </Fade>

      {/* Image Background Options */}
      <Fade in={backgroundType === 'Image'} timeout={300}>
        <Box sx={{ display: backgroundType === 'Image' ? 'block' : 'none' }}>
          <Paper
            sx={{
              p: 2,
              border: '1px solid rgba(0, 180, 216, 0.3)',
              borderRadius: 1,
              background: 'rgba(0, 180, 216, 0.05)',
            }}
          >
            {backgroundImage ? (
              <Box>
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                  Selected: {backgroundImage.name}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setBackgroundImage(null)}
                    sx={{
                      borderColor: 'rgba(239, 68, 68, 0.5)',
                      color: 'error.main',
                      '&:hover': {
                        borderColor: 'error.main',
                        background: 'rgba(239, 68, 68, 0.1)',
                      },
                    }}
                  >
                    Remove
                  </Button>
                  <Button
                    variant="contained"
                    size="small"
                    component="label"
                    startIcon={<UploadIcon />}
                    sx={{
                      background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #00b4d8 20%, #0077b6 120%)',
                      },
                    }}
                  >
                    Change Image
                    <input
                      type="file"
                      hidden
                      accept="image/*"
                      onChange={(e) => handleBackgroundFileUpload(e, 'image')}
                    />
                  </Button>
                </Box>
              </Box>
            ) : (
              <Box sx={{ textAlign: 'center' }}>
                <ImageIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                  No image selected
                </Typography>
                <Button
                  variant="contained"
                  component="label"
                  startIcon={<UploadIcon />}
                  sx={{
                    background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #00b4d8 20%, #0077b6 120%)',
                    },
                  }}
                >
                  Upload Image
                  <input
                    type="file"
                    hidden
                    accept="image/*"
                    onChange={(e) => handleBackgroundFileUpload(e, 'image')}
                  />
                </Button>
              </Box>
            )}
          </Paper>
        </Box>
      </Fade>

      {/* Video Background Options */}
      <Fade in={backgroundType === 'Video'} timeout={300}>
        <Box sx={{ display: backgroundType === 'Video' ? 'block' : 'none' }}>
          <Paper
            sx={{
              p: 2,
              border: '1px solid rgba(0, 180, 216, 0.3)',
              borderRadius: 1,
              background: 'rgba(0, 180, 216, 0.05)',
              mb: 2,
            }}
          >
            {backgroundVideo ? (
              <Box>
                <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                  Selected: {backgroundVideo.name}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setBackgroundVideo(null)}
                    sx={{
                      borderColor: 'rgba(239, 68, 68, 0.5)',
                      color: 'error.main',
                      '&:hover': {
                        borderColor: 'error.main',
                        background: 'rgba(239, 68, 68, 0.1)',
                      },
                    }}
                  >
                    Remove
                  </Button>
                  <Button
                    variant="contained"
                    size="small"
                    component="label"
                    startIcon={<UploadIcon />}
                    sx={{
                      background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #00b4d8 20%, #0077b6 120%)',
                      },
                    }}
                  >
                    Change Video
                    <input
                      type="file"
                      hidden
                      accept="video/*"
                      onChange={(e) => handleBackgroundFileUpload(e, 'video')}
                    />
                  </Button>
                </Box>
              </Box>
            ) : (
              <Box sx={{ textAlign: 'center' }}>
                <VideoIcon sx={{ fontSize: 48, color: 'primary.main', mb: 1 }} />
                <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
                  No video selected
                </Typography>
                <Button
                  variant="contained"
                  component="label"
                  startIcon={<UploadIcon />}
                  sx={{
                    background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #00b4d8 20%, #0077b6 120%)',
                    },
                  }}
                >
                  Upload Video
                  <input
                    type="file"
                    hidden
                    accept="video/*"
                    onChange={(e) => handleBackgroundFileUpload(e, 'video')}
                  />
                </Button>
              </Box>
            )}
          </Paper>
          
          {/* Video Handling Options */}
          {backgroundVideo && (
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  mb: 1,
                  color: 'text.secondary',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  letterSpacing: 0.5,
                }}
              >
                Video Sync Mode
              </Typography>
              <ToggleButtonGroup
                value={videoHandling}
                exclusive
                onChange={(_, value) => value && setVideoHandling(value)}
                size="small"
                fullWidth
                sx={{
                  '& .MuiToggleButton-root': {
                    py: 1,
                  },
                }}
              >
                <ToggleButton value="loop">
                  <Typography variant="caption">Loop</Typography>
                </ToggleButton>
                <ToggleButton value="slow_down">
                  <Typography variant="caption">Slow Down</Typography>
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>
          )}
        </Box>
      </Fade>
    </Box>
  );
}