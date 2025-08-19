import React, { useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Fade,
  CircularProgress,
  Button,
  Chip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  VolumeUp as VolumeIcon,
  VolumeOff as VolumeOffIcon,
  Fullscreen as FullscreenIcon,
  Download as DownloadIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function VideoPreview() {
  const {
    uploadedVideo,
    previewImage,
    processingStatus,
    outputFile,
    downloadVideo,
    progress,
    currentFrame,
    totalFrames,
  } = useVideoProcessor();

  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [isMuted, setIsMuted] = React.useState(true);

  useEffect(() => {
    if (uploadedVideo && videoRef.current) {
      const url = URL.createObjectURL(uploadedVideo);
      videoRef.current.src = url;
      // Auto-play the video
      videoRef.current.play().catch(() => {
        // Ignore autoplay errors
      });
      return () => URL.revokeObjectURL(url);
    }
  }, [uploadedVideo]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const toggleMute = () => {
    if (videoRef.current) {
      videoRef.current.muted = !isMuted;
      setIsMuted(!isMuted);
    }
  };

  const handleFullscreen = () => {
    if (videoRef.current) {
      if (videoRef.current.requestFullscreen) {
        videoRef.current.requestFullscreen();
      }
    }
  };

  // Show processed video when available
  if (processingStatus === 'completed' && outputFile) {
    const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
    const fullUrl = `${API_BASE}${outputFile}`;
    
    return (
      <Fade in timeout={500}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {/* Success Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Chip
              icon={<CheckIcon />}
              label="Processing Complete"
              color="success"
              sx={{ fontWeight: 600 }}
            />
            <Button
              variant="contained"
              startIcon={<DownloadIcon />}
              onClick={downloadVideo}
              sx={{
                ml: 'auto',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
                },
              }}
            >
              Download Video
            </Button>
          </Box>

          {/* Video Container */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: 2,
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <video
              controls
              autoPlay
              loop
              style={{
                width: '100%',
                height: '100%',
                maxHeight: '500px',
                objectFit: 'contain',
              }}
              src={fullUrl}
            />
          </Box>
        </Box>
      </Fade>
    );
  }

  // Show live preview during processing
  if (processingStatus === 'processing' && previewImage) {
    return (
      <Fade in timeout={500}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {/* Processing Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} sx={{ color: 'warning.main' }} />
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'warning.main' }}>
              Processing Frame {currentFrame} of {totalFrames}
            </Typography>
            <Chip
              label={`${progress.toFixed(1)}%`}
              color="warning"
              size="small"
              sx={{ ml: 'auto', fontWeight: 600 }}
            />
          </Box>

          {/* Live Preview Container */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: 2,
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <img
              src={previewImage}
              alt="Processing preview"
              style={{
                width: '100%',
                height: '100%',
                maxHeight: '450px',
                objectFit: 'contain',
              }}
            />
            
            {/* Live Preview Badge */}
            <Paper
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                px: 2,
                py: 0.5,
                background: 'rgba(255, 152, 0, 0.9)',
                backdropFilter: 'blur(10px)',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: 'white',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.3 },
                    '100%': { opacity: 1 },
                  },
                }}
              />
              <Typography variant="caption" sx={{ fontWeight: 600, color: 'white' }}>
                LIVE PREVIEW
              </Typography>
            </Paper>
          </Box>
        </Box>
      </Fade>
    );
  }

  // Show uploaded video preview (idle state)
  if (uploadedVideo && processingStatus === 'idle') {
    return (
      <Fade in timeout={500}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            gap: 2,
          }}
        >
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
              Original Video
            </Typography>
            <Typography variant="caption" sx={{ ml: 'auto', color: 'text.secondary' }}>
              {uploadedVideo.name}
            </Typography>
          </Box>

          {/* Video Container */}
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: 'rgba(0, 0, 0, 0.2)',
              borderRadius: 2,
              overflow: 'hidden',
              position: 'relative',
            }}
          >
            <video
              ref={videoRef}
              style={{
                width: '100%',
                height: '100%',
                maxHeight: '450px',
                objectFit: 'contain',
              }}
              muted={isMuted}
              loop
            />
            
            {/* Video Controls Overlay */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 0,
                left: 0,
                right: 0,
                p: 2,
                background: 'linear-gradient(to top, rgba(0, 0, 0, 0.8), transparent)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 1,
              }}
            >
              <IconButton
                onClick={togglePlay}
                sx={{
                  color: 'white',
                  background: 'rgba(0, 180, 216, 0.3)',
                  '&:hover': {
                    background: 'rgba(0, 180, 216, 0.5)',
                  },
                }}
              >
                {isPlaying ? <PauseIcon /> : <PlayIcon />}
              </IconButton>
              
              <IconButton
                onClick={toggleMute}
                sx={{
                  color: 'white',
                  background: 'rgba(0, 180, 216, 0.3)',
                  '&:hover': {
                    background: 'rgba(0, 180, 216, 0.5)',
                  },
                }}
              >
                {isMuted ? <VolumeOffIcon /> : <VolumeIcon />}
              </IconButton>
              
              <IconButton
                onClick={handleFullscreen}
                sx={{
                  color: 'white',
                  background: 'rgba(0, 180, 216, 0.3)',
                  '&:hover': {
                    background: 'rgba(0, 180, 216, 0.5)',
                  },
                }}
              >
                <FullscreenIcon />
              </IconButton>
            </Box>
          </Box>
        </Box>
      </Fade>
    );
  }

  // Default empty state
  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      <Typography variant="h6" sx={{ color: 'text.secondary', opacity: 0.5 }}>
        Video Preview
      </Typography>
      <Typography variant="body2" sx={{ color: 'text.secondary', opacity: 0.3 }}>
        Upload a video to see preview
      </Typography>
    </Box>
  );
}