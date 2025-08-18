import React, { useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Fade,
  CircularProgress,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  VolumeUp as VolumeIcon,
  VolumeOff as VolumeOffIcon,
  Fullscreen as FullscreenIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function VideoPreview() {
  const {
    uploadedVideo,
    previewImage,
    processingStatus,
    outputFile,
  } = useVideoProcessor();

  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);
  const [isMuted, setIsMuted] = React.useState(true);

  useEffect(() => {
    if (uploadedVideo && videoRef.current) {
      const url = URL.createObjectURL(uploadedVideo);
      videoRef.current.src = url;
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
    return (
      <Box
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        <video
          ref={videoRef}
          controls
          style={{
            width: '100%',
            height: '100%',
            maxHeight: '500px',
            objectFit: 'contain',
          }}
          src={outputFile}
        />
      </Box>
    );
  }

  // Show processing preview
  if ((processingStatus === 'processing' || processingStatus === 'started') && previewImage) {
    return (
      <Fade in timeout={300}>
        <Box
          sx={{
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            p: 2,
          }}
        >
          <Box
            sx={{
              position: 'relative',
              maxWidth: '100%',
              maxHeight: '100%',
            }}
          >
            <img
              src={previewImage}
              alt="Processing preview"
              style={{
                width: '100%',
                height: 'auto',
                maxHeight: '400px',
                objectFit: 'contain',
                borderRadius: '8px',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
              }}
            />
            <Paper
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                px: 2,
                py: 1,
                background: 'rgba(26, 31, 58, 0.9)',
                backdropFilter: 'blur(10px)',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <CircularProgress size={16} sx={{ color: 'warning.main' }} />
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                Live Preview
              </Typography>
            </Paper>
          </Box>
        </Box>
      </Fade>
    );
  }

  // Show uploaded video preview
  if (uploadedVideo) {
    return (
      <Box
        sx={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
        }}
      >
        <video
          ref={videoRef}
          style={{
            width: '100%',
            height: '100%',
            maxHeight: '500px',
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
    );
  }

  // Default state - no video
  return (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
      }}
    >
      <Box
        sx={{
          width: 120,
          height: 120,
          borderRadius: '50%',
          background: 'rgba(0, 180, 216, 0.1)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          mb: 3,
        }}
      >
        <PlayIcon sx={{ fontSize: 60, color: 'primary.main', opacity: 0.5 }} />
      </Box>
      
      <Typography
        variant="h6"
        sx={{
          color: 'text.secondary',
          mb: 1,
          fontWeight: 600,
        }}
      >
        No Preview Available
      </Typography>
      
      <Typography
        variant="body2"
        sx={{
          color: 'text.secondary',
          opacity: 0.7,
          textAlign: 'center',
          maxWidth: 300,
        }}
      >
        Upload a video and start processing to see the preview
      </Typography>
    </Box>
  );
}