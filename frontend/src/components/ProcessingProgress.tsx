import React from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Paper,
  Grid,
} from '@mui/material';
import {
  Timer as TimerIcon,
  Speed as SpeedIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function ProcessingProgress() {
  const { progress, elapsedTime, currentFrame, totalFrames } = useVideoProcessor();
  
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const estimateRemainingTime = () => {
    if (progress > 0 && elapsedTime > 0) {
      const totalEstimated = (elapsedTime / progress) * 100;
      const remaining = totalEstimated - elapsedTime;
      return formatTime(remaining);
    }
    return '--:--';
  };

  const processingSpeed = () => {
    if (currentFrame > 0 && elapsedTime > 0) {
      const fps = currentFrame / elapsedTime;
      return `${fps.toFixed(1)} fps`;
    }
    return '-- fps';
  };

  return (
    <Paper
      sx={{
        p: 3,
        background: 'rgba(26, 31, 58, 0.95)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(0, 180, 216, 0.3)',
      }}
    >
      {/* Progress Bar */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Processing Progress
          </Typography>
          <Typography variant="body2" sx={{ fontWeight: 600, color: 'primary.main' }}>
            {progress.toFixed(1)}%
          </Typography>
        </Box>
        
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 8,
            borderRadius: 4,
            backgroundColor: 'rgba(0, 180, 216, 0.1)',
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: 'linear-gradient(90deg, #00b4d8 0%, #00f5ff 100%)',
              boxShadow: '0 2px 10px rgba(0, 180, 216, 0.5)',
            },
          }}
        />
        
        {totalFrames > 0 && (
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mt: 1,
              color: 'text.secondary',
              textAlign: 'center',
            }}
          >
            Frame {currentFrame} of {totalFrames}
          </Typography>
        )}
      </Box>

      {/* Statistics Grid */}
      <Grid container spacing={2}>
        <Grid item xs={4}>
          <Paper
            sx={{
              p: 1.5,
              background: 'rgba(0, 180, 216, 0.1)',
              border: '1px solid rgba(0, 180, 216, 0.2)',
              textAlign: 'center',
            }}
          >
            <TimerIcon sx={{ fontSize: 24, color: 'primary.main', mb: 0.5 }} />
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
              Elapsed
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {formatTime(elapsedTime)}
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={4}>
          <Paper
            sx={{
              p: 1.5,
              background: 'rgba(0, 180, 216, 0.1)',
              border: '1px solid rgba(0, 180, 216, 0.2)',
              textAlign: 'center',
            }}
          >
            <TimerIcon sx={{ fontSize: 24, color: 'warning.main', mb: 0.5 }} />
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
              Remaining
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {estimateRemainingTime()}
            </Typography>
          </Paper>
        </Grid>
        
        <Grid item xs={4}>
          <Paper
            sx={{
              p: 1.5,
              background: 'rgba(0, 180, 216, 0.1)',
              border: '1px solid rgba(0, 180, 216, 0.2)',
              textAlign: 'center',
            }}
          >
            <SpeedIcon sx={{ fontSize: 24, color: 'success.main', mb: 0.5 }} />
            <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
              Speed
            </Typography>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {processingSpeed()}
            </Typography>
          </Paper>
        </Grid>
      </Grid>

      {/* Status Message */}
      <Box
        sx={{
          mt: 2,
          p: 1.5,
          background: 'rgba(0, 180, 216, 0.05)',
          borderRadius: 1,
          border: '1px solid rgba(0, 180, 216, 0.2)',
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
            background: '#00f5ff',
            animation: 'pulse 2s ease-in-out infinite',
            '@keyframes pulse': {
              '0%, 100%': { opacity: 1, transform: 'scale(1)' },
              '50%': { opacity: 0.5, transform: 'scale(1.2)' },
            },
          }}
        />
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          AI is processing your video with background removal...
        </Typography>
      </Box>
    </Paper>
  );
}