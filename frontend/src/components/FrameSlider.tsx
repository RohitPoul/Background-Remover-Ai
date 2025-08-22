import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Slider, 
  Typography, 
  IconButton, 
  Paper,
  Tooltip,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  SkipPrevious as PrevIcon,
  SkipNext as NextIcon,
  Pause as PauseIcon,
  PlayArrow as PlayIcon,
  FirstPage as FirstIcon,
  LastPage as LastIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function FrameSlider() {
  const {
    processedFrames,
    selectedFrameIndex,
    setSelectedFrameIndex,
    currentFrame,
    totalFrames,
    processingStatus,
    backgroundType,
  } = useVideoProcessor();

  const [isPlaying, setIsPlaying] = useState(false);
  const [frameImageUrl, setFrameImageUrl] = useState<string>('');
  const [isLoadingFrame, setIsLoadingFrame] = useState(false);

  const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');

  // Debug log for processed frames
  useEffect(() => {
    if (processedFrames.length > 0) {
      console.log(`🎞️ [FrameSlider] processedFrames updated: ${processedFrames.length} frames`);
      console.log('🎞️ [FrameSlider] First frame URL:', processedFrames[0]);
    }
  }, [processedFrames]);

  // Load frame image when selection changes
  useEffect(() => {
    if (processedFrames.length > 0 && selectedFrameIndex < processedFrames.length) {
      const frameUrl = `${API_BASE}${processedFrames[selectedFrameIndex]}`;
      console.log(`🖼️ [FrameSlider] Loading frame ${selectedFrameIndex + 1}: ${frameUrl}`);
      setIsLoadingFrame(true);
      setFrameImageUrl(frameUrl);
    }
  }, [selectedFrameIndex, processedFrames, API_BASE]);

  // Auto-play functionality
  useEffect(() => {
    if (isPlaying && processedFrames.length > 0) {
      const timer = setTimeout(() => {
        if (selectedFrameIndex < processedFrames.length - 1) {
          setSelectedFrameIndex(selectedFrameIndex + 1);
        } else {
          setSelectedFrameIndex(0); // Loop back to start
        }
      }, 100); // 10 fps playback
      return () => clearTimeout(timer);
    }
  }, [isPlaying, selectedFrameIndex, processedFrames, setSelectedFrameIndex]);

  // Auto-select latest frame during processing
  useEffect(() => {
    if (processingStatus === 'processing' && processedFrames.length > 0 && !isPlaying) {
      // Auto-select the latest processed frame
      setSelectedFrameIndex(processedFrames.length - 1);
    }
  }, [processedFrames.length, processingStatus, isPlaying, setSelectedFrameIndex]);

  const handleSliderChange = (_: Event, value: number | number[]) => {
    setSelectedFrameIndex(value as number);
    setIsPlaying(false); // Stop playing when manually selecting
  };

  const handlePrevFrame = () => {
    if (selectedFrameIndex > 0) {
      setSelectedFrameIndex(selectedFrameIndex - 1);
      setIsPlaying(false);
    }
  };

  const handleNextFrame = () => {
    if (selectedFrameIndex < processedFrames.length - 1) {
      setSelectedFrameIndex(selectedFrameIndex + 1);
      setIsPlaying(false);
    }
  };

  const handleFirstFrame = () => {
    setSelectedFrameIndex(0);
    setIsPlaying(false);
  };

  const handleLastFrame = () => {
    if (processedFrames.length > 0) {
      setSelectedFrameIndex(processedFrames.length - 1);
      setIsPlaying(false);
    }
  };

  const togglePlayback = () => {
    if (processedFrames.length > 1) {
      setIsPlaying(!isPlaying);
    }
  };

  if (processingStatus === 'idle' || totalFrames === 0) {
    return null;
  }

  return (
    <Paper sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.3)' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="body2" sx={{ fontWeight: 600 }}>
          Frame Preview
        </Typography>
        <Chip 
          label={`${selectedFrameIndex + 1} / ${processedFrames.length || totalFrames}`}
          size="small"
          color={processingStatus === 'processing' ? 'warning' : 'success'}
        />
        {processingStatus === 'processing' && (
          <Typography variant="caption" sx={{ color: 'text.secondary', ml: 'auto' }}>
            Processing... {currentFrame}/{totalFrames}
          </Typography>
        )}
      </Box>

      {/* Frame Display */}
      {frameImageUrl && (
        <Box 
          sx={{ 
            position: 'relative',
            width: '100%', 
            height: 300, 
            mb: 2,
            background: backgroundType === 'Transparent' ? 
              'repeating-conic-gradient(#808080 0% 25%, transparent 0% 50%) 50% / 20px 20px' : 
              'transparent',
            borderRadius: 1,
            overflow: 'hidden',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {isLoadingFrame && (
            <CircularProgress 
              size={40} 
              sx={{ position: 'absolute', zIndex: 1 }} 
            />
          )}
          <img 
            src={frameImageUrl}
            alt={`Frame ${selectedFrameIndex + 1}`}
            style={{ 
              width: '100%', 
              height: '100%', 
              objectFit: 'contain',
              opacity: isLoadingFrame ? 0.5 : 1,
              transition: 'opacity 0.2s',
            }}
            onLoad={() => {
              console.log(`✅ [FrameSlider] Frame loaded successfully: ${frameImageUrl}`);
              setIsLoadingFrame(false);
            }}
            onError={(e) => {
              console.error(`❌ [FrameSlider] Error loading frame: ${frameImageUrl}`, e);
              setIsLoadingFrame(false);
            }}
          />
        </Box>
      )}

      {/* Slider Control */}
      <Box sx={{ px: 2 }}>
        <Slider
          value={selectedFrameIndex}
          min={0}
          max={Math.max(0, (processedFrames.length || 1) - 1)}
          step={1}
          onChange={handleSliderChange}
          disabled={processedFrames.length === 0}
          marks={processedFrames.length <= 30}
          valueLabelDisplay="auto"
          valueLabelFormat={(value) => `Frame ${value + 1}`}
          sx={{
            '& .MuiSlider-thumb': {
              width: 20,
              height: 20,
            },
          }}
        />
      </Box>

      {/* Playback Controls */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, mt: 2 }}>
        <Tooltip title="First Frame">
          <span>
            <IconButton 
              onClick={handleFirstFrame}
              disabled={processedFrames.length === 0 || selectedFrameIndex === 0}
              size="small"
            >
              <FirstIcon />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Previous Frame">
          <span>
            <IconButton 
              onClick={handlePrevFrame}
              disabled={processedFrames.length === 0 || selectedFrameIndex === 0}
              size="small"
            >
              <PrevIcon />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title={isPlaying ? "Pause" : "Play All Frames"}>
          <span>
            <IconButton 
              onClick={togglePlayback}
              disabled={processedFrames.length <= 1}
              color="primary"
              size="medium"
            >
              {isPlaying ? <PauseIcon /> : <PlayIcon />}
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Next Frame">
          <span>
            <IconButton 
              onClick={handleNextFrame}
              disabled={processedFrames.length === 0 || selectedFrameIndex >= processedFrames.length - 1}
              size="small"
            >
              <NextIcon />
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Last Frame">
          <span>
            <IconButton 
              onClick={handleLastFrame}
              disabled={processedFrames.length === 0 || selectedFrameIndex === processedFrames.length - 1}
              size="small"
            >
              <LastIcon />
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {/* Frame Info */}
      <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', px: 1 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {processedFrames.length} frames processed
        </Typography>
        {processingStatus === 'processing' && (
          <Typography variant="caption" sx={{ color: 'warning.main' }}>
            Live preview - check quality before completion
          </Typography>
        )}
      </Box>
    </Paper>
  );
}
