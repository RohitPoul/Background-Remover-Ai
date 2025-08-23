import React, { useEffect, useState } from 'react';
import { 
  Box, 
  Slider, 
  Typography, 
  IconButton,
  Tooltip,
  Chip
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
  } = useVideoProcessor();

  const [isPlaying, setIsPlaying] = useState(false);

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
    <Box sx={{ bgcolor: 'rgba(0,0,0,0.2)', p: 1, borderRadius: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Chip 
          label={`Frame ${selectedFrameIndex + 1} / ${processedFrames.length || totalFrames}`}
          size="small"
          color={processingStatus === 'processing' ? 'warning' : 'success'}
        />
        {processingStatus === 'processing' && (
          <Typography variant="caption" sx={{ color: 'text.secondary', ml: 'auto' }}>
            Processing... {currentFrame}/{totalFrames}
          </Typography>
        )}
      </Box>

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
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0.5, mt: 1 }}>
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

    </Box>
  );
}
