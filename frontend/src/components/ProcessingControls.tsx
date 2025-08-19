import React, { useState } from 'react';
import {
  Box,
  Button,
  Slider,
  Typography,
  FormControlLabel,
  Switch,
  Collapse,
  Tooltip,
  Divider,
  Chip,
  Paper,
  ToggleButtonGroup,
  ToggleButton,
  Alert,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Download as DownloadIcon,
  Speed as SpeedIcon,
  Settings as SettingsIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  AutoAwesome as AIIcon,
  Tune as TuneIcon,
  VideoFile as VideoFileIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function ProcessingControls() {
  const {
    uploadedVideo,
    processingStatus,
    outputFile,
    fps,
    setFps,
    fastMode,
    setFastMode,
    maxWorkers,
    setMaxWorkers,
    outputFormat,
    setOutputFormat,
    backgroundType,
    startProcessing,
    cancelProcessing,
    downloadVideo,
  } = useVideoProcessor();

  const [showAdvanced, setShowAdvanced] = useState(false);

  const isProcessing = processingStatus === 'processing' || processingStatus === 'started';

  return (
    <Box>
      {/* Main Controls */}
      <Box sx={{ mb: 3 }}>
        {processingStatus === 'completed' && outputFile ? (
          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={<DownloadIcon />}
            onClick={downloadVideo}
            sx={{
              py: 2,
              background: 'linear-gradient(135deg, #4ade80 0%, #16a34a 100%)',
              fontSize: '1.1rem',
              fontWeight: 600,
              boxShadow: '0 4px 20px rgba(74, 222, 128, 0.3)',
              '&:hover': {
                background: 'linear-gradient(135deg, #4ade80 20%, #16a34a 120%)',
                boxShadow: '0 6px 30px rgba(74, 222, 128, 0.4)',
              },
            }}
          >
            Download Processed Video
          </Button>
        ) : isProcessing ? (
          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={<StopIcon />}
            onClick={cancelProcessing}
            sx={{
              py: 2,
              background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
              fontSize: '1.1rem',
              fontWeight: 600,
              '&:hover': {
                background: 'linear-gradient(135deg, #ef4444 20%, #dc2626 120%)',
              },
            }}
          >
            Cancel Processing
          </Button>
        ) : (
          <Button
            fullWidth
            variant="contained"
            size="large"
            startIcon={<PlayIcon />}
            onClick={startProcessing}
            disabled={!uploadedVideo || isProcessing}
            sx={{
              py: 2,
              background: uploadedVideo
                ? 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)'
                : 'rgba(255, 255, 255, 0.1)',
              fontSize: '1.1rem',
              fontWeight: 600,
              boxShadow: uploadedVideo ? '0 4px 20px rgba(0, 180, 216, 0.3)' : 'none',
              '&:hover': {
                background: uploadedVideo
                  ? 'linear-gradient(135deg, #00b4d8 20%, #0077b6 120%)'
                  : 'rgba(255, 255, 255, 0.1)',
                boxShadow: uploadedVideo ? '0 6px 30px rgba(0, 180, 216, 0.4)' : 'none',
              },
              '&.Mui-disabled': {
                background: 'rgba(255, 255, 255, 0.05)',
                color: 'rgba(255, 255, 255, 0.3)',
              },
            }}
          >
            Start Processing
          </Button>
        )}
      </Box>

      <Divider sx={{ mb: 2, borderColor: 'rgba(0, 180, 216, 0.2)' }} />

      {/* Quick Settings */}
      <Paper
        sx={{
          p: 2,
          mb: 2,
          background: 'rgba(0, 180, 216, 0.05)',
          border: '1px solid rgba(0, 180, 216, 0.2)',
        }}
      >
        <FormControlLabel
          control={
            <Switch
              checked={fastMode}
              onChange={(e) => setFastMode(e.target.checked)}
              disabled={isProcessing}
              sx={{
                '& .MuiSwitch-switchBase.Mui-checked': {
                  color: '#00b4d8',
                  '& + .MuiSwitch-track': {
                    backgroundColor: '#00b4d8',
                  },
                },
              }}
            />
          }
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <AIIcon sx={{ fontSize: 20, color: 'primary.main' }} />
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                Fast Processing Mode
              </Typography>
              <Chip
                label={fastMode ? 'ON' : 'OFF'}
                size="small"
                color={fastMode ? 'success' : 'default'}
                sx={{ ml: 1 }}
              />
            </Box>
          }
          sx={{ mb: 0 }}
        />
        <Typography variant="caption" sx={{ color: 'text.secondary', ml: 5, display: 'block' }}>
          {fastMode 
            ? 'Faster processing with slightly lower quality'
            : 'Best quality processing (slower)'}
        </Typography>
      </Paper>

      {/* Advanced Settings Toggle */}
      <Button
        fullWidth
        variant="outlined"
        size="small"
        startIcon={<TuneIcon />}
        endIcon={showAdvanced ? <ExpandLessIcon /> : <ExpandMoreIcon />}
        onClick={() => setShowAdvanced(!showAdvanced)}
        disabled={isProcessing}
        sx={{
          mb: 2,
          borderColor: 'rgba(0, 180, 216, 0.3)',
          color: 'text.secondary',
          '&:hover': {
            borderColor: 'primary.main',
            background: 'rgba(0, 180, 216, 0.05)',
          },
        }}
      >
        Advanced Settings
      </Button>

      {/* Advanced Settings */}
      <Collapse in={showAdvanced}>
        <Paper
          sx={{
            p: 2,
            background: 'rgba(0, 180, 216, 0.03)',
            border: '1px solid rgba(0, 180, 216, 0.2)',
          }}
        >
          {/* FPS Control */}
          <Box sx={{ mb: 3 }}>
            <Typography
              variant="subtitle2"
              sx={{
                mb: 1.5,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <SpeedIcon sx={{ fontSize: 18, color: 'primary.main' }} />
              Output FPS
              <Tooltip title="Frames per second for the output video. 0 = use original FPS">
                <Chip label="?" size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
              </Tooltip>
            </Typography>
            <Box sx={{ px: 1 }}>
              <Slider
                value={fps}
                onChange={(_, value) => setFps(value as number)}
                disabled={isProcessing}
                min={0}
                max={60}
                step={5}
                marks={[
                  { value: 0, label: 'Auto' },
                  { value: 15, label: '15' },
                  { value: 30, label: '30' },
                  { value: 60, label: '60' },
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => value === 0 ? 'Original' : `${value} fps`}
                sx={{
                  '& .MuiSlider-track': {
                    background: 'linear-gradient(90deg, #00b4d8 0%, #0077b6 100%)',
                  },
                  '& .MuiSlider-thumb': {
                    backgroundColor: '#00b4d8',
                    '&:hover': {
                      boxShadow: '0 0 0 8px rgba(0, 180, 216, 0.16)',
                    },
                  },
                }}
              />
            </Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', mt: 1, display: 'block' }}>
              Current: {fps === 0 ? 'Original FPS' : `${fps} fps`}
            </Typography>
          </Box>

          {/* Output Format Control */}
          <Box sx={{ mb: 3 }}>
            <Typography
              variant="subtitle2"
              sx={{
                mb: 1.5,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <VideoFileIcon sx={{ fontSize: 18, color: 'primary.main' }} />
              Output Format
              <Tooltip title="Choose the output video format. WebM and MOV support transparency.">
                <Chip label="?" size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
              </Tooltip>
            </Typography>
            <ToggleButtonGroup
              value={outputFormat}
              exclusive
              onChange={(_, value) => value && setOutputFormat(value)}
              size="small"
              fullWidth
              sx={{
                '& .MuiToggleButton-root': {
                  py: 1,
                  textTransform: 'none',
                },
              }}
            >
              <ToggleButton value="mp4">
                <Box>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>MP4</Typography>
                  <Typography variant="caption" sx={{ display: 'block', fontSize: '0.65rem', color: 'text.secondary' }}>
                    Standard
                  </Typography>
                </Box>
              </ToggleButton>
              <ToggleButton value="webm">
                <Box>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>WebM</Typography>
                  <Typography variant="caption" sx={{ display: 'block', fontSize: '0.65rem', color: 'text.secondary' }}>
                    Alpha support
                  </Typography>
                </Box>
              </ToggleButton>
              <ToggleButton value="mov">
                <Box>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>MOV</Typography>
                  <Typography variant="caption" sx={{ display: 'block', fontSize: '0.65rem', color: 'text.secondary' }}>
                    ProRes 4444
                  </Typography>
                </Box>
              </ToggleButton>
            </ToggleButtonGroup>
            {backgroundType === 'Transparent' && outputFormat === 'mp4' && (
              <Alert severity="warning" sx={{ mt: 1 }}>
                <Typography variant="caption">
                  MP4 doesn't support transparency. Switch to WebM or MOV for transparent backgrounds.
                </Typography>
              </Alert>
            )}
          </Box>

          {/* Worker Threads Control */}
          <Box>
            <Typography
              variant="subtitle2"
              sx={{
                mb: 1.5,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <SettingsIcon sx={{ fontSize: 18, color: 'primary.main' }} />
              Processing Threads
              <Tooltip title="Number of parallel processing threads. More threads = faster but uses more memory">
                <Chip label="?" size="small" sx={{ height: 18, fontSize: '0.7rem' }} />
              </Tooltip>
            </Typography>
            <Box sx={{ px: 1 }}>
              <Slider
                value={maxWorkers}
                onChange={(_, value) => setMaxWorkers(value as number)}
                disabled={isProcessing}
                min={1}
                max={4}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                  { value: 3, label: '3' },
                  { value: 4, label: '4' },
                ]}
                valueLabelDisplay="auto"
                sx={{
                  '& .MuiSlider-track': {
                    background: 'linear-gradient(90deg, #00b4d8 0%, #0077b6 100%)',
                  },
                  '& .MuiSlider-thumb': {
                    backgroundColor: '#00b4d8',
                    '&:hover': {
                      boxShadow: '0 0 0 8px rgba(0, 180, 216, 0.16)',
                    },
                  },
                }}
              />
            </Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', mt: 1, display: 'block' }}>
              Current: {maxWorkers} thread{maxWorkers > 1 ? 's' : ''}
              {maxWorkers <= 2 ? ' (Recommended for stability)' : ' (May use more memory)'}
            </Typography>
          </Box>
        </Paper>
      </Collapse>
    </Box>
  );
}