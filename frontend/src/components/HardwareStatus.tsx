import React, { useEffect, useState } from 'react';
import { Box, Chip, Typography, Tooltip, CircularProgress } from '@mui/material';
import { 
  Memory as CpuIcon, 
  GraphicEq as GpuIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon 
} from '@mui/icons-material';

const API_BASE = process.env.NODE_ENV === 'production' 
  ? 'http://localhost:5000'  
  : 'http://localhost:5000';

interface HardwareInfo {
  processing_device: string;
  gpu_name: string;
  gpu_available: boolean;
  cpu_cores: number;
  cpu_threads: number;
  memory_gb: number;
  memory_available_gb: number;
  memory_usage_percent: number;
}

interface Settings {
  batch_size: number;
  max_workers: number;
  mixed_precision: boolean;
  model_quantization: boolean;
  model_preference: string;
  optimization_profile: string;
}

export default function HardwareStatus() {
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  useEffect(() => {
    // Try fetching immediately, then every 5 seconds if failed, up to 3 times
    const tryFetch = () => {
      fetchHardwareStatus();
      if (retryCount < 3 && error) {
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
          tryFetch();
        }, 5000);
      }
    };
    
    tryFetch();
    
    // Refresh every 30 seconds once connected
    const interval = setInterval(() => {
      if (!error) fetchHardwareStatus();
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchHardwareStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/hardware_status`);
      if (response.ok) {
        const data = await response.json();
        setHardware(data.hardware);
        setSettings(data.settings);
        setError(null);
      } else {
        setError('Failed to fetch hardware status');
      }
    } catch (err) {
      setError('Cannot connect to backend');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="caption">Detecting hardware...</Typography>
      </Box>
    );
  }

  if (error || !hardware) {
    return (
      <Box sx={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: 1.5, 
        p: 1.5,
        bgcolor: 'rgba(255, 152, 0, 0.1)',
        borderRadius: 1,
        border: '1px solid rgba(255, 152, 0, 0.3)'
      }}>
        <WarningIcon sx={{ fontSize: 18, color: 'warning.main' }} />
        <Box sx={{ flex: 1 }}>
          <Typography variant="caption" sx={{ color: 'warning.main', fontWeight: 600 }}>
            Backend not connected
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem', mt: 0.5 }}>
            Start the backend server: Run `npm start` or `python api_server.py`
          </Typography>
        </Box>
      </Box>
    );
  }

  const isGPU = hardware.processing_device === 'CUDA' || hardware.processing_device === 'MPS';
  const deviceColor = isGPU ? 'success' : 'warning';
  const deviceIcon = isGPU ? <GpuIcon /> : <CpuIcon />;
  const speedLevel = settings?.optimization_profile || 'unknown';
  
  // Determine performance level
  const getPerformanceColor = () => {
    switch (speedLevel.toLowerCase()) {
      case 'ultra':
      case 'high':
        return 'success';
      case 'medium':
        return 'info';
      case 'low':
      case 'potato':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 2, 
      p: 1.5,
      bgcolor: 'rgba(0,0,0,0.2)',
      borderRadius: 1,
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      {/* Processing Device */}
      <Tooltip title={
        <Box>
          <Typography variant="body2">
            {isGPU ? `GPU: ${hardware.gpu_name}` : `CPU: ${hardware.cpu_cores} cores, ${hardware.cpu_threads} threads`}
          </Typography>
          <Typography variant="caption">
            Memory: {hardware.memory_available_gb.toFixed(1)} / {hardware.memory_gb.toFixed(1)} GB available
          </Typography>
        </Box>
      }>
        <Chip
          icon={deviceIcon}
          label={isGPU ? 'GPU' : 'CPU'}
          color={deviceColor}
          size="small"
          sx={{ fontWeight: 600 }}
        />
      </Tooltip>

      {/* Performance Profile */}
      <Tooltip title={
        <Box>
          <Typography variant="body2">Optimization Profile: {speedLevel}</Typography>
          <Typography variant="caption">
            • Batch Size: {settings?.batch_size || 1}<br />
            • Workers: {settings?.max_workers || 1}<br />
            • Mixed Precision: {settings?.mixed_precision ? 'Yes' : 'No'}<br />
            • Quantization: {settings?.model_quantization ? 'Yes' : 'No'}
          </Typography>
        </Box>
      }>
        <Chip
          icon={<SpeedIcon />}
          label={speedLevel.toUpperCase()}
          color={getPerformanceColor()}
          size="small"
          variant="outlined"
        />
      </Tooltip>

      {/* Memory Usage */}
      <Tooltip title={`Memory Usage: ${hardware.memory_usage_percent.toFixed(1)}%`}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            RAM:
          </Typography>
          <Box sx={{ 
            width: 60, 
            height: 6, 
            bgcolor: 'rgba(255,255,255,0.1)', 
            borderRadius: 3,
            overflow: 'hidden'
          }}>
            <Box sx={{ 
              width: `${hardware.memory_usage_percent}%`,
              height: '100%',
              bgcolor: hardware.memory_usage_percent > 80 ? 'error.main' : 
                      hardware.memory_usage_percent > 60 ? 'warning.main' : 'success.main',
              transition: 'width 0.3s ease'
            }} />
          </Box>
          <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 35 }}>
            {hardware.memory_usage_percent.toFixed(0)}%
          </Typography>
        </Box>
      </Tooltip>

      {/* Status Indicator */}
      <Tooltip title={isGPU ? 
        `GPU Accelerated (${hardware.gpu_name})` : 
        'CPU Processing (Install CUDA for GPU acceleration)'
      }>
        <CheckIcon sx={{ 
          fontSize: 16, 
          color: isGPU ? 'success.main' : 'info.main'
        }} />
      </Tooltip>
    </Box>
  );
}
