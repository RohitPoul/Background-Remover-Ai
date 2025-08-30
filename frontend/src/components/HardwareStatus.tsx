import React, { useEffect, useState } from 'react';
import { Box, Chip, Typography, Tooltip, CircularProgress } from '@mui/material';
import { 
  Memory as CpuIcon, 
  GraphicEq as GpuIcon,
  Speed as SpeedIcon,
  CheckCircle as CheckIcon,
  Warning as WarningIcon 
} from '@mui/icons-material';

const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');

interface HardwareInfo {
  processing_device: string;
  gpu_name: string;
  gpu_available: boolean;
  gpu_vram?: {
    vram_total_gb: number;
    vram_allocated_gb: number;
    vram_reserved_gb: number;
    vram_free_gb: number;
    vram_usage_percent: number;
  };
  gpu_utilization?: number;
  gpu_temperature?: number;
  cpu_cores: number;
  cpu_threads: number;
  cpu_usage_percent?: number;
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
  const [isInitializing, setIsInitializing] = useState(true);
  const [backendConnected, setBackendConnected] = useState(false);
  const [updateCounter, setUpdateCounter] = useState(0);

  // First, check if backend is available
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        const response = await fetch(`${API_BASE}/api/health`, {
          signal: AbortSignal.timeout(2000)
        });
        if (response.ok) {
          setBackendConnected(true);
          setIsInitializing(false);
        }
      } catch {
        // Backend not ready yet
      }
    };

    // Check backend connection every 3 seconds until connected
    const connectionInterval = setInterval(() => {
      if (!backendConnected) {
        checkBackendConnection();
      }
    }, 3000);

    // Initial check
    checkBackendConnection();

    // Give backend 5 seconds to start before showing error
    const initTimer = setTimeout(() => {
      setIsInitializing(false);
    }, 5000);

    return () => {
      clearInterval(connectionInterval);
      clearTimeout(initTimer);
    };
  }, [backendConnected]);

  // Only fetch hardware status after backend is connected
  useEffect(() => {
    if (!backendConnected) return;

    // Fetch immediately when backend connects
    fetchHardwareStatus();
    
    // Retry logic for hardware status
    const retryInterval = setInterval(() => {
      if (error && retryCount < 10) {
        setRetryCount(prev => prev + 1);
        fetchHardwareStatus();
      }
    }, 2000);
    
    // Refresh every 2 seconds for real-time monitoring
    const refreshInterval = setInterval(() => {
      if (!error) {
        fetchHardwareStatus();
        setUpdateCounter(prev => prev + 1);
      }
    }, 2000);
    
    return () => {
      clearInterval(retryInterval);
      clearInterval(refreshInterval);
    };
  }, [backendConnected, error, retryCount]);

  const fetchHardwareStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/hardware_status`, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      if (response.ok) {
        const data = await response.json();
        // Extract hardware and settings from the response
        if (data.hardware) {
          setHardware(data.hardware);
        }
        if (data.settings) {
          setSettings(data.settings);
        }
        setError(null);
        setRetryCount(0); // Reset retry count on success
      } else {
        // Only set error if not initializing
        if (!isInitializing) {
          setError('Failed to fetch hardware status');
        }
      }
    } catch (err) {
      // Only set error if not initializing and haven't retried too many times
      if (!isInitializing && retryCount > 2) {
        setError('Cannot connect to backend');
      }
    } finally {
      setLoading(false);
    }
  };

  if (loading || isInitializing) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1 }}>
        <CircularProgress size={16} />
        <Typography variant="caption">
          {!backendConnected ? 'Connecting to backend...' : 'Detecting hardware...'}
        </Typography>
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
            {!backendConnected ? 'Waiting for backend...' : 'Backend not responding'}
          </Typography>
          <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.7rem', mt: 0.5 }}>
            {!backendConnected 
              ? 'Starting Python server...' 
              : 'Backend connected but hardware status unavailable'}
          </Typography>
        </Box>
      </Box>
    );
  }

  const isGPU = hardware.processing_device === 'CUDA' || hardware.processing_device === 'MPS';
  const deviceColor = isGPU ? 'success' : 'warning';
  const deviceIcon = isGPU ? <GpuIcon /> : <CpuIcon />;
  const speedLevel = settings?.optimization_profile || 'detecting';
  const profileName = speedLevel === 'unknown' ? 'Custom' : speedLevel;
  
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
      case 'unknown':
      case 'detecting':
      case 'custom':
        return 'default';
      default:
        return 'default';
    }
  };
  
  const getUtilizationColor = (percent: number) => {
    if (percent > 90) return 'error.main';
    if (percent > 70) return 'warning.main';
    return 'success.main';
  };

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column',
      gap: 1.5,
      p: 1.5,
      bgcolor: 'rgba(0,0,0,0.2)',
      borderRadius: 1,
      border: '1px solid rgba(255,255,255,0.1)'
    }}>
      {/* Top row - Device and Profile */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        {/* Processing Device */}
        <Tooltip title={
          <Box>
            <Typography variant="body2">
              {isGPU ? `GPU: ${hardware.gpu_name}` : `CPU: ${hardware.cpu_cores} cores, ${hardware.cpu_threads} threads`}
            </Typography>
            {isGPU && hardware.gpu_temperature ? (
              <Typography variant="caption">
                Temperature: {hardware.gpu_temperature}°C
              </Typography>
            ) : null}
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
            <Typography variant="body2">Optimization Profile: {profileName}</Typography>
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
            label={profileName.toUpperCase()}
            color={getPerformanceColor()}
            size="small"
            variant="outlined"
          />
        </Tooltip>

        {/* Status Indicator */}
        <Box sx={{ flex: 1 }} />
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

      {/* Performance Metrics Row */}
      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {/* GPU Usage (if GPU available) */}
        {isGPU && (
          <>
            {/* GPU Core Utilization */}
            {hardware.gpu_utilization !== undefined && hardware.gpu_utilization !== 0 && (
              <Tooltip title={`GPU Core Utilization: ${hardware.gpu_utilization}%`}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 110 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                    GPU:
                  </Typography>
                  <Box sx={{ 
                    flex: 1,
                    height: 6, 
                    bgcolor: 'rgba(255,255,255,0.1)', 
                    borderRadius: 3,
                    overflow: 'hidden'
                  }}>
                    <Box sx={{ 
                      width: `${hardware.gpu_utilization}%`,
                      height: '100%',
                      bgcolor: getUtilizationColor(hardware.gpu_utilization),
                      transition: 'width 0.3s ease'
                    }} />
                  </Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 35, fontSize: '0.7rem' }}>
                    {hardware.gpu_utilization.toFixed(0)}%
                  </Typography>
                </Box>
              </Tooltip>
            )}

            {/* VRAM Usage */}
            {hardware.gpu_vram && (
              <Tooltip title={
                <Box>
                  <Typography variant="body2">Video Memory</Typography>
                  <Typography variant="caption">
                    Used: {hardware.gpu_vram.vram_allocated_gb.toFixed(1)} GB<br />
                    Total: {hardware.gpu_vram.vram_total_gb.toFixed(1)} GB<br />
                    Free: {hardware.gpu_vram.vram_free_gb.toFixed(1)} GB
                  </Typography>
                </Box>
              }>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 110 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                    VRAM:
                  </Typography>
                  <Box sx={{ 
                    flex: 1,
                    height: 6, 
                    bgcolor: 'rgba(255,255,255,0.1)', 
                    borderRadius: 3,
                    overflow: 'hidden'
                  }}>
                    <Box sx={{ 
                      width: `${hardware.gpu_vram.vram_usage_percent}%`,
                      height: '100%',
                      bgcolor: getUtilizationColor(hardware.gpu_vram.vram_usage_percent),
                      transition: 'width 0.3s ease'
                    }} />
                  </Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 35, fontSize: '0.7rem' }}>
                    {hardware.gpu_vram.vram_usage_percent.toFixed(0)}%
                  </Typography>
                </Box>
              </Tooltip>
            )}
          </>
        )}

        {/* CPU Usage */}
        {hardware.cpu_usage_percent !== undefined && (
          <Tooltip title={`CPU Usage: ${hardware.cpu_usage_percent.toFixed(1)}%`}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 110 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
                CPU:
              </Typography>
              <Box sx={{ 
                flex: 1,
                height: 6, 
                bgcolor: 'rgba(255,255,255,0.1)', 
                borderRadius: 3,
                overflow: 'hidden'
              }}>
                <Box sx={{ 
                  width: `${hardware.cpu_usage_percent}%`,
                  height: '100%',
                  bgcolor: getUtilizationColor(hardware.cpu_usage_percent),
                  transition: 'width 0.3s ease'
                }} />
              </Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 35, fontSize: '0.7rem' }}>
                {hardware.cpu_usage_percent.toFixed(0)}%
              </Typography>
            </Box>
          </Tooltip>
        )}

        {/* System RAM Usage */}
        <Tooltip title={
          <Box>
            <Typography variant="body2">System Memory</Typography>
            <Typography variant="caption">
              Available: {hardware.memory_available_gb.toFixed(1)} GB<br />
              Total: {hardware.memory_gb.toFixed(1)} GB
            </Typography>
          </Box>
        }>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, minWidth: 110 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
              RAM:
            </Typography>
            <Box sx={{ 
              flex: 1,
              height: 6, 
              bgcolor: 'rgba(255,255,255,0.1)', 
              borderRadius: 3,
              overflow: 'hidden'
            }}>
              <Box sx={{ 
                width: `${hardware.memory_usage_percent}%`,
                height: '100%',
                bgcolor: getUtilizationColor(hardware.memory_usage_percent),
                transition: 'width 0.3s ease'
              }} />
            </Box>
            <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 35, fontSize: '0.7rem' }}>
              {hardware.memory_usage_percent.toFixed(0)}%
            </Typography>
          </Box>
        </Tooltip>
      </Box>
    </Box>
  );
}
