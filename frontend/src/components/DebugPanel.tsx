import React, { useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  IconButton,
  List,
  ListItem,
  ListItemText,
  Divider,
  Tooltip,
} from '@mui/material';
import {
  Close as CloseIcon,
  Delete as DeleteIcon,
  BugReport as BugIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  ContentCopy as CopyIcon,
  CleaningServices as CleanupIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function DebugPanel() {
  const {
    debugLogs,
    showDebugPanel,
    toggleDebugPanel,
    clearDebugLogs,
    sessionId,
  } = useVideoProcessor();
  
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logsEndRef.current && showDebugPanel) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [debugLogs, showDebugPanel]);
  
  // Format timestamp
  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };
  
  // Export logs as text file
  const exportLogs = () => {
    const logText = debugLogs.map(log => 
      `[${formatTime(log.timestamp)}] [${log.sessionId}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `debug-logs-${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };
  
  // Copy all logs to clipboard
  const copyAllLogs = () => {
    const logText = debugLogs.map(log => 
      `[${formatTime(log.timestamp)}] [${log.sessionId}] ${log.message}`
    ).join('\n');
    
    navigator.clipboard.writeText(logText).then(() => {
      alert('Debug logs copied to clipboard!');
    }).catch(err => {
      console.error('Failed to copy logs:', err);
      alert('Failed to copy logs to clipboard');
    });
  };
  
  // Manual cleanup of processed files
  const cleanupProcessedFiles = async () => {
    try {
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      const response = await fetch(`${API_BASE}/api/cleanup_files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`Cleanup complete! Removed ${result.files_removed?.length || 0} processed video files.`);
      } else {
        alert('Cleanup failed. Check debug console for details.');
      }
    } catch (error) {
      console.error('Cleanup error:', error);
      alert('Failed to cleanup files. Check debug console for details.');
    }
  };
  
  if (!showDebugPanel) return null;
  
  return (
    <Paper
      sx={{
        position: 'fixed',
        bottom: 0,
        right: 0,
        width: '500px',
        height: '300px',
        display: 'flex',
        flexDirection: 'column',
        zIndex: 9999,
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(0, 180, 216, 0.5)',
        boxShadow: '0 0 20px rgba(0, 0, 0, 0.5)',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          p: 1,
          borderBottom: '1px solid rgba(0, 180, 216, 0.5)',
          backgroundColor: 'rgba(0, 180, 216, 0.2)',
        }}
      >
        <BugIcon sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="subtitle2" sx={{ fontWeight: 600, flexGrow: 1 }}>
          Debug Console {sessionId && `(Session: ${sessionId.substring(0, 8)}...)`}
        </Typography>
        <Tooltip title="Refresh page to reconnect">
          <IconButton size="small" onClick={() => window.location.reload()} sx={{ color: 'success.main' }}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Copy all logs">
          <IconButton size="small" onClick={copyAllLogs} sx={{ color: 'secondary.main' }}>
            <CopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Export logs">
          <IconButton size="small" onClick={exportLogs} sx={{ color: 'primary.main' }}>
            <DownloadIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Clear logs">
          <IconButton size="small" onClick={clearDebugLogs} sx={{ color: 'warning.main' }}>
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Cleanup processed files">
          <IconButton size="small" onClick={cleanupProcessedFiles} sx={{ color: 'error.main' }}>
            <CleanupIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Close">
          <IconButton size="small" onClick={toggleDebugPanel} sx={{ color: 'error.main' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      {/* Log content */}
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          p: 1,
          fontFamily: 'monospace',
          fontSize: '0.75rem',
        }}
      >
        {debugLogs.length === 0 ? (
          <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', mt: 2 }}>
            No logs yet. Processing events will appear here.
          </Typography>
        ) : (
          <List dense disablePadding>
            {debugLogs.map((log, index) => (
              <React.Fragment key={`${log.timestamp}-${index}`}>
                <ListItem disablePadding sx={{ py: 0.25 }}>
                  <ListItemText
                    primary={
                      <Box component="span" sx={{ display: 'flex', alignItems: 'flex-start' }}>
                        <Typography
                          component="span"
                          variant="caption"
                          sx={{ color: 'primary.main', minWidth: '60px', mr: 1 }}
                        >
                          {formatTime(log.timestamp)}
                        </Typography>
                        <Typography
                          component="span"
                          variant="caption"
                          sx={{ 
                            color: log.message.includes('ERROR') || log.message.includes('failed') ? 
                              'error.main' : 'text.primary',
                            wordBreak: 'break-word'
                          }}
                        >
                          {log.message}
                        </Typography>
                      </Box>
                    }
                  />
                </ListItem>
                {index < debugLogs.length - 1 && (
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.05)' }} />
                )}
              </React.Fragment>
            ))}
            <div ref={logsEndRef} />
          </List>
        )}
      </Box>
    </Paper>
  );
}
