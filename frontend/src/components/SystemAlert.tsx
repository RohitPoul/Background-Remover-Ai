import React, { useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Alert,
  AlertTitle,
  Snackbar,
  Box,
  Typography,
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function SystemAlert() {
  const { systemWarning } = useVideoProcessor();
  const [open, setOpen] = React.useState(false);

  useEffect(() => {
    if (systemWarning) {
      setOpen(true);
    }
  }, [systemWarning]);

  const handleClose = () => {
    setOpen(false);
  };

  if (!systemWarning) return null;

  const getIcon = () => {
    switch (systemWarning.severity) {
      case 'error':
        return <ErrorIcon sx={{ fontSize: 48, color: 'error.main' }} />;
      case 'warning':
        return <WarningIcon sx={{ fontSize: 48, color: 'warning.main' }} />;
      default:
        return <InfoIcon sx={{ fontSize: 48, color: 'info.main' }} />;
    }
  };

  const getSeverityColor = () => {
    switch (systemWarning.severity) {
      case 'error':
        return 'error.main';
      case 'warning':
        return 'warning.main';
      default:
        return 'info.main';
    }
  };

  // For critical errors (like insufficient memory), show a dialog
  if (systemWarning.type === 'insufficient_memory' || systemWarning.type === 'critical_memory') {
    return (
      <Dialog
        open={open}
        onClose={systemWarning.severity !== 'error' ? handleClose : undefined}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            bgcolor: 'background.paper',
            backgroundImage: 'none',
            border: '2px solid',
            borderColor: getSeverityColor(),
          }
        }}
      >
        <DialogTitle sx={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: 2,
          bgcolor: `${getSeverityColor()}.dark`,
          color: 'white'
        }}>
          {getIcon()}
          <Typography variant="h6">{systemWarning.title}</Typography>
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <DialogContentText sx={{ color: 'text.primary', fontSize: '1.1rem' }}>
            {systemWarning.message}
          </DialogContentText>
          
          {systemWarning.type === 'insufficient_memory' && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'warning.dark', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ color: 'warning.contrastText', fontWeight: 600 }}>
                What you can do:
              </Typography>
              <Typography variant="body2" sx={{ color: 'warning.contrastText', mt: 1 }}>
                1. Close other applications (browsers, IDEs, etc.)<br />
                2. Restart your computer to free up memory<br />
                3. Check Task Manager to see what's using memory<br />
                4. Consider upgrading your RAM for better performance
              </Typography>
            </Box>
          )}
          
          {systemWarning.type === 'critical_memory' && (
            <Box sx={{ mt: 3, p: 2, bgcolor: 'error.dark', borderRadius: 1 }}>
              <Typography variant="subtitle2" sx={{ color: 'error.contrastText', fontWeight: 600 }}>
                Processing has been stopped to prevent system crash!
              </Typography>
              <Typography variant="body2" sx={{ color: 'error.contrastText', mt: 1 }}>
                Your system memory is critically low. Please:<br />
                1. Save any important work in other applications<br />
                2. Close unnecessary programs immediately<br />
                3. Try processing again with a smaller video or lower resolution
              </Typography>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button 
            onClick={handleClose} 
            variant="contained"
            color={systemWarning.severity === 'error' ? 'error' : 'warning'}
          >
            {systemWarning.severity === 'error' ? 'I Understand' : 'OK'}
          </Button>
        </DialogActions>
      </Dialog>
    );
  }

  // For warnings, show a snackbar
  return (
    <Snackbar
      open={open}
      autoHideDuration={systemWarning.severity === 'warning' ? 10000 : null}
      onClose={handleClose}
      anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
    >
      <Alert
        onClose={handleClose}
        severity={systemWarning.severity}
        sx={{
          width: '100%',
          maxWidth: 600,
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
          border: '1px solid',
          borderColor: getSeverityColor(),
          background: 'rgba(26, 31, 58, 0.95)',
          '& .MuiAlert-icon': {
            fontSize: 28
          }
        }}
      >
        <AlertTitle sx={{ fontWeight: 600 }}>{systemWarning.title}</AlertTitle>
        {systemWarning.message}
      </Alert>
    </Snackbar>
  );
}
