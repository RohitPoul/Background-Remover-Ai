import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Typography,
  Paper,
  LinearProgress,
  IconButton,
  Chip,
  Fade,
  Zoom,
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Movie as MovieIcon,
  Delete as DeleteIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function VideoUpload() {
  const { uploadedVideo, setUploadedVideo } = useVideoProcessor();
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setIsUploading(true);
      setUploadProgress(0);
      
      // Simulate upload progress
      const interval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setIsUploading(false);
            return 100;
          }
          return prev + 10;
        });
      }, 100);

      setTimeout(() => {
        setUploadedVideo(file);
        setIsUploading(false);
        setUploadProgress(100);
      }, 1000);
    }
  }, [setUploadedVideo]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.webm'],
    },
    maxFiles: 1,
    disabled: isUploading,
  });

  const handleRemoveVideo = () => {
    setUploadedVideo(null);
    setUploadProgress(0);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  if (uploadedVideo && !isUploading) {
    return (
      <Zoom in timeout={300}>
        <Paper
          sx={{
            p: 3,
            background: 'linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 119, 182, 0.05) 100%)',
            border: '2px solid rgba(0, 180, 216, 0.3)',
            borderRadius: 2,
            position: 'relative',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Box
              sx={{
                width: 60,
                height: 60,
                borderRadius: 2,
                background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <MovieIcon sx={{ fontSize: 32, color: 'white' }} />
            </Box>
            
            <Box sx={{ flexGrow: 1 }}>
              <Typography
                variant="subtitle1"
                sx={{
                  fontWeight: 600,
                  color: 'text.primary',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                {uploadedVideo.name}
                <CheckIcon sx={{ color: 'success.main', fontSize: 20 }} />
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                <Chip
                  label={`Type: ${uploadedVideo.type.split('/')[1]?.toUpperCase() || 'VIDEO'}`}
                  size="small"
                  sx={{
                    background: 'rgba(0, 180, 216, 0.2)',
                    border: '1px solid rgba(0, 180, 216, 0.3)',
                  }}
                />
                <Chip
                  label={`Size: ${formatFileSize(uploadedVideo.size)}`}
                  size="small"
                  sx={{
                    background: 'rgba(0, 180, 216, 0.2)',
                    border: '1px solid rgba(0, 180, 216, 0.3)',
                  }}
                />
              </Box>
            </Box>
            
            <IconButton
              onClick={handleRemoveVideo}
              sx={{
                color: 'error.main',
                '&:hover': {
                  background: 'rgba(239, 68, 68, 0.1)',
                },
              }}
            >
              <DeleteIcon />
            </IconButton>
          </Box>
        </Paper>
      </Zoom>
    );
  }

  return (
    <Box>
      <Paper
        {...getRootProps()}
        sx={{
          p: 4,
          textAlign: 'center',
          cursor: isUploading ? 'not-allowed' : 'pointer',
          border: isDragActive
            ? '2px dashed #00f5ff'
            : '2px dashed rgba(0, 180, 216, 0.3)',
          borderRadius: 2,
          background: isDragActive
            ? 'rgba(0, 180, 216, 0.05)'
            : 'transparent',
          transition: 'all 0.3s ease',
          position: 'relative',
          overflow: 'hidden',
          '&:hover': {
            border: '2px dashed rgba(0, 180, 216, 0.5)',
            background: 'rgba(0, 180, 216, 0.02)',
          },
        }}
      >
        <input {...getInputProps()} />
        
        {isUploading ? (
          <Fade in>
            <Box>
              <Box
                sx={{
                  width: 60,
                  height: 60,
                  borderRadius: '50%',
                  background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  mb: 2,
                  animation: 'pulse 2s ease-in-out infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.1)' },
                  },
                }}
              >
                <UploadIcon sx={{ fontSize: 32, color: 'white' }} />
              </Box>
              <Typography variant="h6" sx={{ mb: 1, fontWeight: 600 }}>
                Uploading...
              </Typography>
              <Box sx={{ width: '100%', mt: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={uploadProgress}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    '& .MuiLinearProgress-bar': {
                      background: 'linear-gradient(90deg, #00b4d8 0%, #00f5ff 100%)',
                    },
                  }}
                />
                <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                  {uploadProgress}% Complete
                </Typography>
              </Box>
            </Box>
          </Fade>
        ) : (
          <Box>
            <Box
              sx={{
                width: 80,
                height: 80,
                borderRadius: '50%',
                background: isDragActive
                  ? 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)'
                  : 'rgba(0, 180, 216, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto',
                mb: 2,
                transition: 'all 0.3s ease',
              }}
            >
              <UploadIcon
                sx={{
                  fontSize: 40,
                  color: isDragActive ? 'white' : 'primary.main',
                }}
              />
            </Box>
            
            <Typography
              variant="h6"
              sx={{
                mb: 1,
                fontWeight: 600,
                color: isDragActive ? 'primary.main' : 'text.primary',
              }}
            >
              {isDragActive ? 'Drop your video here' : 'Drag & drop your video'}
            </Typography>
            
            <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2 }}>
              or click to browse files
            </Typography>
            
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Supports MP4, MOV, AVI, WebM (max 500MB)
            </Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
}