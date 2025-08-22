import React, { useEffect, useRef, useState } from 'react';
import { Box, Typography, Paper, Fade, CircularProgress, Chip, Grid, Button, ButtonGroup, Tooltip } from '@mui/material';
import { 
  CheckCircle as CheckIcon, 
  Download as DownloadIcon,

} from '@mui/icons-material';
import { useVideoProcessor } from '../context/VideoProcessorContext';

export default function DualVideoPreview() {
  const {
    uploadedVideo,
    previewImage,
    processingStatus,
    outputFile,
    downloadVideo,
    backgroundType,
    outputFormat,
    progress,
    currentFrame,
    totalFrames,
    debugInfo,
  } = useVideoProcessor();
  
  // Track download attempts for debugging
  const [downloadAttempts, setDownloadAttempts] = useState(0);

  const originalVideoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (uploadedVideo && originalVideoRef.current) {
      const url = URL.createObjectURL(uploadedVideo);
      originalVideoRef.current.src = url;
      originalVideoRef.current.play().catch(() => {});
      return () => URL.revokeObjectURL(url);
    }
  }, [uploadedVideo]);

  const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
  const [processedBlobUrl, setProcessedBlobUrl] = useState<string>('');
  const processedUrl = outputFile ? `${API_BASE}${outputFile}` : '';
  
  // Convert processed video to blob URL to avoid CSP issues
  useEffect(() => {
    if (processedUrl && processingStatus === 'completed') {
      console.log('🎬 Fetching processed video as blob:', processedUrl);
      console.log('📦 Output format:', outputFormat);
      console.log('🎨 Background type:', backgroundType);
      
      // Fetch the video as blob and create a blob URL
      fetch(processedUrl)
        .then(response => {
          if (!response.ok) throw new Error(`Failed to fetch: ${response.status}`);
          return response.blob();
        })
        .then(blob => {
          const blobUrl = URL.createObjectURL(blob);
          console.log('✅ Blob URL created for preview:', blobUrl);
          console.log('📏 Blob size:', blob.size, 'bytes');
          setProcessedBlobUrl(blobUrl);
        })
        .catch(error => {
          console.error('❌ Failed to create blob URL:', error);
        });
    }
    
    // Cleanup blob URL when component unmounts or URL changes
    return () => {
      if (processedBlobUrl) {
        URL.revokeObjectURL(processedBlobUrl);
      }
    };
  }, [processedUrl, processingStatus, outputFormat, backgroundType]);

  const Header = () => {
    if (processingStatus === 'processing' || processingStatus === 'started') {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <CircularProgress size={20} sx={{ color: 'warning.main' }} />
          <Typography variant="body2" sx={{ fontWeight: 600, color: 'warning.main' }}>
            Processing Frame {currentFrame} of {totalFrames}
          </Typography>
          <Chip label={`${progress.toFixed(1)}%`} color="warning" size="small" sx={{ ml: 'auto', fontWeight: 600 }} />
        </Box>
      );
    }
    if (processingStatus === 'completed' && processedUrl) {
      return (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip icon={<CheckIcon />} label="Processing Complete" color="success" sx={{ fontWeight: 600 }} />
          
          {/* Debug info display */}
          {downloadAttempts > 0 && (
            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.7rem' }}>
              Attempts: {downloadAttempts}
            </Typography>
          )}
          
          <Button
            variant="contained"
            size="small"
            startIcon={<DownloadIcon />}
            onClick={() => {
              setDownloadAttempts(prev => prev + 1);
              downloadVideo();
            }}
            color="primary"
            data-debug-label="download-video"
            sx={{ ml: 'auto' }}
          >
            Download
          </Button>
        </Box>
      );
    }
    if (uploadedVideo) {
      return (
        <Typography variant="body2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          Video Preview - Ready to Process
        </Typography>
      );
    }
    return null;
  };

  // Always render a dual layout to avoid confusion and ensure visibility
  if (!uploadedVideo && processingStatus === 'idle') {
    return (
      <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 2 }}>
        <Typography variant="h6" sx={{ color: 'text.secondary', opacity: 0.5 }}>
          Video Preview
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary', opacity: 0.3 }}>
          Upload a video to see preview
        </Typography>
      </Box>
    );
  }

  return (
    <Fade in timeout={500}>
      <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 2, minHeight: 0 }}>
        <Header />
        <Grid container spacing={2} sx={{ flex: 1, minHeight: 0 }}>
          {/* Original */}
          <Grid item xs={6} sx={{ minHeight: 0 }}>
            <Paper sx={{ height: '100%', p: 1, bgcolor: 'rgba(0,0,0,0.2)', display: 'flex', flexDirection: 'column' }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
                Original
              </Typography>
              {uploadedVideo ? (
                <video ref={originalVideoRef} muted={processingStatus !== 'completed'} loop autoPlay controls
                  style={{ width: '100%', height: '100%', objectFit: 'contain', flex: 1 }} />
              ) : (
                <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>No video loaded</Typography>
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Processed / Live Preview */}
          <Grid item xs={6} sx={{ minHeight: 0 }}>
            <Paper sx={{ height: '100%', p: 1, bgcolor: 'rgba(0,0,0,0.2)', position: 'relative', display: 'flex', flexDirection: 'column' }}>
              <Typography
                variant="caption"
                sx={{ color: processingStatus === 'completed' ? 'success.main' : processingStatus === 'processing' ? 'warning.main' : 'text.secondary', mb: 1, display: 'block' }}
              >
                {processingStatus === 'completed' ? 'Processed' : processingStatus === 'processing' ? 'Live Preview' : 'Output'}
              </Typography>

              {processingStatus === 'completed' && processedBlobUrl && (
                <Box sx={{ width: '100%', height: '100%', flex: 1, position: 'relative', 
                  background: backgroundType === 'Transparent' ? 
                    'repeating-conic-gradient(#808080 0% 25%, transparent 0% 50%) 50% / 20px 20px' : 
                    'transparent' }}>
                  <video
                    src={processedBlobUrl}
                    controls
                    autoPlay
                    loop
                    muted
                    style={{ 
                      width: '100%', 
                      height: '100%', 
                      objectFit: 'contain',
                      position: 'absolute',
                      top: 0,
                      left: 0
                    }}
                    onLoadedData={() => {
                      console.log('✅ Processed video loaded successfully');
                    }}
                    onError={(e) => {
                      console.error('❌ Video playback error:', e);
                      console.log('Blob URL:', processedBlobUrl);
                      console.log('Output format:', outputFormat);
                    }}
                  />
                </Box>
              )}

              {processingStatus === 'processing' && (
                previewImage ? (
                  <img src={previewImage} alt="Processing preview" style={{ width: '100%', height: '100%', objectFit: 'contain', flex: 1 }} />
                ) : (
                  <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <CircularProgress />
                  </Box>
                )
              )}

              {processingStatus === 'idle' && (
                <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>No output yet</Typography>
                </Box>
              )}

              {processingStatus === 'processing' && (
                <Paper sx={{ position: 'absolute', top: 32, right: 8, px: 1.5, py: 0.5, background: 'rgba(255, 152, 0, 0.9)', backdropFilter: 'blur(10px)', display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'white', animation: 'pulse 2s infinite', '@keyframes pulse': { '0%': { opacity: 1 }, '50%': { opacity: 0.3 }, '100%': { opacity: 1 } } }} />
                  <Typography variant="caption" sx={{ fontWeight: 600, color: 'white' }}>LIVE</Typography>
                </Paper>
              )}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Fade>
  );
}
