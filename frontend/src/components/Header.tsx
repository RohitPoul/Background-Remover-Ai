import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  IconButton,
  Tooltip,
  Chip,
} from '@mui/material';
import {
  VideoLibrary as VideoIcon,
  Help as HelpIcon,
  GitHub as GitHubIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import HelpDialog from './HelpDialog';

export default function Header() {
  const [helpOpen, setHelpOpen] = useState(false);

  return (
    <>
      <HelpDialog open={helpOpen} onClose={() => setHelpOpen(false)} />
    <AppBar
      position="static"
      sx={{
        background: 'rgba(26, 31, 58, 0.9)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(0, 180, 216, 0.2)',
        boxShadow: '0 4px 30px rgba(0, 180, 216, 0.1)',
      }}
    >
      <Toolbar sx={{ py: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <VideoIcon
            sx={{
              fontSize: 36,
              mr: 2,
              color: 'primary.main',
              filter: 'drop-shadow(0 0 10px rgba(0, 180, 216, 0.5))',
            }}
          />
          <Box>
            <Typography
              variant="h5"
              component="h1"
              sx={{
                fontWeight: 800,
                background: 'linear-gradient(135deg, #00b4d8 0%, #00f5ff 100%)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
                letterSpacing: '-0.5px',
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              Video Background Remover
              <Chip
                icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
                label="AI Powered"
                size="small"
                sx={{
                  ml: 1,
                  background: 'linear-gradient(135deg, #00b4d8 0%, #0077b6 100%)',
                  '& .MuiChip-icon': { color: 'white' },
                  fontSize: '0.75rem',
                  height: 24,
                  fontWeight: 600,
                }}
              />
            </Typography>
            <Typography
              variant="caption"
              sx={{
                color: 'text.secondary',
                display: 'block',
                mt: 0.5,
                fontSize: '0.85rem',
              }}
            >
              Professional AI-Powered Background Processing
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Tooltip title="Help & How to Use">
            <IconButton
              onClick={() => setHelpOpen(true)}
              sx={{
                color: 'text.secondary',
                '&:hover': {
                  color: 'primary.main',
                  background: 'rgba(0, 180, 216, 0.1)',
                },
              }}
            >
              <HelpIcon />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="GitHub Repository">
            <IconButton
              sx={{
                color: 'text.secondary',
                '&:hover': {
                  color: 'primary.main',
                  background: 'rgba(0, 180, 216, 0.1)',
                },
              }}
              onClick={() => window.open('https://github.com', '_blank')}
            >
              <GitHubIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Toolbar>
    </AppBar>
    </>
  );
}