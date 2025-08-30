import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Box,
  IconButton,
  Tooltip,
  Chip,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Fade,
  Zoom,
} from '@mui/material';
import {
  VideoLibrary as VideoIcon,
  Help as HelpIcon,
  GitHub as GitHubIcon,
  AutoAwesome as AutoAwesomeIcon,
  Psychology as AiIcon,
  Code as CodeIcon,
  OpenInNew as OpenIcon,
  Person as PersonIcon,
} from '@mui/icons-material';
import HelpDialog from './HelpDialog';

export default function Header() {
  const [helpOpen, setHelpOpen] = useState(false);
  const [githubMenuAnchor, setGithubMenuAnchor] = useState<null | HTMLElement>(null);
  const [githubIconRotate, setGithubIconRotate] = useState(false);
  const [aboutMenuAnchor, setAboutMenuAnchor] = useState<null | HTMLElement>(null);

  const handleGithubClick = (event: React.MouseEvent<HTMLElement>) => {
    setGithubMenuAnchor(event.currentTarget);
    setGithubIconRotate(true);
    setTimeout(() => setGithubIconRotate(false), 600);
  };

  const handleGithubClose = () => {
    setGithubMenuAnchor(null);
  };

  const handleAboutClick = (event: React.MouseEvent<HTMLElement>) => {
    setAboutMenuAnchor(event.currentTarget);
  };

  const handleAboutClose = () => {
    setAboutMenuAnchor(null);
  };

  const openLink = (url: string) => {
    window.open(url, '_blank');
    handleGithubClose();
  };

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
          <Tooltip title="About Me">
            <IconButton
              onClick={handleAboutClick}
              sx={{
                color: 'text.secondary',
                '&:hover': {
                  color: 'primary.main',
                  background: 'rgba(0, 180, 216, 0.1)',
                },
              }}
            >
              <PersonIcon />
            </IconButton>
          </Tooltip>

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
          
          <Tooltip title="GitHub Repositories">
            <IconButton
              sx={{
                color: 'text.secondary',
                transition: 'all 0.3s ease',
                '&:hover': {
                  color: 'primary.main',
                  background: 'rgba(0, 180, 216, 0.1)',
                  transform: 'scale(1.1)',
                },
                animation: githubIconRotate ? 'rotateIcon 0.6s ease-in-out' : 'none',
                '@keyframes rotateIcon': {
                  '0%': { transform: 'rotate(0deg) scale(1)' },
                  '50%': { transform: 'rotate(180deg) scale(1.2)' },
                  '100%': { transform: 'rotate(360deg) scale(1)' },
                },
              }}
              onClick={handleGithubClick}
            >
              <GitHubIcon />
            </IconButton>
          </Tooltip>

          <Menu
            anchorEl={githubMenuAnchor}
            open={Boolean(githubMenuAnchor)}
            onClose={handleGithubClose}
            TransitionComponent={Fade}
            transitionDuration={300}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            PaperProps={{
              sx: {
                mt: 1,
                background: 'rgba(26, 31, 58, 0.95)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(0, 180, 216, 0.3)',
                boxShadow: '0 8px 32px rgba(0, 180, 216, 0.2)',
                minWidth: 280,
                '& .MuiMenuItem-root': {
                  py: 1.5,
                  px: 2,
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    background: 'rgba(0, 180, 216, 0.1)',
                    transform: 'translateX(4px)',
                  },
                },
              },
            }}
          >
            <Box sx={{ px: 2, py: 1.5 }}>
              <Typography variant="subtitle2" sx={{ 
                color: 'primary.main', 
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 1 
              }}>
                <GitHubIcon sx={{ fontSize: 18 }} />
                GitHub Repositories
              </Typography>
            </Box>
            
            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)' }} />
            
            <Zoom in={Boolean(githubMenuAnchor)} style={{ transitionDelay: '100ms' }}>
              <MenuItem onClick={() => openLink('https://github.com/ZhengPeng7/BiRefNet')}>
                <ListItemIcon>
                  <AiIcon sx={{ color: 'warning.main' }} />
                </ListItemIcon>
                <ListItemText 
                  primary={
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      BiRefNet AI Model
                    </Typography>
                  }
                  secondary={
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      The AI model powering this app
                    </Typography>
                  }
                />
                <OpenIcon sx={{ fontSize: 16, color: 'text.secondary', ml: 1 }} />
              </MenuItem>
            </Zoom>
            
            <Zoom in={Boolean(githubMenuAnchor)} style={{ transitionDelay: '200ms' }}>
              <MenuItem onClick={() => openLink('https://github.com/RohitPoul/Background-Remover-Ai')}>
                <ListItemIcon>
                  <CodeIcon sx={{ color: 'success.main' }} />
                </ListItemIcon>
                <ListItemText 
                  primary={
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      This Project
                    </Typography>
                  }
                  secondary={
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      Video Background Removal App
                    </Typography>
                  }
                />
                <OpenIcon sx={{ fontSize: 16, color: 'text.secondary', ml: 1 }} />
              </MenuItem>
            </Zoom>

            <Divider sx={{ borderColor: 'rgba(255,255,255,0.1)', mt: 1 }} />
            
            <Box sx={{ px: 2, py: 1, mt: 1 }}>
              <Typography variant="caption" sx={{ 
                color: 'text.secondary',
                display: 'flex',
                alignItems: 'center',
                gap: 0.5
              }}>
                <AutoAwesomeIcon sx={{ fontSize: 12 }} />
                Powered by AI & Open Source
              </Typography>
            </Box>
          </Menu>

          {/* About Menu */}
          <Menu
            anchorEl={aboutMenuAnchor}
            open={Boolean(aboutMenuAnchor)}
            onClose={handleAboutClose}
            TransitionComponent={Fade}
            transitionDuration={300}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            PaperProps={{
              sx: {
                mt: 1,
                background: 'rgba(26, 31, 58, 0.95)',
                backdropFilter: 'blur(20px)',
                border: '1px solid rgba(0, 180, 216, 0.3)',
                boxShadow: '0 8px 32px rgba(0, 180, 216, 0.2)',
                minWidth: 300,
                maxWidth: 400,
                p: 2,
              },
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
              <PersonIcon sx={{ color: 'primary.main', mr: 1.5, fontSize: 28 }} />
              <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main' }}>
                About Me
              </Typography>
            </Box>
            
            <Divider sx={{ borderColor: 'rgba(0, 180, 216, 0.2)', mb: 2 }} />
            
            <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.8 }}>
              {/* Content will be added here later */}
              Coming soon...
            </Typography>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
    </>
  );
}