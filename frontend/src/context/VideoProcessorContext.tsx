import React, { createContext, useContext, useState, useCallback, useEffect, useRef, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';

// Types
interface DebugLogEntry {
  message: string;
  timestamp: number;
  sessionId: string;
}

interface VideoProcessorState {
  // Upload state
  uploadedVideo: File | null;
  
  // Processing state
  processingStatus: 'idle' | 'started' | 'processing' | 'completed' | 'error';
  sessionId: string | null;
  progress: number;
  elapsedTime: number;
  currentFrame: number;
  totalFrames: number;
  previewImage: string | null;
  outputFile: string | null;
  statusMessage: string | null;
  debugInfo?: any; // Debug information from backend
  debugLogs: DebugLogEntry[]; // Debug logs from backend
  showDebugPanel: boolean; // Whether to show debug panel
  
  // Connection state
  connectionError: string | null;
  
  // Settings
  backgroundType: 'Color' | 'Image' | 'Video' | 'Transparent';
  backgroundColor: string;
  backgroundImage: File | null;
  backgroundVideo: File | null;
  videoHandling: 'loop' | 'slow_down';
  fps: number;
  fastMode: boolean;
  maxWorkers: number;
  outputFormat: 'mp4' | 'webm' | 'mov';
}

interface VideoProcessorContextType extends VideoProcessorState {
  // Setters
  setUploadedVideo: (file: File | null) => void;
  setBackgroundType: (type: 'Color' | 'Image' | 'Video' | 'Transparent') => void;
  setBackgroundColor: (color: string) => void;
  setBackgroundImage: (file: File | null) => void;
  setBackgroundVideo: (file: File | null) => void;
  setVideoHandling: (handling: 'loop' | 'slow_down') => void;
  setFps: (fps: number) => void;
  setFastMode: (fastMode: boolean) => void;
  setMaxWorkers: (workers: number) => void;
  setOutputFormat: (format: 'mp4' | 'webm' | 'mov') => void;
  
  // Actions
  startProcessing: () => void;
  cancelProcessing: () => void;
  downloadVideo: () => void;

  resetState: () => void;
  toggleDebugPanel: () => void; // Toggle debug panel
  clearDebugLogs: () => void; // Clear debug logs
}

// Context
const VideoProcessorContext = createContext<VideoProcessorContextType | undefined>(undefined);

// Export context directly
export { VideoProcessorContext };

// Provider
export function VideoProcessorProvider({ children }: { children: ReactNode }) {
  // Initialize state
  const [state, setState] = useState<VideoProcessorState>({
    uploadedVideo: null,
    processingStatus: 'idle',
    sessionId: null,
    progress: 0,
    elapsedTime: 0,
    currentFrame: 0,
    totalFrames: 0,
    previewImage: null,
    outputFile: null,
    statusMessage: null,
    connectionError: null,
    backgroundType: 'Color',
    backgroundColor: '#00FF00',
    backgroundImage: null,
    backgroundVideo: null,
    videoHandling: 'slow_down',
    fps: 0,
    fastMode: true,
    maxWorkers: 2,
    outputFormat: 'mp4',
    debugLogs: [],
    showDebugPanel: true, // Show debug panel by default
  });

  // Socket connection - store as ref to prevent re-creation
  const socketRef = useRef<Socket | null>(null);
  const isConnecting = useRef<boolean>(false);
  // Track latest session id to avoid stale closures in socket handlers
  const sessionIdRef = useRef<string | null>(null);

  // Keep ref in sync with state
  useEffect(() => {
    console.log('🔄 UPDATING SESSION ID REF:', state.sessionId);
    sessionIdRef.current = state.sessionId;
  }, [state.sessionId]);

  // Add global click handler for debugging
  useEffect(() => {
    const handleGlobalClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const button = target.closest('button');
      if (button) {
        // Prefer a stable debug label if provided
        const debugLabel = button.getAttribute('data-debug-label');
        const buttonText = debugLabel || button.getAttribute('aria-label') || button.innerText || 'Unknown button';
        setState(prev => ({
          ...prev,
          debugLogs: [
            ...prev.debugLogs,
            {
              message: `🖱️ [UI] Button clicked: "${buttonText}"`,
              timestamp: Date.now() / 1000,
              sessionId: 'frontend'
            }
          ].slice(-100)
        }));
      }
    };
    
    document.addEventListener('click', handleGlobalClick);
    return () => document.removeEventListener('click', handleGlobalClick);
  }, []);
  
  // Initialize socket connection with retry logic
  useEffect(() => {
    // Prevent duplicate initialization
    if (socketRef.current || isConnecting.current) {
      return;
    }
    
    isConnecting.current = true;
    const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
    let connectionAttempts = 0;
    
    const initSocket = () => {
      // Create socket with polling-only transport to avoid WebSocket issues
      const newSocket = io(API_BASE, {
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 2000,
        reconnectionDelayMax: 10000,
        timeout: 20000,
        autoConnect: false,
        transports: ['polling'], // Use polling only to avoid WebSocket frame issues
        upgrade: false, // Disable transport upgrade
      });

      newSocket.on('connect', () => {
        console.log('Connected to Python server');
        connectionAttempts = 0;
        setState(prev => ({ 
          ...prev, 
          connectionError: null,
          debugLogs: [
            ...prev.debugLogs,
            {
              message: '✅ Connected to Python backend server',
              timestamp: Date.now() / 1000,
              sessionId: 'frontend'
            }
          ].slice(-100)
        }));
      });

      newSocket.on('connect_error', (err) => {
        connectionAttempts++;
        // Only show error after 3 attempts to avoid showing transient errors
        if (connectionAttempts > 3) {
          console.log('Connection error after', connectionAttempts, 'attempts:', err.message);
          setState(prev => ({ 
            ...prev, 
            connectionError: 'Waiting for Python server to start...' 
          }));
        }
      });

      newSocket.on('disconnect', (reason) => {
        console.log('Disconnected from server:', reason);
        // Only show error if it's not a normal disconnect
        if (reason !== 'io client disconnect' && reason !== 'io server disconnect') {
          setState(prev => ({ 
            ...prev, 
            connectionError: 'Connection lost. Attempting to reconnect...' 
          }));
        }
      });

      newSocket.on('processing_update', (data) => {
        if (data?.session_id !== sessionIdRef.current) return;
        setState(prev => ({
          ...prev,
          processingStatus: data.status === 'started' ? 'started' : 'processing',
          progress: data.progress ?? prev.progress,
          elapsedTime: data.elapsed_time ?? prev.elapsedTime,
          currentFrame: data.currentFrame ?? prev.currentFrame,
          totalFrames: data.totalFrames ?? prev.totalFrames,
          previewImage: data.preview_image ?? prev.previewImage,
          statusMessage: data.message ?? prev.statusMessage,
        }));
      });

      newSocket.on('processing_complete', (data) => {
        console.log('🎉 PROCESSING_COMPLETE EVENT RECEIVED:', data);
        console.log('📋 Session ID from server:', data?.session_id);
        console.log('📋 Current session ID ref:', sessionIdRef.current);
        console.log('🔍 Session IDs match:', data?.session_id === sessionIdRef.current);
        
        if (data?.session_id !== sessionIdRef.current) {
          console.log('❌ SESSION ID MISMATCH - BLOCKING COMPLETION EVENT!');
          console.log('   Server session:', data?.session_id);
          console.log('   Frontend session:', sessionIdRef.current);
          return;
        }
        
        console.log('✅ Session ID matches - processing completion event');
        // Debug logging for completion event
        console.log('DOWNLOAD DEBUG: Received processing_complete event', data);
        console.log('DOWNLOAD DEBUG: Output file from server:', data.output_file);
        console.log('DOWNLOAD DEBUG: Debug info:', data.debug_info);
        
        const downloadPath = `/api/download/${data.output_file}`;
        console.log('DOWNLOAD DEBUG: Setting download path to:', downloadPath);
        
        setState(prev => ({
          ...prev,
          processingStatus: 'completed',
          outputFile: downloadPath,
          elapsedTime: data.elapsed_time ?? prev.elapsedTime,
          statusMessage: data.message ?? 'Processing complete!',
          debugInfo: data.debug_info
        }));
      });

      newSocket.on('processing_error', (data) => {
        if (data?.session_id !== sessionIdRef.current) return;
        setState(prev => ({
          ...prev,
          processingStatus: 'error',
          connectionError: data?.message || 'Processing error occurred',
          statusMessage: data?.message || 'Processing error occurred',
        }));
      });
      
      // Handle cancellation notifications from backend
      newSocket.on('processing_cancelled', (data) => {
        if (data?.session_id !== sessionIdRef.current) return;
        setState(prev => ({
          ...prev,
          processingStatus: 'idle',
          statusMessage: data?.message || 'Processing cancelled',
          sessionId: null,
        }));
      });
      
      // Debug logs from backend
      newSocket.on('debug_log', (data) => {
        console.log('🔥 RECEIVED DEBUG LOG FROM BACKEND:', data);
        setState(prev => ({
          ...prev,
          debugLogs: [
            ...prev.debugLogs,
            {
              message: data.message,
              timestamp: data.timestamp || Date.now() / 1000,
              sessionId: data.session_id || 'unknown'
            }
          ].slice(-500) // Keep only last 500 logs
        }));
      });

      socketRef.current = newSocket;
      
      // Simple connection with delay
      const connectWithDelay = () => {
        // Try to connect after a short delay
        setTimeout(() => {
          if (!newSocket.connected) {
            console.log('Attempting to connect to server...');
            newSocket.connect();
          }
        }, 1500);
      };
      
      connectWithDelay();

      return newSocket;
    };

    // Initialize socket
    initSocket();

    // Cleanup function
    return () => {
      isConnecting.current = false;
      if (socketRef.current) {
        console.log('Cleaning up socket connection');
        socketRef.current.removeAllListeners();
        socketRef.current.disconnect();
        socketRef.current = null;
      }
    };
  }, []); // Empty dependency to run only once

  // Setters
  const setUploadedVideo = useCallback((file: File | null) => {
    setState(prev => ({ ...prev, uploadedVideo: file }));
  }, []);

  const setBackgroundType = useCallback((type: 'Color' | 'Image' | 'Video' | 'Transparent') => {
    setState(prev => ({ ...prev, backgroundType: type }));
  }, []);

  const setBackgroundColor = useCallback((color: string) => {
    setState(prev => ({ ...prev, backgroundColor: color }));
  }, []);

  const setBackgroundImage = useCallback((file: File | null) => {
    setState(prev => ({ ...prev, backgroundImage: file }));
  }, []);

  const setBackgroundVideo = useCallback((file: File | null) => {
    setState(prev => ({ ...prev, backgroundVideo: file }));
  }, []);

  const setVideoHandling = useCallback((handling: 'loop' | 'slow_down') => {
    setState(prev => ({ ...prev, videoHandling: handling }));
  }, []);

  const setFps = useCallback((fps: number) => {
    setState(prev => ({ ...prev, fps }));
  }, []);

  const setFastMode = useCallback((fastMode: boolean) => {
    setState(prev => ({ ...prev, fastMode }));
  }, []);

  const setMaxWorkers = useCallback((workers: number) => {
    setState(prev => ({ ...prev, maxWorkers: workers }));
  }, []);

  const setOutputFormat = useCallback((format: 'mp4' | 'webm' | 'mov') => {
    setState(prev => ({ ...prev, outputFormat: format }));
  }, []);

  // Actions
  const startProcessing = useCallback(async () => {
    if (!state.uploadedVideo) return;

    // Add debug message
    setState(prev => ({
      ...prev,
      debugLogs: [
        ...prev.debugLogs,
        {
          message: `📹 Starting video processing - File: ${state.uploadedVideo?.name || 'Unknown'}, Format: ${state.outputFormat}, Background: ${state.backgroundType}`,
          timestamp: Date.now() / 1000,
          sessionId: 'frontend'
        }
      ].slice(-500)
    }));

    try {
      // Validate inputs
      if (state.backgroundType === 'Image' && !state.backgroundImage) {
        throw new Error('Please select a background image');
      }
      if (state.backgroundType === 'Video' && !state.backgroundVideo) {
        throw new Error('Please select a background video');
      }

      // Prepare form data
      const formData = new FormData();
      formData.append('video', state.uploadedVideo);
      formData.append('bg_type', state.backgroundType);
      formData.append('color', state.backgroundColor);
      formData.append('fps', state.fps.toString());
      formData.append('video_handling', state.videoHandling);
      formData.append('fast_mode', state.fastMode.toString());
      formData.append('max_workers', state.maxWorkers.toString());
      formData.append('output_format', state.outputFormat);

      // Add background files if needed
      if (state.backgroundType === 'Image' && state.backgroundImage) {
        formData.append('background', state.backgroundImage);
      } else if (state.backgroundType === 'Video' && state.backgroundVideo) {
        formData.append('background', state.backgroundVideo);
      }

      // Update state
      setState(prev => ({
        ...prev,
        processingStatus: 'started',
        progress: 0,
        currentFrame: 0,
        totalFrames: 0,
        elapsedTime: 0,
        previewImage: null,
        outputFile: null,
        connectionError: null,
      }));

      // Send request
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      const response = await fetch(`${API_BASE}/api/process_video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const result = await response.json();
      console.log('🆔 NEW SESSION ID ASSIGNED:', result.session_id);
      setState(prev => ({ ...prev, sessionId: result.session_id }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        processingStatus: 'error',
        connectionError: error instanceof Error ? error.message : 'Failed to start processing',
      }));
    }
  }, [state]);

  const cancelProcessing = useCallback(async () => {
    if (!state.sessionId) return;

    try {
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      await fetch(`${API_BASE}/api/cancel_processing`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: state.sessionId }),
      });

      setState(prev => ({
        ...prev,
        processingStatus: 'idle',
        sessionId: null,
        progress: 0,
        currentFrame: 0,
        totalFrames: 0,
        elapsedTime: 0,
        previewImage: null,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: '🛑 Cancel request sent to backend',
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-100)
      }));
    } catch (error) {
      console.error('Error canceling:', error);
    }
  }, [state.sessionId]);

  const downloadVideo = useCallback(async () => {
    // Add debug message
    setState(prev => ({
      ...prev,
      debugLogs: [
        ...prev.debugLogs,
        {
          message: `🔽 [DOWNLOAD] Standard download button clicked`,
          timestamp: Date.now() / 1000,
          sessionId: 'frontend'
        }
      ].slice(-500)
    }));
    
    if (!state.outputFile) {
      console.error('DOWNLOAD DEBUG: No output file available for download');
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `❌ [DOWNLOAD] ERROR: No output file available`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      return;
    }
    
    try {
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      const fullUrl = `${API_BASE}${state.outputFile}`;
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `[DOWNLOAD] Output file: ${state.outputFile}`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          },
          {
            message: `[DOWNLOAD] Full URL: ${fullUrl}`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          },
          {
            message: `[DOWNLOAD] Fetching file from server...`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));

      const response = await fetch(fullUrl, { mode: 'cors' });
      
      if (!response.ok) {
        const errorText = await response.text();
        setState(prev => ({
          ...prev,
          debugLogs: [
            ...prev.debugLogs,
            {
              message: `❌ [DOWNLOAD] Fetch failed - Status: ${response.status}`,
              timestamp: Date.now() / 1000,
              sessionId: 'frontend'
            },
            {
              message: `❌ [DOWNLOAD] Error: ${errorText}`,
              timestamp: Date.now() / 1000,
              sessionId: 'frontend'
            }
          ].slice(-500)
        }));
        throw new Error(`Download failed: ${response.status}`);
      }
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `✅ [DOWNLOAD] Fetch successful, converting to blob...`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      
      const blob = await response.blob();
      const blobSizeMB = (blob.size / (1024 * 1024)).toFixed(2);
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `[DOWNLOAD] Blob created - Size: ${blob.size} bytes (${blobSizeMB} MB)`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      
      const blobUrl = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = blobUrl;
      
      // Derive extension from actual output file path to avoid mismatches if backend falls back
      const lastSlash = state.outputFile.lastIndexOf('/');
      const fileSegment = lastSlash >= 0 ? state.outputFile.slice(lastSlash + 1) : state.outputFile;
      const dotIndex = fileSegment.lastIndexOf('.');
      const ext = dotIndex >= 0 ? fileSegment.slice(dotIndex + 1) : (state.outputFormat || 'mp4');
      
      const filename = `processed_video_${Date.now()}.${ext}`;
      link.download = filename;
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `[DOWNLOAD] Triggering download - Filename: ${filename}`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      
      // Trigger download
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(blobUrl);
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `✅ [DOWNLOAD] Download triggered successfully!`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
    } catch (e: any) {
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `❌ [DOWNLOAD] Error: ${e?.message || 'Unknown error'}`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      alert(`Download failed: ${e?.message || 'Unknown error'}. Check debug console for details.`);
    }
  }, [state.outputFile, state.outputFormat, state.debugInfo]);

  const resetState = useCallback(() => {
    setState(prev => ({
      ...prev,
      uploadedVideo: null,
      processingStatus: 'idle',
      sessionId: null,
      progress: 0,
      currentFrame: 0,
      totalFrames: 0,
      elapsedTime: 0,
      previewImage: null,
      outputFile: null,
    }));
  }, []);

  // Direct download method that gets raw video data
  const downloadVideoDirectly = useCallback(async () => {
    setState(prev => ({
      ...prev,
      debugLogs: [
        ...prev.debugLogs,
        {
          message: `💾 [DIRECT DOWNLOAD] Direct download button clicked`,
          timestamp: Date.now() / 1000,
          sessionId: 'frontend'
        }
      ].slice(-500)
    }));
    
    if (!state.outputFile) {
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `❌ [DIRECT DOWNLOAD] ERROR: No output file available`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      return;
    }
    
    try {
      // Extract filename from outputFile path
      const lastSlash = state.outputFile.lastIndexOf('/');
      const filename = lastSlash >= 0 ? state.outputFile.slice(lastSlash + 1) : state.outputFile;
      
      setState(prev => ({
        ...prev,
        debugLogs: [
          ...prev.debugLogs,
          {
            message: `[DIRECT DOWNLOAD] Extracted filename: ${filename}`,
            timestamp: Date.now() / 1000,
            sessionId: 'frontend'
          }
        ].slice(-500)
      }));
      
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      const dataUrl = `${API_BASE}/api/get_video_data/${filename}`;
      console.log('DIRECT DOWNLOAD: Fetching from:', dataUrl);
      
      const response = await fetch(dataUrl);
      if (!response.ok) {
        console.error('DIRECT DOWNLOAD: API request failed:', response.status);
        const errorText = await response.text();
        console.error('DIRECT DOWNLOAD: Error details:', errorText);
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const result = await response.json();
      console.log('DIRECT DOWNLOAD: Received data:', { ...result, data: '[DATA URI TRUNCATED]' });
      
      if (!result.success || !result.data) {
        throw new Error('Invalid response from server');
      }
      
      // Create download from data URI
      const link = document.createElement('a');
      link.href = result.data;
      link.download = result.filename || `processed_video_${Date.now()}.${state.outputFormat}`;
      console.log('DIRECT DOWNLOAD: Triggering download with filename:', link.download);
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      console.log('DIRECT DOWNLOAD: Direct download process complete');
    } catch (e: any) {
      console.error('DIRECT DOWNLOAD: Error during direct download:', e);
      alert(`Direct download failed: ${e?.message || 'Unknown error'}. Check console for details.`);
    }
  }, [state.outputFile, state.outputFormat]);

  // Toggle debug panel
  const toggleDebugPanel = useCallback(() => {
    setState(prev => ({
      ...prev,
      showDebugPanel: !prev.showDebugPanel
    }));
  }, []);
  
  // Clear debug logs
  const clearDebugLogs = useCallback(() => {
    setState(prev => ({
      ...prev,
      debugLogs: []
    }));
  }, []);

  // Create context value
  const value: VideoProcessorContextType = {
    ...state,
    setUploadedVideo,
    setBackgroundType,
    setBackgroundColor,
    setBackgroundImage,
    setBackgroundVideo,
    setVideoHandling,
    setFps,
    setFastMode,
    setMaxWorkers,
    setOutputFormat,
    startProcessing,
    cancelProcessing,
    downloadVideo,

    resetState,
    toggleDebugPanel,
    clearDebugLogs,
  };

  return (
    <VideoProcessorContext.Provider value={value}>
      {children}
    </VideoProcessorContext.Provider>
  );
}

// Hook
export function useVideoProcessor() {
  const context = useContext(VideoProcessorContext);
  if (context === undefined) {
    throw new Error('useVideoProcessor must be used within a VideoProcessorProvider');
  }
  return context;
}