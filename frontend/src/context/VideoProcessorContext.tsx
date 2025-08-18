import React, { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';

// Types
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
}

// Context
const VideoProcessorContext = createContext<VideoProcessorContextType | undefined>(undefined);

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
  });

  // Socket connection
  const [socket, setSocket] = useState<Socket | null>(null);

  // Initialize socket connection with retry logic
  useEffect(() => {
    const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
    let currentSocket: Socket | null = null;
    let timeoutId: NodeJS.Timeout | null = null;
    
    const initSocket = () => {
      const newSocket = io(API_BASE, {
        reconnectionAttempts: 10,
        reconnectionDelay: 2000,
        timeout: 15000,
        autoConnect: false, // Don't connect immediately
      });

      newSocket.on('connect', () => {
        console.log('Connected to Python server');
        setState(prev => ({ ...prev, connectionError: null }));
      });

      newSocket.on('connect_error', (err) => {
        console.log('Connection error:', err);
        setState(prev => ({ 
          ...prev, 
          connectionError: 'Waiting for Python server to start...' 
        }));
      });

      newSocket.on('disconnect', () => {
        console.log('Disconnected from server');
        setState(prev => ({ 
          ...prev, 
          connectionError: 'Connection lost. Attempting to reconnect...' 
        }));
      });

      newSocket.on('processing_update', (data) => {
        if (data.session_id === state.sessionId) {
          setState(prev => ({
            ...prev,
            processingStatus: data.status === 'started' ? 'started' : 'processing',
            progress: data.progress || prev.progress,
            elapsedTime: data.elapsed_time || prev.elapsedTime,
            currentFrame: data.currentFrame || prev.currentFrame,
            totalFrames: data.totalFrames || prev.totalFrames,
            previewImage: data.preview_image || prev.previewImage,
          }));
        }
      });

      newSocket.on('processing_complete', (data) => {
        if (data.session_id === state.sessionId) {
          setState(prev => ({
            ...prev,
            processingStatus: 'completed',
            outputFile: `/api/download/${data.output_file}`,
            elapsedTime: data.elapsed_time || prev.elapsedTime,
          }));
        }
      });

      newSocket.on('processing_error', (data) => {
        if (data.session_id === state.sessionId) {
          setState(prev => ({
            ...prev,
            processingStatus: 'error',
            connectionError: data.message || 'Processing error occurred',
          }));
        }
      });

      setSocket(newSocket);
      currentSocket = newSocket;
      
      // Try to connect after a delay to allow Python server to start
      timeoutId = setTimeout(() => {
        newSocket.connect();
      }, 3000);

      return newSocket;
    };

    // Initialize socket with delay
    const socket = initSocket();

    // Cleanup function
    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      if (currentSocket) {
        currentSocket.disconnect();
      }
    };
  }, [state.sessionId]);

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
      }));
    } catch (error) {
      console.error('Error canceling:', error);
    }
  }, [state.sessionId]);

  const downloadVideo = useCallback(() => {
    if (state.outputFile) {
      const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000').replace(/\/$/, '');
      const fullUrl = `${API_BASE}${state.outputFile}`;
      window.open(fullUrl, '_blank');
    }
  }, [state.outputFile]);

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