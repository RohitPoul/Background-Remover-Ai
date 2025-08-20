# 🎬 AI Video Background Remover

A professional-grade AI-powered desktop application for removing and replacing video backgrounds using state-of-the-art BiRefNet models. Features real-time processing, transparent video support, and advanced debugging tools.

## ✨ Key Features

### 🎯 **Professional Background Removal**
- **AI-Powered**: Uses BiRefNet and BiRefNet Lite models for precise background segmentation
- **High Quality**: Supports up to 4K video processing with quality/speed modes
- **Transparent Output**: Create transparent videos (WebM/MOV) perfect for stickers and overlays
- **Multiple Formats**: Export as MP4, WebM (with alpha), or MOV (ProRes 4444)

### 🚀 **Advanced Processing Options**
- **Dual Model System**: 
  - **Quality Mode**: BiRefNet full model for best results
  - **Fast Mode**: BiRefNet Lite for 4x faster processing
- **Background Types**:
  - 🎨 Solid colors with color picker
  - 🖼️ Custom background images  
  - 🎥 Background videos with smart handling (loop/slow down)
  - ✨ **Transparent backgrounds** for sticker creation
- **Parallel Processing**: Multi-threaded frame processing (1-4 workers)
- **Custom FPS**: Control output frame rate (auto or 15-60 fps)

### 🔧 **Developer-Friendly Tools**
- **Real-Time Debug Console**: Complete visibility into processing pipeline
- **Live Preview**: See results as frames are processed
- **Progress Tracking**: Detailed progress with frame counters and timing
- **Copy/Export Logs**: Share debug information easily
- **Session Management**: Robust error handling and cancellation support

### 💻 **Modern Desktop Experience**
- **Beautiful UI**: Material Design with dark theme and smooth animations
- **Drag & Drop**: Intuitive file upload interface
- **Dual Preview**: Side-by-side original/processed video comparison
- **Download Options**: Standard and direct download methods
- **Cross-Platform**: Electron-based desktop application

## 🏗️ Architecture

```
┌─────────────────┐    WebSocket    ┌──────────────────┐
│   Electron UI   │ ←──────────────→ │  Python Backend  │
│                 │                  │                  │
│ • React + TS    │                  │ • FastAPI        │
│ • Material-UI   │                  │ • BiRefNet AI    │
│ • Debug Console │                  │ • FFmpeg         │
│ • Live Preview  │                  │ • Parallel Proc. │
└─────────────────┘                  └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (with pip)
- **Node.js 16+** (with npm)  
- **8GB+ RAM** recommended for AI processing
- **Windows 10+** (primary platform)

### One-Click Setup (Windows)
```batch
restart_backend.bat
```

This automated script will:
1. ✅ Install all Python dependencies
2. 🧠 Load AI models (BiRefNet + BiRefNet Lite)
3. 🚀 Start the backend server with debug logging
4. 📡 Enable WebSocket communication

### Manual Setup

```bash
# 1. Install root dependencies
npm install

# 2. Install frontend dependencies  
cd frontend && npm install && cd ..

# 3. Install Python dependencies
python -m pip install -r api_requirements.txt

# 4. Start backend server
python api_server.py

# 5. Start frontend (in new terminal)
cd frontend && npm start
```

## 📱 Usage

### Basic Workflow
1. **📂 Upload Video**: Drag & drop or select your video file
2. **🎨 Choose Background**: Select transparent, color, image, or video background
3. **⚙️ Configure Settings**: Adjust quality mode, FPS, and processing options
4. **▶️ Start Processing**: Watch real-time progress and live preview
5. **📥 Download Result**: Choose standard or direct download method

### Pro Tips
- **🔥 Fast Mode**: Enable for 4x faster processing of large videos
- **🎯 Quality Mode**: Disable fast mode for best results on smaller videos  
- **📊 Debug Console**: Monitor processing with detailed logs and copy functionality
- **🎬 Transparent Videos**: Use WebM or MOV format for sticker creation
- **⚡ Parallel Processing**: Increase workers (2-4) for faster processing

## 🎯 Output Formats

| Format | Transparency | Quality | Use Case |
|--------|-------------|---------|----------|
| **MP4** | ❌ No | High | Standard videos, social media |
| **WebM** | ✅ Yes | High | Web stickers, transparent overlays |
| **MOV** | ✅ Yes | Highest | Professional editing, ProRes 4444 |

## 🔧 Advanced Features

### Debug Console
- **Real-Time Logging**: See every processing step
- **Copy Functionality**: Share debug info easily
- **Export Logs**: Download complete processing logs
- **Session Tracking**: Monitor multiple processing sessions
- **Performance Metrics**: Frame timing and success rates

### Processing Pipeline Visibility
- **Model Loading**: Track AI model initialization
- **Frame Extraction**: Monitor video frame extraction
- **AI Inference**: See per-frame processing time and results
- **Video Compilation**: Track final video assembly and encoding
- **File Output**: Verify file creation and download URLs

## 📁 Project Structure

```
video-background-removal/
├── 📂 electron/                 # Electron main process
├── 📂 frontend/                 # React TypeScript UI
│   ├── 📂 src/components/       # UI components
│   │   ├── VideoProcessor.tsx   # Main processing interface
│   │   ├── DualVideoPreview.tsx # Side-by-side video preview
│   │   ├── ProcessingControls.tsx # Start/cancel/download controls
│   │   ├── DebugPanel.tsx       # Advanced debugging console
│   │   ├── BackgroundSettings.tsx # Background configuration
│   │   └── ProcessingProgress.tsx # Progress tracking
│   ├── 📂 src/context/          # React context providers
│   │   └── VideoProcessorContext.tsx # Main state management
│   └── 📂 public/               # Static assets
├── 📄 api_server.py             # Python backend with AI processing
├── 📄 api_requirements.txt      # Python dependencies
├── 📂 bin/                      # FFmpeg binaries
├── 📂 model_cache/              # Cached AI models
├── 📂 logs/                     # Application logs
├── 📄 restart_backend.bat       # Backend restart script
└── 📄 package.json              # Node.js configuration
```

## 🧠 AI Models & Performance

### BiRefNet Models
- **BiRefNet Full**: Best quality, ~60s per 4K frame
- **BiRefNet Lite**: 4x faster, ~15s per 4K frame, good quality
- **Auto-Loading**: Models download and cache automatically
- **CPU Processing**: Optimized for CPU-only environments

### Performance Guidelines

| Video Resolution | Fast Mode | Quality Mode | Recommended |
|-----------------|-----------|--------------|-------------|
| **1080p** | ~5s/frame | ~15s/frame | Quality Mode |
| **2K** | ~10s/frame | ~30s/frame | Fast Mode |
| **4K** | ~15s/frame | ~60s/frame | Fast Mode |

## 🔧 Troubleshooting

### Debug Console
The built-in Debug Console provides complete visibility:
- **Connection Status**: Backend connectivity
- **Model Loading**: AI model initialization progress  
- **Frame Processing**: Per-frame AI inference results
- **Error Details**: Complete error traces and solutions
- **Performance Metrics**: Processing speed and memory usage

### Common Solutions

**🚫 Processing Stuck/Slow**
- Enable **Fast Mode** for large videos
- Reduce video resolution before processing
- Close memory-intensive applications
- Check Debug Console for specific bottlenecks

**🔌 Connection Issues**  
- Restart backend: `restart_backend.bat`
- Check port 5000 availability
- Verify Python dependencies installed

**🧠 Model Loading Failures**
- Ensure 8GB+ RAM available
- Check internet connection for model download
- Clear model cache and restart

**📱 UI/Frontend Issues**
- Refresh page (F5) to reconnect
- Check browser console for errors
- Verify Node.js dependencies installed

## 🛠️ Development

### Building from Source
```bash
# Development mode
npm run dev

# Production build
npm run build

# Package for distribution
npm run dist
```

### API Endpoints
- `POST /api/process_video` - Start video processing
- `POST /api/cancel_processing` - Cancel active processing  
- `GET /api/download/<filename>` - Download processed video
- `GET /api/get_video_data/<filename>` - Get video as data URI
- `GET /api/health` - System health check

### WebSocket Events
- `processing_update` - Real-time progress updates
- `processing_complete` - Processing finished notification
- `processing_error` - Error notifications
- `debug_log` - Backend debug messages

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8GB (16GB recommended for 4K)
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: 2GB free space (for models and temp files)
- **Network**: Internet connection for initial model download

### Recommended Specs
- **RAM**: 16GB+ for smooth 4K processing
- **CPU**: 8+ cores for optimal parallel processing
- **SSD**: Fast storage for improved frame I/O
- **GPU**: Not required (CPU-optimized processing)

## 🎬 Use Cases

- **🎮 Content Creation**: Remove backgrounds for gaming overlays
- **📱 Social Media**: Create transparent stickers and animations  
- **🎥 Video Editing**: Pre-process footage for professional editing
- **📺 Streaming**: Create overlay content for live streams
- **🎪 Marketing**: Generate transparent product videos
- **🎨 Creative Projects**: Artistic video compositions

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

### 🧠 AI Model Credits
This application is powered by the groundbreaking BiRefNet research:

- **BiRefNet Model**: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://github.com/ZhengPeng7/BiRefNet)
- **Authors**: Zheng Peng, Jianbo Jiao, and research team
- **Institution**: University of Birmingham & collaborating institutions
- **Paper**: "Bilateral Reference for High-Resolution Dichotomous Image Segmentation"

### 🛠️ Technology Stack
- **[Electron](https://electronjs.org/)**: Cross-platform desktop framework
- **[React](https://reactjs.org/)**: Modern UI library with TypeScript  
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance Python web framework
- **[Material-UI](https://mui.com/)**: Beautiful React component library
- **[FFmpeg](https://ffmpeg.org/)**: Professional video processing
- **[PyTorch](https://pytorch.org/)**: Machine learning framework
- **[Socket.IO](https://socket.io/)**: Real-time communication

---

## 🌟 Recent Updates

### Version 2.0 - Major Improvements
- ✅ **Fixed Critical Bugs**: SocketIO communication, session management
- ✅ **Enhanced Debug Tools**: Copy logs, export functionality, real-time visibility
- ✅ **Improved AI Models**: Proper quality/fast mode selection  
- ✅ **Better Performance**: Optimized 4K processing, parallel frame handling
- ✅ **Code Cleanup**: Removed 400+ lines of duplicate code
- ✅ **UI Enhancements**: Stable debug labels, better error handling
- ✅ **Transparent Video Support**: Perfect for sticker creation

**Ready for professional video background removal workflows!** 🚀