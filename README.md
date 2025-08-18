# Video Background Remover

A powerful AI-powered desktop application for removing and replacing video backgrounds using BiRefNet models. Built with Electron frontend and Python backend architecture for optimal performance.

## 🏗️ Architecture

- **Frontend**: React + TypeScript + Material-UI running in Electron
- **Backend**: Python FastAPI server with BiRefNet AI models
- **Processing**: Cached model loading with parallel video processing
- **Communication**: WebSocket real-time updates between frontend and backend

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (with pip)
- **Node.js 16+** (with npm)
- **8GB+ RAM** recommended for AI model processing

### One-Click Setup (Windows)

```batch
start.bat
```

This automated script will:
1. ✅ Verify Python and Node.js installation
2. 📦 Install all Node.js and Python dependencies
3. ⬇️ Download FFmpeg binaries automatically
4. 🏗️ Build the React frontend
5. 🚀 Launch the complete application

### Manual Setup

```bash
# 1. Install root dependencies (Electron, build tools)
npm install

# 2. Install frontend dependencies
cd frontend && npm install && cd ..

# 3. Install Python API dependencies
python -m pip install -r api_requirements.txt

# 4. Download FFmpeg (if needed)
npm run download-ffmpeg

# 5. Build and start the application
npm start
```

## ✨ Features

- **Modern Desktop UI**: Beautiful, responsive interface with smooth animations
- **Multiple Background Options**:
  - Solid colors with color picker
  - Custom background images
  - Background videos with smart handling (loop or slow down)
- **Real-time Processing**: Live preview updates during video processing
- **Advanced Settings**:
  - Configurable FPS output
  - Fast mode toggle (BiRefNet vs BiRefNet_lite)
  - Parallel processing with adjustable worker count

## 📁 Project Structure

```
video-background-removal/
├── 📂 electron/           # Electron main process files
├── 📂 frontend/           # React TypeScript frontend
│   ├── 📂 src/components/ # UI components
│   ├── 📂 src/context/    # React context providers
│   └── 📂 public/         # Static assets
├── 📄 api_server.py       # Python FastAPI backend
├── 📄 api_requirements.txt # Python dependencies
├── 📄 download_ffmpeg.js  # FFmpeg auto-download script
├── 📄 start.bat          # One-click Windows launcher
└── 📄 package.json       # Node.js dependencies & scripts
```

## 🔧 Troubleshooting

### First Steps
- **Check the console**: Electron app shows detailed error messages
- **Verify prerequisites**: Ensure Python 3.8+ and Node.js 16+ are installed
- **Run start.bat**: The automated script handles most setup issues

### Common Issues

**🚫 FFMPEG Missing**
```bash
npm run download-ffmpeg
```
Or manually download from [FFmpeg builds](https://www.gyan.dev/ffmpeg/builds/) and place in `bin/` directory.

**🐍 Python Issues**
- Install Python 3.8+ and ensure it's in your PATH
- Try: `python --version` and `pip --version`
- Reinstall dependencies: `pip install -r api_requirements.txt`

**🧠 Memory/AI Model Issues**
- **Minimum 8GB RAM** required for AI processing
- Close other memory-intensive applications
- Use "Fast Mode" in settings for lighter processing
- Process shorter videos or reduce resolution

**🔌 Connection Issues**
- Backend runs on `http://localhost:5000`
- Frontend connects automatically via WebSocket
- Check Windows Firewall/antivirus blocking ports

**📦 Build/Dependency Issues**
```bash
# Clean reinstall
rm -rf node_modules frontend/node_modules
npm install
```

## 🧠 How It Works

The application uses advanced AI and modern web technologies to deliver professional video background removal:

### 🔄 Processing Pipeline
1. **📹 Video Upload**: Drag & drop or select video files
2. **🎯 Frame Extraction**: FFmpeg splits video into individual frames
3. **🤖 AI Segmentation**: BiRefNet models create precise masks for each frame
4. **🎨 Background Replacement**: Replace with colors, images, or videos
5. **⚡ Parallel Processing**: Multiple worker threads for faster rendering
6. **📦 Video Reconstruction**: FFmpeg combines processed frames back to video

### 🧩 Technology Stack
- **Frontend**: React 18 + TypeScript + Material-UI + Framer Motion
- **Backend**: Python FastAPI + WebSocket + BiRefNet AI models
- **Desktop**: Electron for cross-platform native experience
- **Processing**: FFmpeg for video manipulation + PyTorch for AI inference
- **State Management**: React Context + Real-time WebSocket updates

### 🎯 AI Models Used
- **BiRefNet**: High-quality background segmentation
- **BiRefNet_lite**: Faster processing with good quality (Fast Mode)
- **Model Caching**: Models downloaded once and cached locally

## 🚀 Building & Distribution

```bash
# Build for production
npm run build

# Package for current platform
npm run dist

# Development build
npm run pack
```

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

### 🧠 AI Model Credits
This application is built upon the excellent work of the BiRefNet research team:

- **BiRefNet Model**: [Bilateral Reference for High-Resolution Dichotomous Image Segmentation](https://github.com/ZhengPeng7/BiRefNet)
- **Original Authors**: Zheng Peng, Jianbo Jiao, and colleagues
- **Research Institution**: University of Birmingham & other collaborating institutions
- **Paper**: "Bilateral Reference for High-Resolution Dichotomous Image Segmentation"
- **License**: MIT License (original repository)

We extend our deepest gratitude to the BiRefNet research team for making their groundbreaking work open-source and accessible to the developer community.

### 🛠️ Technology Stack Credits
- **[Electron](https://electronjs.org/)**: Cross-platform desktop app framework  
- **[React](https://reactjs.org/)**: Modern UI library with TypeScript
- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance Python web framework
- **[Material-UI](https://mui.com/)**: Beautiful React component library
- **[FFmpeg](https://ffmpeg.org/)**: Video processing and encoding