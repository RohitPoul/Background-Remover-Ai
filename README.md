# 🎬 Video Background Remover AI

An AI-powered desktop application for removing and replacing video backgrounds using state-of-the-art BiRefNet models. Built with Electron, React, and Python for cross-platform compatibility.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node](https://img.shields.io/badge/node-16+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## ✨ Features

- **🤖 AI-Powered Background Removal** - Uses BiRefNet/BiRefNet-Lite models for precise segmentation
- **🚀 GPU Acceleration** - Automatic CUDA detection for NVIDIA GPUs (16x faster than CPU)
- **🎨 Multiple Background Options**:
  - Solid colors
  - Custom images
  - Video backgrounds
  - Transparent (alpha channel preserved)
- **📊 Real-time Processing** - Live preview with frame-by-frame progress
- **🎯 Smart Hardware Detection** - Automatically optimizes settings based on your system
- **💾 Multiple Output Formats** - MP4, WebM, MOV (with transparency support)
- **🖥️ Native Desktop App** - Electron-based for Windows, macOS, and Linux

## 🖼️ Screenshots

<details>
<summary>Click to view screenshots</summary>

### Main Interface
The application features a modern, intuitive interface with real-time preview capabilities.

### Hardware Detection
Automatic GPU detection and optimization for maximum performance.

### Processing Options
Multiple background types and customization options.

</details>

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
2. **Node.js 16+** - [Download Node.js](https://nodejs.org/)
3. **FFmpeg** - [Download FFmpeg](https://www.gyan.dev/ffmpeg/builds/)
   - Extract `ffmpeg.exe` to the `bin/` folder

### Installation

```bash
# Clone the repository
git clone https://github.com/RohitPoul/Background-Remover-Ai.git
cd Background-Remover-Ai

# Install dependencies
npm install
pip install -r api_requirements.txt
```

### Running the Application

```bash
# Start the application (builds and launches Electron app)
npm start
```

That's it! The app will:
1. Detect your hardware (GPU/CPU)
2. Start the Python backend
3. Launch the Electron interface

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: Dual-core processor
- **RAM**: 4GB
- **GPU**: Not required (CPU fallback available)
- **Storage**: 2GB free space

### Recommended Requirements
- **CPU**: Quad-core or better
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Storage**: 5GB free space

### Supported GPUs
- ✅ NVIDIA GPUs (CUDA 11.8+)
- ✅ AMD GPUs (ROCm - limited support)
- ✅ Apple Silicon (M1/M2 via MPS)
- ✅ Intel GPUs (limited support)

## 📊 Performance

| Hardware | Processing Speed | Configuration |
|----------|-----------------|---------------|
| RTX 4090 | ~60 FPS | Ultra profile, batch size 16 |
| RTX 3080 | ~30 FPS | High profile, batch size 8 |
| GTX 1650 | ~9 FPS | High profile, batch size 4 |
| CPU Only | ~0.6 FPS | Low profile, single frame |

## 🎯 Features in Detail

### AI Models
- **BiRefNet** - High quality model for best results
- **BiRefNet-Lite** - Optimized model for faster processing
- Automatic model selection based on hardware

### Background Types
1. **Transparent** - Preserves alpha channel (MOV/WebM)
2. **Solid Color** - Any color picker selection
3. **Image** - JPG, PNG, or other image formats
4. **Video** - MP4, WebM, or MOV backgrounds

### Hardware Optimization
- Automatic GPU detection and configuration
- Dynamic batch size adjustment
- Memory-aware processing
- Mixed precision support for compatible GPUs

### Output Formats
- **MP4** - Standard video format
- **WebM** - Web-optimized with transparency
- **MOV** - Professional format with alpha channel

## 🔧 Development

### Project Structure
```
video-background-removal/
├── electron/           # Electron main process
├── frontend/          # React UI application
│   ├── src/
│   │   ├── components/  # UI components
│   │   ├── context/     # React context
│   │   └── styles/      # CSS styles
├── api_server.py      # Python backend server
├── hardware_optimizer.py # Hardware detection
├── bin/              # FFmpeg binaries
└── model_cache/      # AI model storage
```

### Key Technologies
- **Frontend**: React, TypeScript, Material-UI
- **Backend**: Python, FastAPI, PyTorch
- **Desktop**: Electron
- **AI Models**: Transformers, BiRefNet
- **Video Processing**: FFmpeg, MoviePy

### Development Mode

```bash
# Run in development mode with hot reload
npm run dev
```

## 🐛 Troubleshooting

### Common Issues

1. **"Backend not connected" error**
   - Ensure Python is installed and in PATH
   - Check if port 5000 is available
   - Restart the application

2. **GPU not detected**
   - Install CUDA toolkit for NVIDIA GPUs
   - Update GPU drivers
   - Check PyTorch CUDA installation

3. **Video processing fails**
   - Ensure FFmpeg is in the `bin/` folder
   - Check available disk space
   - Try reducing video resolution

4. **Out of memory errors**
   - Use Fast Mode (BiRefNet-Lite)
   - Reduce video resolution
   - Close other applications

## 📈 Roadmap

- [ ] Cloud processing support
- [ ] Batch video processing
- [ ] Real-time webcam support
- [ ] Mobile app version
- [ ] More AI model options
- [ ] Custom model training

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - The AI model powering background removal
- [Electron](https://www.electronjs.org/) - Desktop application framework
- [React](https://reactjs.org/) - UI framework
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FFmpeg](https://ffmpeg.org/) - Video processing

## 👨‍💻 Author

**Rohit Poul**
- GitHub: [@RohitPoul](https://github.com/RohitPoul)
- Project: [Background-Remover-Ai](https://github.com/RohitPoul/Background-Remover-Ai)

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub!

---

<p align="center">Made with ❤️ using AI and Open Source technologies</p>