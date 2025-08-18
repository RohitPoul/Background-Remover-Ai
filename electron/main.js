let electronModule = require('electron');
const { spawn: spawnProcess } = require('child_process');

// If not running under Electron runtime (e.g., ELECTRON_RUN_AS_NODE set), relaunch properly
if (!electronModule || !electronModule.app || typeof electronModule.app.whenReady !== 'function') {
  try {
    const electronPath = require('electron');
    const env = { ...process.env };
    delete env.ELECTRON_RUN_AS_NODE; // ensure proper electron mode
    spawnProcess(electronPath, ['.'], { stdio: 'inherit', env });
    process.exit(0);
  } catch (e) {
    console.error('Failed to launch Electron runtime:', e);
    process.exit(1);
  }
}

const { app, BrowserWindow, ipcMain, dialog, shell } = electronModule;
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const Store = require('electron-store');

const store = new Store();
let mainWindow;
let pythonProcess;

// Force production UI (always load built React app inside Electron)
const isDev = false;

function createWindow() {
  // Create the browser window
  const isMac = process.platform === 'darwin';
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: isMac ? 'hiddenInset' : 'default',
    show: false,
    backgroundColor: '#1a1a1a',
    icon: path.join(__dirname, '../frontend/public/logo512.png')
  });

  // Windows: set AppUserModelID to avoid taskbar grouping/registration error
  try {
    if (process.platform === 'win32') {
      app.setAppUserModelId('com.videoprocessor.app');
    }
  } catch {}

  // Load the app with proper base URL for resources
  mainWindow.loadFile(path.join(__dirname, '../frontend/build/index.html'));
  
  // No base-tag injection needed; paths are relative via homepage ./ in frontend/package.json

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

function startPythonServer() {
  console.log('Starting Python server...');
  
  // Check if FFMPEG exists
  const ffmpegPath = path.join(process.cwd(), 'bin', 'ffmpeg.exe');
  if (!fs.existsSync(ffmpegPath)) {
    console.warn(`FFMPEG not found at ${ffmpegPath}`);
    dialog.showMessageBox(mainWindow, {
      type: 'warning',
      title: 'FFMPEG Not Found',
      message: 'FFMPEG is required for video processing',
      detail: `FFMPEG was not found at ${ffmpegPath}.\n\nPlease download FFMPEG from https://www.gyan.dev/ffmpeg/builds/ and place ffmpeg.exe in the bin directory.\n\nThe application will continue but video processing won't work without FFMPEG.`,
      buttons: ['Continue', 'Exit']
    }).then((result) => {
      if (result.response === 1) {
        app.quit();
      }
    });
    return;
  }
  
  // Start Python server
  pythonProcess = spawn('python', ['api_server.py'], {
    cwd: process.cwd(),
    stdio: 'pipe',
    env: {
      ...process.env,
      IMAGEIO_FFMPEG_EXE: ffmpegPath
    }
  });

  // Create log directory if it doesn't exist
  const logsDir = path.join(process.cwd(), 'logs');
  if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir);
  }
  
  // Create log file streams
  const stdoutLog = fs.createWriteStream(path.join(logsDir, 'python_stdout.log'), { flags: 'a' });
  const stderrLog = fs.createWriteStream(path.join(logsDir, 'python_stderr.log'), { flags: 'a' });

  pythonProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`Python: ${output}`);
    stdoutLog.write(`[${new Date().toISOString()}] ${output}`);
    
    // Check for specific messages
    if (output.includes('Running on')) {
      console.log('Python server is ready');
    }
  });

  pythonProcess.stderr.on('data', (data) => {
    const error = data.toString();
    console.error(`Python Error: ${error}`);
    stderrLog.write(`[${new Date().toISOString()}] ${error}`);
    
    // Show error dialog for critical errors
    if (error.includes('RuntimeError') || error.includes('ImportError')) {
      dialog.showErrorBox(
        'Python Error',
        `An error occurred in the Python backend:\n\n${error}\n\nCheck logs/python_stderr.log for details.`
      );
    }
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    if (code !== 0 && code !== null) {
      dialog.showErrorBox(
        'Python Server Error',
        `The Python server exited unexpectedly with code ${code}.\n\nCheck logs/python_stderr.log for details.`
      );
      app.quit();
    }
  });
}

// Wait for app to be ready
app.whenReady().then(() => {
  // Create window and start Python server
  createWindow();
  startPythonServer();
  console.log('Electron window created, loading production build');

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
});

// IPC handlers
ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('get-store-value', (event, key) => {
  return store.get(key);
});

ipcMain.handle('set-store-value', (event, key, value) => {
  store.set(key, value);
});

ipcMain.handle('open-external', (event, url) => {
  shell.openExternal(url);
});

// Window controls
ipcMain.handle('minimize-window', () => {
  if (mainWindow) {
    mainWindow.minimize();
  }
});

ipcMain.handle('maximize-window', () => {
  if (mainWindow) {
    if (mainWindow.isMaximized()) {
      mainWindow.unmaximize();
    } else {
      mainWindow.maximize();
    }
  }
});

ipcMain.handle('close-window', () => {
  if (mainWindow) {
    mainWindow.close();
  }
});