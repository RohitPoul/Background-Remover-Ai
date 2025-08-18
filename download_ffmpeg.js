/**
 * FFMPEG Downloader for Video Background Remover
 * 
 * This script downloads and extracts the FFMPEG executable needed for video processing.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { spawn } = require('child_process');
const extract = require('extract-zip');

// Configuration
const FFMPEG_URL = 'https://github.com/GyanD/codexffmpeg/releases/download/5.1.2/ffmpeg-5.1.2-essentials_build.zip';
const DOWNLOAD_PATH = path.join(__dirname, 'ffmpeg.zip');
const EXTRACT_PATH = path.join(__dirname, 'ffmpeg-temp');
const BIN_DIR = path.join(__dirname, 'bin');

// Create bin directory if it doesn't exist
if (!fs.existsSync(BIN_DIR)) {
  fs.mkdirSync(BIN_DIR);
  console.log(`Created bin directory at ${BIN_DIR}`);
}

// Check if FFMPEG already exists
const ffmpegPath = path.join(BIN_DIR, 'ffmpeg.exe');
if (fs.existsSync(ffmpegPath)) {
  console.log(`FFMPEG already exists at ${ffmpegPath}`);
  process.exit(0);
}

console.log('Downloading FFMPEG...');
console.log(`URL: ${FFMPEG_URL}`);
console.log(`Download path: ${DOWNLOAD_PATH}`);

// Download FFMPEG
const file = fs.createWriteStream(DOWNLOAD_PATH);
function fetchWithRedirect(url, redirectsLeft = 5, onResponse) {
  https.get(url, (response) => {
    if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location && redirectsLeft > 0) {
      const nextUrl = response.headers.location.startsWith('http')
        ? response.headers.location
        : new URL(response.headers.location, url).toString();
      response.resume();
      return fetchWithRedirect(nextUrl, redirectsLeft - 1, onResponse);
    }
    onResponse(response);
  }).on('error', (err) => {
    console.error(`Error downloading FFMPEG: ${err.message}`);
    if (fs.existsSync(DOWNLOAD_PATH)) {
      try { fs.unlinkSync(DOWNLOAD_PATH); } catch {}
    }
    process.exit(1);
  });
}

fetchWithRedirect(FFMPEG_URL, 5, (response) => {
  if (response.statusCode !== 200) {
    console.error(`Failed to download FFMPEG: ${response.statusCode} ${response.statusMessage}`);
    if (fs.existsSync(DOWNLOAD_PATH)) {
      try { fs.unlinkSync(DOWNLOAD_PATH); } catch {}
    }
    process.exit(1);
  }

  const totalSize = parseInt(response.headers['content-length'], 10);
  let downloadedSize = 0;
  
  response.on('data', (chunk) => {
    downloadedSize += chunk.length;
    const progress = (downloadedSize / totalSize * 100).toFixed(2);
    process.stdout.write(`Downloading: ${progress}% (${downloadedSize}/${totalSize} bytes)\r`);
  });
  
  response.pipe(file);
  
  file.on('finish', () => {
    file.close();
    console.log('\nDownload complete!');
    extractFFMPEG();
  });
});

// Extract FFMPEG
async function extractFFMPEG() {
  console.log(`Extracting FFMPEG to ${EXTRACT_PATH}...`);
  
  try {
    // Create extract directory if it doesn't exist
    if (!fs.existsSync(EXTRACT_PATH)) {
      fs.mkdirSync(EXTRACT_PATH);
    }
    
    // Extract zip file
    await extract(DOWNLOAD_PATH, { dir: EXTRACT_PATH });
    console.log('Extraction complete!');
    
    // Find ffmpeg.exe in the extracted files
    const ffmpegExe = findFFMPEGExe(EXTRACT_PATH);
    if (!ffmpegExe) {
      console.error('Could not find ffmpeg.exe in the extracted files');
      cleanup();
      process.exit(1);
    }
    
    // Copy ffmpeg.exe to bin directory
    fs.copyFileSync(ffmpegExe, ffmpegPath);
    console.log(`FFMPEG copied to ${ffmpegPath}`);
    
    // Cleanup
    cleanup();
    console.log('FFMPEG installation complete!');
  } catch (err) {
    console.error(`Error extracting FFMPEG: ${err.message}`);
    cleanup();
    process.exit(1);
  }
}

// Find ffmpeg.exe in the extracted files
function findFFMPEGExe(dir) {
  const files = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const file of files) {
    const filePath = path.join(dir, file.name);
    
    if (file.isDirectory()) {
      const result = findFFMPEGExe(filePath);
      if (result) {
        return result;
      }
    } else if (file.name === 'ffmpeg.exe') {
      return filePath;
    }
  }
  
  return null;
}

// Cleanup temporary files
function cleanup() {
  console.log('Cleaning up temporary files...');
  
  try {
    if (fs.existsSync(DOWNLOAD_PATH)) {
      fs.unlinkSync(DOWNLOAD_PATH);
    }
    
    if (fs.existsSync(EXTRACT_PATH)) {
      fs.rmdirSync(EXTRACT_PATH, { recursive: true });
    }
  } catch (err) {
    console.error(`Error cleaning up: ${err.message}`);
  }
}
