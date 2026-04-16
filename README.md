# Gemma 4 Audio Transcriber

Local speech-to-text using Google's Gemma 4 E4B model via [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Runs entirely on your machine -- no cloud, no API keys needed (just a HuggingFace token for the initial model download).

## Features

- **Native GUI** (egui) with status indicator, mic level meter, and transcription history
- **Global hotkey** (default: F9) -- works in any app, configurable to F5-F12, Scroll Lock, or Pause
- **Push-to-Talk** -- hold the hotkey to record, release to transcribe
- **VAD mode** -- automatic voice activity detection (energy-based)
- **Auto-paste** -- transcription is pasted directly into the focused app via clipboard
- **GPU accelerated** -- CUDA with Q4K quantization, fits in 12GB VRAM
- **Local model cache** -- model files stored in `./models/`, downloaded once
- **Garbage filtering** -- HTML/malformed responses are discarded

## Requirements

- Windows 10/11
- NVIDIA GPU with 12GB+ VRAM (tested on RTX 5070)
- CUDA Toolkit installed
- [HuggingFace token](https://huggingface.co/settings/tokens) with access to [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- Visual Studio Build Tools (for CUDA kernel compilation)

## Quick Start (pre-built binary)

1. Download `gemma4-audio-transcriber.exe` from [Releases](https://github.com/dx111ge/gemma4-audio-transcriber/releases)
2. Set your HuggingFace token:
   ```powershell
   $env:HF_TOKEN = "hf_your_token_here"
   ```
3. Run the exe -- first launch downloads the model (~15GB) and quantizes it (~2 min)
4. Once status shows **READY**, hold **F9** in any app and speak
5. Enable **Auto-paste** to have transcriptions typed into the focused app

## Building from Source

```powershell
# Set MSVC compiler path for CUDA kernel compilation
$env:NVCC_CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"
$env:PATH = "$env:NVCC_CCBIN;$env:PATH"

# Build
cargo build --release

# Run
$env:HF_TOKEN = "hf_your_token_here"
.\target\release\gemma4-audio-test.exe
```

## Usage

| Control | Description |
|---------|-------------|
| **F9** (default) | Hold to record, release to transcribe |
| **Hotkey dropdown** | Change hotkey (F5-F12, Scroll Lock, Pause) |
| **Push-to-Talk / VAD** | Toggle between manual and automatic recording |
| **Auto-paste** | When checked, transcriptions are pasted into the focused app |

## Tips

- Hold the hotkey for at least **2 seconds** for best results
- Very short clips (<1s) may produce empty or garbled output
- The transcription prompt is set to **German** -- edit the prompt in `src/main.rs` for other languages
- First startup takes ~2 minutes (model quantization). Subsequent starts are slightly faster as model files are cached locally
- The model uses ~11.7GB VRAM with Q4K quantization

## Architecture

```
[Global Hotkey Thread] ---> [Audio Capture (cpal)] ---> [Transcription (mistral.rs/Gemma4)]
     (GetAsyncKeyState)          48kHz stereo              16kHz mono WAV
                                    |                           |
                              [Resample + Mono]          [Auto-paste (enigo)]
                                    |                           |
                              [Silence Padding]          [GUI Update (egui)]
```

## License

MIT
