# SpecBlender

A Rust-based spectral sound processing toolkit using STFT.

## Requirements

FFmpeg must be available in PATH

## Usage

```bash
specblender [ALGORITHM] input1.wav input2.wav output.wav [OPTIONS]
```

## Algorithms

### **min-mag** (Default)
Minimum magnitude spectral blending. Merges two audio files by selecting the quieter spectral components at each frequency bin.

<img src="minmag.png" width="615" height="262"/>

### **max-mag** 
Maximum magnitude spectral blending. Selects the louder spectral components at each frequency bin.

### **sub**
Spectral subtraction (input1 - input2).

### **copy-phase**
Copies phase information from input1 to the magnitude of input2.

## Options

| Option | Description |
|--------|-------------|
| `--mono` | Export mono output (average channels before processing) |
| `--mono-post` | Process in stereo, then mix to mono (matches manual mixing) |
| `--n_fft=N` | FFT window size (default: 2048) |
| `--hop=H` | Hop length in samples (default: 512) |
| `--window=TYPE` | Window function: `hann` or `hamming` (default: hann) |
| `--stft=MODE` | STFT processing mode (see below) |
| `--phase=SOURCE` | Phase source control (see below) |

### STFT Modes

- **`single`** (default) - Standard STFT processing
- **`multi`** - Multi-resolution processing:
  - Low frequencies: FFT=4096 for better frequency resolution
  - High frequencies: FFT=1024 for better time resolution
  - Intelligent frequency band blending via FFmpeg filters

### Phase Control

- **`auto`** (default) - Use phase from signal with appropriate magnitude (algorithm-dependent)
- **`input1`** - Always use phase from first input file
- **`input2`** - Always use phase from second input file

## Building from Source

```bash
git clone https://github.com/aufr33/specblender
cd specblender
cargo build --release
```
