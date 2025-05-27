use std::env;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::path::Path;

mod audio_io;
mod algorithms;
mod utils;
mod processing;

use algorithms::{MinMag, MaxMag, Sub, CopyPhase};
use utils::{parse_phase_source, parse_stft_mode};
use processing::process_with_streaming_detection;

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

fn setup_ctrl_c_handler() {
    let interrupted = Arc::new(AtomicBool::new(false));
    let interrupted_clone = interrupted.clone();
    
    ctrlc::set_handler(move || {
        eprintln!("\nInterrupted by user. Cleaning up...");
        interrupted_clone.store(true, Ordering::SeqCst);
        INTERRUPTED.store(true, Ordering::SeqCst);
    }).expect("Error setting Ctrl+C handler");
}

fn validate_input_files(input1: &str, input2: &str) -> Result<(), String> {
    // Check if files exist
    if !Path::new(input1).exists() {
        return Err(format!("Input file '{}' does not exist", input1));
    }
    if !Path::new(input2).exists() {
        return Err(format!("Input file '{}' does not exist", input2));
    }
    
    // Check if files are readable
    if let Err(e) = std::fs::File::open(input1) {
        return Err(format!("Cannot read input file '{}': {}", input1, e));
    }
    if let Err(e) = std::fs::File::open(input2) {
        return Err(format!("Cannot read input file '{}': {}", input2, e));
    }
    
    // Check if files have reasonable size (not empty, not too large)
    let metadata1 = std::fs::metadata(input1).map_err(|e| format!("Cannot get metadata for '{}': {}", input1, e))?;
    let metadata2 = std::fs::metadata(input2).map_err(|e| format!("Cannot get metadata for '{}': {}", input2, e))?;
    
    if metadata1.len() == 0 {
        return Err(format!("Input file '{}' is empty", input1));
    }
    if metadata2.len() == 0 {
        return Err(format!("Input file '{}' is empty", input2));
    }
    
    // Warn about very large files (>1GB)
    const MAX_REASONABLE_SIZE: u64 = 1024 * 1024 * 1024; // 1GB
    if metadata1.len() > MAX_REASONABLE_SIZE {
        eprintln!("Warning: Input file '{}' is very large ({:.1} MB). Consider using --streaming=on", 
                  input1, metadata1.len() as f64 / (1024.0 * 1024.0));
    }
    if metadata2.len() > MAX_REASONABLE_SIZE {
        eprintln!("Warning: Input file '{}' is very large ({:.1} MB). Consider using --streaming=on", 
                  input2, metadata2.len() as f64 / (1024.0 * 1024.0));
    }
    
    Ok(())
}

fn validate_output_path(output: &str) -> Result<(), String> {
    let output_path = Path::new(output);
    
    // Check if parent directory exists and is writable
    if let Some(parent) = output_path.parent() {
        if !parent.exists() {
            return Err(format!("Output directory '{}' does not exist", parent.display()));
        }
        
        // Try to create a temporary file to check write permissions
        let temp_path = parent.join(".specblender_write_test");
        match std::fs::File::create(&temp_path) {
            Ok(_) => {
                let _ = std::fs::remove_file(&temp_path); // Clean up
            }
            Err(e) => {
                return Err(format!("Cannot write to output directory '{}': {}", parent.display(), e));
            }
        }
    }
    
    // Check if output file already exists and warn
    if output_path.exists() {
        eprintln!("Warning: Output file '{}' already exists and will be overwritten", output);
    }
    
    Ok(())
}

pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::SeqCst)
}

macro_rules! dispatch_algorithm {
    ($algorithm:expr, $input1:expr, $input2:expr, $output:expr, $mono_mode:expr, $mono_post:expr, $n_fft:expr, $hop:expr, $window_type:expr, $stft_mode:expr, $phase_source:expr, $streaming_mode:expr, $use_float32:expr) => {
        match $algorithm {
            "min-mag" => {
                // Проверяем совместимость фазы с алгоритмом
                if let Err(e) = algorithms::validate_phase_compatibility("min-mag", $phase_source) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                process_with_streaming_detection::<MinMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source, $streaming_mode, $use_float32)
            },
            "max-mag" => {
                if let Err(e) = algorithms::validate_phase_compatibility("max-mag", $phase_source) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                process_with_streaming_detection::<MaxMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source, $streaming_mode, $use_float32)
            },
            "sub" => {
                if let Err(e) = algorithms::validate_phase_compatibility("sub", $phase_source) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                process_with_streaming_detection::<Sub>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source, $streaming_mode, $use_float32)
            },
            "copy-phase" => {
                if let Err(e) = algorithms::validate_phase_compatibility("copy-phase", $phase_source) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                process_with_streaming_detection::<CopyPhase>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source, $streaming_mode, $use_float32)
            },
            _ => Err(format!("Unknown algorithm '{}'", $algorithm).into())
        }
    };
}

fn print_help() {
    println!("SpecBlender v0.4.0");
    println!("Spectral audio processing toolkit");
    println!("https://github.com/aufr33/specblender");
    println!();
    println!("Usage:");
    println!("  specblender [ALGORITHM] input1.wav input2.wav output.wav [OPTIONS]");
    println!();
    println!("Algorithms:");
    println!("  min-mag          Minimum magnitude spectral blending (default)");
    println!("  max-mag          Maximum magnitude spectral blending");
    println!("  sub              Spectral subtraction (input1-input2)");
    println!("  copy-phase       Copy phase from input1 to input2 magnitude");
    println!();
    println!("Options:");
    println!("  --mono           Export mono output (average of both channels)");
    println!("  --mono-post      Process stereo, then mix to mono (matches manual mixing)");
    println!("  --n_fft=N        FFT window size (default: 2048, range: 32-65536)");
    println!("  --hop=H          Hop length (default: 512, must be ≤ n_fft)");
    println!("  --window=TYPE    Window function: hann or hamming (default: hann)");
    println!("  --stft=MODE      STFT mode: single or multi (default: single)");
    println!("                   multi: processes with two FFT sizes and blends frequency bands");
    println!("  --streaming=MODE Streaming mode: off, on, auto (default: auto)");
    println!("                   auto: use streaming for files longer than 10 minutes");
    println!("  --pcm_type=TYPE  PCM type: pcm24 or float32 (default: float32)");
    println!("  --phase=SOURCE   Phase source: auto, input1, input2 (default: auto)");
    println!("                   auto: use phase from signal with min/max magnitude");
    println!("                   input1: always use phase from first input (min-mag, max-mag only)");
    println!("                   input2: always use phase from second input (min-mag, max-mag only)");
    println!("                   Note: 'sub' and 'copy-phase' algorithms ignore this setting");
    println!("  --help           Show this message");
    println!("  --version        Show version");
}

fn main() {
    setup_ctrl_c_handler();
    
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 || args.iter().any(|a| a == "--help") {
        print_help();
        return;
    }
    
    if args.iter().any(|a| a == "--version") {
        println!("SpecBlender v0.4.0");
        return;
    }

    let mut opts: HashMap<String, String> = HashMap::new();
    let mut positional = Vec::new();

    for arg in &args[1..] {
        if arg.starts_with("--") {
            if let Some(eq_pos) = arg.find('=') {
                let (key, val) = arg.split_at(eq_pos);
                opts.insert(key.trim_start_matches("--").to_string(), val[1..].to_string());
            } else {
                opts.insert(arg.trim_start_matches("--").to_string(), "true".to_string());
            }
        } else {
            positional.push(arg.clone());
        }
    }

    let (algorithm_name, input1, input2, output) = if positional.len() == 3 {
        ("min-mag", &positional[0], &positional[1], &positional[2])
    } else if positional.len() == 4 {
        (positional[0].as_str(), &positional[1], &positional[2], &positional[3])
    } else {
        eprintln!("Error: Expected 3 or 4 file arguments");
        print_help();
        return;
    };

    // Validate input and output files early
    if let Err(e) = validate_input_files(input1, input2) {
        eprintln!("Error: {}", e);
        return;
    }
    
    if let Err(e) = validate_output_path(output) {
        eprintln!("Error: {}", e);
        return;
    }

    let mono_mode = opts.get("mono").map(|v| v == "true").unwrap_or(false);
    let mono_post = opts.get("mono-post").map(|v| v == "true").unwrap_or(false);
    let n_fft = opts.get("n_fft").and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
    let hop = opts.get("hop").and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
    let window_type = opts.get("window").map(|s| s.as_str()).unwrap_or("hann");
    let stft_mode = parse_stft_mode(opts.get("stft").map(|s| s.as_str()).unwrap_or("single"));
    let phase_source = parse_phase_source(opts.get("phase").map(|s| s.as_str()).unwrap_or("auto"));
    let streaming_mode = opts.get("streaming").map(|s| s.as_str()).unwrap_or("auto");
    let pcm_type = opts.get("pcm_type").map(|s| s.as_str()).unwrap_or("float32");
	
	if n_fft == 0 {
		eprintln!("Error: n_fft must be positive");
		return;
	}
	
	if n_fft < 32 {
		eprintln!("Error: n_fft must be at least 32 for reasonable frequency resolution");
		return;
	}
	
	if n_fft > 65536 {
		eprintln!("Error: n_fft must not exceed 65536 (memory and performance considerations)");
		return;
	}

	if hop == 0 {
		eprintln!("Error: hop must be positive");
		return;
	}

	if hop > n_fft {
		eprintln!("Error: hop ({}) cannot be larger than n_fft ({})", hop, n_fft);
		return;
	}

    if mono_mode && mono_post {
        eprintln!("Error: Cannot use both --mono and --mono-post at the same time");
        return;
    }

    // Validate PCM type
    let use_float32 = match pcm_type {
        "pcm24" => false,
        "float32" => true,
        _ => {
            eprintln!("Error: Invalid PCM type '{}'. Use 'pcm24' or 'float32'", pcm_type);
            return;
        }
    };

    // Validate streaming mode
    match streaming_mode {
        "off" | "on" | "auto" => {},
        _ => {
            eprintln!("Error: Invalid streaming mode '{}'. Use 'off', 'on', or 'auto'", streaming_mode);
            return;
        }
    }

    if use_float32 {
        println!("Output format: 32-bit float WAV (default)");
    } else {
        println!("Output format: 24-bit WAV");
    }

    if let Err(e) = dispatch_algorithm!(
        algorithm_name, input1, input2, output, mono_mode, mono_post, 
        n_fft, hop, window_type, stft_mode, phase_source, streaming_mode, use_float32
    ) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
    
    if is_interrupted() {
        eprintln!("Processing was interrupted");
        std::process::exit(130); // Standard exit code for Ctrl+C
    }
}