use std::env;
use std::collections::HashMap;

mod audio_io;
mod algorithms;
mod utils;
mod processing;

use algorithms::{MinMag, MaxMag, Sub, CopyPhase};
use utils::{parse_phase_source, parse_stft_mode, StftMode};
use processing::{process_audio_static, process_audio_streaming};

macro_rules! dispatch_algorithm {
    ($algorithm:expr, $input1:expr, $input2:expr, $output:expr, $mono_mode:expr, $mono_post:expr, $n_fft:expr, $hop:expr, $window_type:expr, $stft_mode:expr, $phase_source:expr, $streaming:expr) => {
        if $streaming {
            match $algorithm {
                "min-mag" => process_audio_streaming::<MinMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $phase_source),
                "max-mag" => process_audio_streaming::<MaxMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $phase_source),
                "sub" => process_audio_streaming::<Sub>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $phase_source),
                "copy-phase" => process_audio_streaming::<CopyPhase>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $phase_source),
                _ => Err(format!("Unknown algorithm '{}'", $algorithm).into())
            }
        } else {
            match $algorithm {
                "min-mag" => process_audio_static::<MinMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source),
                "max-mag" => process_audio_static::<MaxMag>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source),
                "sub" => process_audio_static::<Sub>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source),
                "copy-phase" => process_audio_static::<CopyPhase>($input1, $input2, $output, $mono_mode, $mono_post, $n_fft, $hop, $window_type, $stft_mode, $phase_source),
                _ => Err(format!("Unknown algorithm '{}'", $algorithm).into())
            }
        }
    };
}

fn print_help() {
    println!("SpecBlender v0.3.0");
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
    println!("  --n_fft=N        FFT window size (default: 2048)");
    println!("  --hop=H          Hop length (default: 512)");
    println!("  --window=TYPE    Window function: hann or hamming (default: hann)");
    println!("  --stft=MODE      STFT mode: single or multi (default: single)");
    println!("                   multi: processes with two FFT sizes and blends frequency bands");
    println!("  --phase=SOURCE   Phase source: auto, input1, input2 (default: auto)");
    println!("                   auto: use phase from signal with min/max magnitude");
    println!("                   input1: always use phase from first input");
    println!("                   input2: always use phase from second input");
    println!("  --help           Show this message");
    println!("  --version        Show version");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 || args.iter().any(|a| a == "--help") {
        print_help();
        return;
    }
    
    if args.iter().any(|a| a == "--version") {
        println!("SpecBlender v0.3.0");
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

    let mono_mode = opts.get("mono").map(|v| v == "true").unwrap_or(false);
    let mono_post = opts.get("mono-post").map(|v| v == "true").unwrap_or(false);
    let n_fft = opts.get("n_fft").and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
    let hop = opts.get("hop").and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
    let window_type = opts.get("window").map(|s| s.as_str()).unwrap_or("hann");
    let stft_mode = parse_stft_mode(opts.get("stft").map(|s| s.as_str()).unwrap_or("single"));
    let phase_source = parse_phase_source(opts.get("phase").map(|s| s.as_str()).unwrap_or("auto"));
    let streaming_mode = opts.get("streaming").map(|v| v == "true").unwrap_or(false);

    if mono_mode && mono_post {
        eprintln!("Error: Cannot use both --mono and --mono-post at the same time");
        return;
    }

    if streaming_mode && matches!(stft_mode, StftMode::Multi) {
        eprintln!("Error: Streaming mode is not compatible with multi-STFT mode");
        return;
    }

    if let Err(e) = dispatch_algorithm!(
        algorithm_name, input1, input2, output, mono_mode, mono_post, 
        n_fft, hop, window_type, stft_mode, phase_source, streaming_mode
    ) {
        eprintln!("Error: {}", e);
    }
}
