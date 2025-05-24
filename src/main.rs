use std::env;
use std::collections::HashMap;

mod audio_io;
mod algorithms;
mod utils;

use audio_io::{read_audio_file, write_wav_24bit, resample_if_needed};
use algorithms::{Algorithm, MinMag, MaxMag, Sub, CopyPhase};
use utils::{select_window, PhaseSource, parse_phase_source, StftMode, parse_stft_mode};

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

    let algorithm: Box<dyn Algorithm> = match algorithm_name {
        "min-mag" => Box::new(MinMag),
        "max-mag" => Box::new(MaxMag),
        "sub" => Box::new(Sub),
        "copy-phase" => Box::new(CopyPhase),
        _ => {
            eprintln!("Error: Unknown algorithm '{}'", algorithm_name);
            eprintln!("Available algorithms: min-mag, max-mag, sub, copy-phase");
            return;
        }
    };

    let mono_mode = opts.get("mono").map(|v| v == "true").unwrap_or(false);
    let mono_post = opts.get("mono-post").map(|v| v == "true").unwrap_or(false);
    let n_fft = opts.get("n_fft").and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
    let hop = opts.get("hop").and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
    let window_type = opts.get("window").map(|s| s.as_str()).unwrap_or("hann");
    let stft_mode = parse_stft_mode(opts.get("stft").map(|s| s.as_str()).unwrap_or("single"));
    let phase_source = parse_phase_source(opts.get("phase").map(|s| s.as_str()).unwrap_or("auto"));

    if mono_mode && mono_post {
        eprintln!("Error: Cannot use both --mono and --mono-post at the same time");
        return;
    }

    if let Err(e) = process_audio(
        algorithm,
        input1,
        input2,
        output,
        mono_mode,
        mono_post,
        n_fft,
        hop,
        window_type,
        stft_mode,
        phase_source,
    ) {
        eprintln!("Error: {}", e);
    }
}

fn process_audio(
    algorithm: Box<dyn Algorithm>,
    input1: &str,
    input2: &str,
    output: &str,
    mono_mode: bool,
    mono_post: bool,
    n_fft: usize,
    hop: usize,
    window_type: &str,
    stft_mode: StftMode,
    phase_source: PhaseSource,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    use indicatif::ProgressBar;
    use rustfft::FftPlanner;

    let start_time = Instant::now();

    // If multi-stft, perform dual processing
    if matches!(stft_mode, StftMode::Multi) {
        println!("Multi-STFT mode: processing with dual FFT sizes and frequency band blending");
        return process_multi_stft(
            algorithm,
            input1,
            input2,
            output,
            mono_mode,
            mono_post,
            window_type,
            phase_source,
        );
    }

    let win = select_window(window_type, n_fft);

    let phase_info = match phase_source {
        PhaseSource::MinMagnitude => "auto (from selected signal)",
        PhaseSource::Input1 => "from input1",
        PhaseSource::Input2 => "from input2",
    };

    println!("Algorithm: {}", algorithm.name());
    println!("STFT mode: single resolution");
    if mono_mode {
        println!("Mode: Pre-processing mono (mix first, then process)");
    } else if mono_post {
        println!("Mode: Post-processing mono (process stereo, then mix)");
    } else {
        println!("Mode: Stereo processing");
    }
    println!("Phase source: {}", phase_info);

    println!("Reading audio files...");
    let read_start = Instant::now();

    let (mut a, sr1, _) = read_audio_file(input1)?;
    let (mut b, sr2, _) = read_audio_file(input2)?;

    let read_duration = read_start.elapsed();
    println!("File reading took: {:.2}s", read_duration.as_secs_f32());

    let target_sr = sr1.max(sr2);

    if sr1 != target_sr {
        a = resample_if_needed(a, sr1, target_sr);
        println!("Resampled file 1 to {}Hz", target_sr);
    }
    if sr2 != target_sr {
        b = resample_if_needed(b, sr2, target_sr);
        println!("Resampled file 2 to {}Hz", target_sr);
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        let a_mono: Vec<f32> = a.iter().fold(vec![0.0; a[0].len()], |mut acc, ch| {
            acc.iter_mut().zip(ch.iter()).for_each(|(a, b)| *a += *b);
            acc
        }).into_iter().map(|x| x / a.len() as f32).collect();

        let b_mono: Vec<f32> = b.iter().fold(vec![0.0; b[0].len()], |mut acc, ch| {
            acc.iter_mut().zip(ch.iter()).for_each(|(a, b)| *a += *b);
            acc
        }).into_iter().map(|x| x / b.len() as f32).collect();

        (vec![a_mono], vec![b_mono])
    } else {
        while a.len() < 2 { a.push(a[0].clone()); }
        while b.len() < 2 { b.push(b[0].clone()); }
        (a, b)
    };

    for i in 0..a.len() {
        let len = a[i].len().min(b[i].len());
        a[i].truncate(len);
        b[i].truncate(len);
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    println!("Processing audio with STFT...");

    let mut output_channels = vec![Vec::new(); a.len()];

    for ch in 0..a.len() {
        println!("Processing channel {}...", ch + 1);
        
        let pb = ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = utils::stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = utils::stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        pb.finish();
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        output_channels[ch] = utils::istft(&result_spec, &win, hop, ifft.as_ref());
    }

    println!("Writing output file...");
    
    let final_output_channels = if mono_post {
        println!("Converting stereo output to mono...");
        let mono_channel: Vec<f32> = output_channels[0].iter()
            .zip(output_channels[1].iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect();
        vec![mono_channel]
    } else {
        output_channels
    };
    
    write_wav_24bit(output, &final_output_channels, target_sr)?;

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    let audio_length = a[0].len() as f32 / target_sr as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    Ok(())
}

fn process_multi_stft(
    algorithm: Box<dyn Algorithm>,
    input1: &str,
    input2: &str,
    output: &str,
    mono_mode: bool,
    mono_post: bool,
    window_type: &str,
    phase_source: PhaseSource,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::process::Command;
    use std::fs;
    use std::path::Path;
    use std::time::Instant;

    let start_time = Instant::now();

    println!("Step 1/3: Processing with FFT size 4096 (low frequencies)...");
    
    // Create temporary files
    let temp_low = format!("{}_temp_low.wav", output.trim_end_matches(".wav"));
    let temp_high = format!("{}_temp_high.wav", output.trim_end_matches(".wav"));
    
    // Processing with large FFT (4096) for low frequencies
    process_single_stft(
        &*algorithm,
        input1,
        input2,
        &temp_low,
        mono_mode,
        mono_post,
        4096,  // Large FFT for better frequency resolution on LF
        1024,  // hop = n_fft / 4
        window_type,
        phase_source,
    )?;

    println!("Step 2/3: Processing with FFT size 1024 (high frequencies)...");
    
    // Processing with small FFT (1024) for high frequencies
    process_single_stft(
        &*algorithm,
        input1,
        input2,
        &temp_high,
        mono_mode,
        mono_post,
        1024,  // Small FFT for better time resolution on HF
        256,   // hop = n_fft / 4
        window_type,
        phase_source,
    )?;

    println!("Step 3/3: Blending frequency bands with ffmpeg...");
    
    // Define crossover frequency (approximately 1 kHz for better separation)
    let crossover_freq = 1000;
    
    // Mix through ffmpeg using highpass filters and subtraction
    let ffmpeg_result = Command::new("ffmpeg")
        .args([
            "-y", // Overwrite output file
            "-v", "error", // Hide banner, show only errors
            "-i", &temp_low,   // input 0 - signal for low frequencies
            "-i", &temp_high,  // input 1 - signal for high frequencies
            "-filter_complex",
            &format!(
                "[0:a]highpass=f={}:poles=2[hp_low];[0:a][hp_low]amerge,pan=stereo|c0=c0-c2|c1=c1-c3[lowpass_result];[1:a]highpass=f={}:poles=2[hp_high];[lowpass_result][hp_high]amix=inputs=2:duration=longest:normalize=0,volume=1.0",
                crossover_freq, crossover_freq
            ),
            "-c:a", "pcm_s24le", // 24-bit PCM to match our format
            output
        ])
        .output();

    // Delete temporary files
    if Path::new(&temp_low).exists() {
        let _ = fs::remove_file(&temp_low);
    }
    if Path::new(&temp_high).exists() {
        let _ = fs::remove_file(&temp_high);
    }

    match ffmpeg_result {
        Ok(output_result) => {
            if output_result.status.success() {
                println!("Multi-STFT processing completed successfully!");
                println!("Low-frequency processing: FFT=4096, Crossover: {}Hz (highpass subtraction)", crossover_freq);
                println!("High-frequency processing: FFT=1024, Crossover: {}Hz (highpass filter)", crossover_freq);
            } else {
                let error_msg = String::from_utf8_lossy(&output_result.stderr);
                return Err(format!("ffmpeg failed: {}", error_msg).into());
            }
        }
        Err(e) => {
            return Err(format!("Failed to run ffmpeg: {}. Make sure ffmpeg is installed and available in PATH.", e).into());
        }
    }

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    // For audio time we need to read one of the files
    let (audio_data, sample_rate, _) = read_audio_file(input1)?;
    let audio_length = audio_data[0].len() as f32 / sample_rate as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    Ok(())
}

fn process_single_stft(
    algorithm: &dyn Algorithm,
    input1: &str,
    input2: &str,
    output: &str,
    mono_mode: bool,
    mono_post: bool,
    n_fft: usize,
    hop: usize,
    window_type: &str,
    phase_source: PhaseSource,
) -> Result<(), Box<dyn std::error::Error>> {
    use rustfft::FftPlanner;

    let win = select_window(window_type, n_fft);

    let (mut a, sr1, _) = read_audio_file(input1)?;
    let (mut b, sr2, _) = read_audio_file(input2)?;

    let target_sr = sr1.max(sr2);

    if sr1 != target_sr {
        a = resample_if_needed(a, sr1, target_sr);
    }
    if sr2 != target_sr {
        b = resample_if_needed(b, sr2, target_sr);
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        let a_mono: Vec<f32> = a.iter().fold(vec![0.0; a[0].len()], |mut acc, ch| {
            acc.iter_mut().zip(ch.iter()).for_each(|(a, b)| *a += *b);
            acc
        }).into_iter().map(|x| x / a.len() as f32).collect();

        let b_mono: Vec<f32> = b.iter().fold(vec![0.0; b[0].len()], |mut acc, ch| {
            acc.iter_mut().zip(ch.iter()).for_each(|(a, b)| *a += *b);
            acc
        }).into_iter().map(|x| x / b.len() as f32).collect();

        (vec![a_mono], vec![b_mono])
    } else {
        while a.len() < 2 { a.push(a[0].clone()); }
        while b.len() < 2 { b.push(b[0].clone()); }
        (a, b)
    };

    for i in 0..a.len() {
        let len = a[i].len().min(b[i].len());
        a[i].truncate(len);
        b[i].truncate(len);
    }

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut output_channels = vec![Vec::new(); a.len()];

    for ch in 0..a.len() {
        let pb = indicatif::ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = utils::stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = utils::stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        pb.finish();
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        output_channels[ch] = utils::istft(&result_spec, &win, hop, ifft.as_ref());
    }
    
    let final_output_channels = if mono_post {
        let mono_channel: Vec<f32> = output_channels[0].iter()
            .zip(output_channels[1].iter())
            .map(|(l, r)| (l + r) / 2.0)
            .collect();
        vec![mono_channel]
    } else {
        output_channels
    };
    
    write_wav_24bit(output, &final_output_channels, target_sr)?;

    Ok(())
}
