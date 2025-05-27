use std::time::Instant;
use indicatif::ProgressBar;
use rustfft::FftPlanner;

use crate::algorithms::Algorithm;
use crate::audio_io::{read_audio_file_with_mode, write_wav_24bit, write_wav_32bit_float, cleanup_temp_files};
use crate::utils::{select_window, PhaseSource, StftMode};
use super::{check_interrupted, process_multi_stft_static};

pub fn process_audio_static<A: Algorithm>(
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
    use_float32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let algorithm = A::new();

    if matches!(stft_mode, StftMode::Multi) {
        println!("Multi-STFT mode: processing with dual FFT sizes and frequency band blending");
        return process_multi_stft_static::<A>(
            input1, input2, output, mono_mode, mono_post, window_type, phase_source, use_float32
        );
    }

    let win = select_window(window_type, n_fft);

    println!("Algorithm: {}", algorithm.name());
    println!("STFT mode: single resolution");
    if mono_mode {
        println!("Mode: Pre-processing mono (mix first, then process)");
    } else if mono_post {
        println!("Mode: Post-processing mono (process stereo, then mix)");
    } else {
        println!("Mode: Stereo processing");
    }
    
    crate::algorithms::print_phase_usage_info(algorithm.name(), phase_source);

    println!("Reading audio files...");
    let read_start = Instant::now();

    let (a, sr1, tmp1) = read_audio_file_with_mode(input1, mono_mode)?;
    let (b, sr2, tmp2) = read_audio_file_with_mode(input2, mono_mode)?;

    let read_duration = read_start.elapsed();
    println!("File reading took: {:.2}s", read_duration.as_secs_f32());

    if sr1 != sr2 {
        return Err("Sample rates must match".into());
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        // Already mono from ffmpeg
        (vec![a[0].clone()], vec![b[0].clone()])
    } else {
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
    let mut output_channels = Vec::new();

    for ch in 0..a.len() {
        check_interrupted()?;
        println!("Processing channel {}...", ch + 1);
        let pb = ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = crate::utils::stft(&a[ch], &win, hop, fft.as_ref(), &pb)?;
        let spec_b = crate::utils::stft(&b[ch], &win, hop, fft.as_ref(), &pb)?;
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        let mut y = crate::utils::istft(&result_spec, &win, hop, ifft.as_ref());
        
        let samples_to_mute = (sr1 as f32 * 0.006) as usize; // 6ms
        let mute_count = samples_to_mute.min(y.len());
        for i in 0..mute_count {
            y[i] = 0.0;
        }
        
        output_channels.push(y);
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
    
    // Write output based on PCM type
    if use_float32 {
        write_wav_32bit_float(output, &final_output_channels, sr1)?;
        println!("Output written as 32-bit float WAV");
    } else {
        write_wav_24bit(output, &final_output_channels, sr1)?;
        println!("Output written as 24-bit WAV");
    }

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    let audio_length = a[0].len() as f32 / sr1 as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    cleanup_temp_files(vec![tmp1, tmp2]);

    Ok(())
}
