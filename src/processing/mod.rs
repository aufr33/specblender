// src/processing/mod.rs
use std::time::Instant;
use std::process::Command;
use std::fs;
use std::path::Path;
use indicatif::ProgressBar;
use rustfft::FftPlanner;

use crate::algorithms::Algorithm;
use crate::audio_io::{read_audio_file, write_wav_24bit, write_wav_24bit_streaming, cleanup_temp_files};
use crate::utils::{select_window, PhaseSource, StftMode};

const CHUNK_SIZE_SECONDS: f32 = 30.0;
const OVERLAP_SECONDS: f32 = 2.0;

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
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let algorithm = A::new();

    if matches!(stft_mode, StftMode::Multi) {
        println!("Multi-STFT mode: processing with dual FFT sizes and frequency band blending");
        return process_multi_stft_static::<A>(
            input1, input2, output, mono_mode, mono_post, window_type, phase_source
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

    let (a, sr1, tmp1) = read_audio_file(input1)?;
    let (b, sr2, tmp2) = read_audio_file(input2)?;

    let read_duration = read_start.elapsed();
    println!("File reading took: {:.2}s", read_duration.as_secs_f32());

    if sr1 != sr2 {
        return Err("Sample rates must match".into());
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        let a_mono: Vec<f32> = a[0].iter().zip(a[1].iter()).map(|(l, r)| (l + r) / 2.0).collect();
        let b_mono: Vec<f32> = b[0].iter().zip(b[1].iter()).map(|(l, r)| (l + r) / 2.0).collect();
        (vec![a_mono], vec![b_mono])
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
        println!("Processing channel {}...", ch + 1);
        let pb = ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = crate::utils::stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = crate::utils::stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        let y = crate::utils::istft(&result_spec, &win, hop, ifft.as_ref());
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
    
    write_wav_24bit(output, &final_output_channels, sr1)?;

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    let audio_length = a[0].len() as f32 / sr1 as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    cleanup_temp_files(vec![tmp1, tmp2]);

    Ok(())
}

pub fn process_audio_streaming<A: Algorithm>(
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
    let start_time = Instant::now();
    let algorithm = A::new();

    println!("Algorithm: {} (streaming mode)", algorithm.name());
    println!("Reading audio file headers...");

    let (a_full, sr1, tmp1) = read_audio_file(input1)?;
    let (b_full, sr2, tmp2) = read_audio_file(input2)?;

    if sr1 != sr2 {
        cleanup_temp_files(vec![tmp1, tmp2]);
        return Err("Sample rates must match".into());
    }

    let sample_rate = sr1;
    let total_samples = a_full[0].len().min(b_full[0].len());
    let chunk_samples = (CHUNK_SIZE_SECONDS * sample_rate as f32) as usize;
    let overlap_samples = (OVERLAP_SECONDS * sample_rate as f32) as usize;
    let step_samples = chunk_samples - overlap_samples;

    println!("Total duration: {:.1}s", total_samples as f32 / sample_rate as f32);
    println!("Processing in {:.1}s chunks with {:.1}s overlap", CHUNK_SIZE_SECONDS, OVERLAP_SECONDS);
    println!("Estimated memory usage: ~{:.1} MB", estimate_memory_usage(chunk_samples, n_fft));

    let win = select_window(window_type, n_fft);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut wav_writer = write_wav_24bit_streaming(output, sample_rate, if mono_post { 1 } else { 2 })?;

    let num_chunks = (total_samples + step_samples - 1) / step_samples;
    let overall_pb = ProgressBar::new(num_chunks as u64);
    overall_pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("Overall: [{wide_bar}] {pos}/{len} chunks ({eta})")
            .unwrap()
    );

    let mut overlap_buffer_left = vec![0.0f32; overlap_samples];
    let mut overlap_buffer_right = vec![0.0f32; overlap_samples];

    for chunk_idx in 0..num_chunks {
        let start_sample = chunk_idx * step_samples;
        let end_sample = (start_sample + chunk_samples).min(total_samples);
        let current_chunk_size = end_sample - start_sample;

        if current_chunk_size < n_fft {
            break;
        }

        let a_chunk = extract_chunk(&a_full, start_sample, current_chunk_size, mono_mode);
        let b_chunk = extract_chunk(&b_full, start_sample, current_chunk_size, mono_mode);

        let processed_chunk = process_chunk::<A>(
            &algorithm,
            &a_chunk,
            &b_chunk,
            &win,
            hop,
            fft.as_ref(),
            ifft.as_ref(),
            phase_source,
        )?;

        let faded_chunk = if chunk_idx == 0 {
            processed_chunk
        } else {
            apply_crossfade(&processed_chunk, &overlap_buffer_left, &overlap_buffer_right, overlap_samples)
        };

        if chunk_idx < num_chunks - 1 {
            let save_start = faded_chunk[0].len().saturating_sub(overlap_samples);
            overlap_buffer_left.copy_from_slice(&faded_chunk[0][save_start..]);
            if faded_chunk.len() > 1 {
                overlap_buffer_right.copy_from_slice(&faded_chunk[1][save_start..]);
            }
        }

        let write_samples = if chunk_idx == num_chunks - 1 {
            faded_chunk[0].len()
        } else {
            faded_chunk[0].len() - overlap_samples
        };

        let final_chunk = if mono_post && faded_chunk.len() > 1 {
            let mono_chunk: Vec<f32> = faded_chunk[0][..write_samples].iter()
                .zip(faded_chunk[1][..write_samples].iter())
                .map(|(l, r)| (l + r) / 2.0)
                .collect();
            vec![mono_chunk]
        } else {
            faded_chunk.into_iter()
                .map(|ch| ch[..write_samples].to_vec())
                .collect()
        };

        wav_writer.write_chunk(&final_chunk)?;
        overall_pb.inc(1);
    }

    wav_writer.finalize()?;
    overall_pb.finish();

    cleanup_temp_files(vec![tmp1, tmp2]);

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    let audio_length = total_samples as f32 / sample_rate as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    Ok(())
}

fn extract_chunk(
    full_data: &[Vec<f32>],
    start: usize,
    size: usize,
    mono_mode: bool,
) -> Vec<Vec<f32>> {
    if mono_mode {
        let mono_chunk: Vec<f32> = (start..start + size)
            .map(|i| (full_data[0][i] + full_data[1][i]) / 2.0)
            .collect();
        vec![mono_chunk]
    } else {
        vec![
            full_data[0][start..start + size].to_vec(),
            full_data[1][start..start + size].to_vec(),
        ]
    }
}

fn process_chunk<A: Algorithm>(
    algorithm: &A,
    a_chunk: &[Vec<f32>],
    b_chunk: &[Vec<f32>],
    win: &[f32],
    hop: usize,
    fft: &dyn rustfft::Fft<f32>,
    ifft: &dyn rustfft::Fft<f32>,
    phase_source: PhaseSource,
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut output_channels = Vec::new();

    for ch in 0..a_chunk.len() {
        let pb = ProgressBar::new(((a_chunk[ch].len() - win.len()) / hop + 1) as u64);
        pb.set_style(
            indicatif::ProgressStyle::default_spinner()
                .template("  Ch{}: {spinner} {pos}/{len}")
                .unwrap()
        );

        let spec_a = crate::utils::stft(&a_chunk[ch], win, hop, fft, &pb);
        let spec_b = crate::utils::stft(&b_chunk[ch], win, hop, fft, &pb);
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        let y = crate::utils::istft(&result_spec, win, hop, ifft);
        
        output_channels.push(y);
    }

    Ok(output_channels)
}

fn apply_crossfade(
    current: &[Vec<f32>],
    prev_left: &[f32],
    prev_right: &[f32],
    overlap_len: usize,
) -> Vec<Vec<f32>> {
    let mut result = current.to_vec();

    for i in 0..overlap_len.min(current[0].len()) {
        let fade_factor = i as f32 / overlap_len as f32;
        
        result[0][i] = prev_left[i] * (1.0 - fade_factor) + current[0][i] * fade_factor;
        
        if result.len() > 1 && !prev_right.is_empty() {
            result[1][i] = prev_right[i] * (1.0 - fade_factor) + current[1][i] * fade_factor;
        }
    }

    result
}

fn estimate_memory_usage(chunk_samples: usize, n_fft: usize) -> f32 {
    let stft_frames = chunk_samples / (n_fft / 4);
    let complex_size = 8;
    
    let chunk_data = chunk_samples * 2 * 4;
    let stft_data = stft_frames * n_fft * complex_size * 2;
    let output_data = chunk_samples * 2 * 4;
    
    (chunk_data + stft_data + output_data) as f32 / (1024.0 * 1024.0)
}

fn process_multi_stft_static<A: Algorithm>(
    input1: &str,
    input2: &str,
    output: &str,
    mono_mode: bool,
    mono_post: bool,
    window_type: &str,
    phase_source: PhaseSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    println!("Step 1/3: Processing with FFT size 4096 (low frequencies)...");
    
    let temp_low = format!("{}_temp_low.wav", output.trim_end_matches(".wav"));
    let temp_high = format!("{}_temp_high.wav", output.trim_end_matches(".wav"));
    
    process_single_stft_static::<A>(
        input1,
        input2,
        &temp_low,
        mono_mode,
        mono_post,
        4096,
        1024,
        window_type,
        phase_source,
    )?;

    println!("Step 2/3: Processing with FFT size 1024 (high frequencies)...");
    
    process_single_stft_static::<A>(
        input1,
        input2,
        &temp_high,
        mono_mode,
        mono_post,
        1024,
        256,
        window_type,
        phase_source,
    )?;

    println!("Step 3/3: Blending frequency bands with ffmpeg...");
    
    let crossover_freq = 1000;
    
    let ffmpeg_result = Command::new("ffmpeg")
        .args([
            "-y",
            "-v", "error",
            "-i", &temp_low,
            "-i", &temp_high,
            "-filter_complex",
            &format!(
                "[0:a]highpass=f={}:poles=2[hp_low];[0:a][hp_low]amerge,pan=stereo|c0=c0-c2|c1=c1-c3[lowpass_result];[1:a]highpass=f={}:poles=2[hp_high];[lowpass_result][hp_high]amix=inputs=2:duration=longest:normalize=0,volume=1.0",
                crossover_freq, crossover_freq
            ),
            "-c:a", "pcm_s24le",
            output
        ])
        .output();

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

    let (audio_data, sample_rate, tmp_file) = read_audio_file(input1)?;
    let audio_length = audio_data[0].len() as f32 / sample_rate as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);

    cleanup_temp_files(vec![tmp_file]);

    Ok(())
}

fn process_single_stft_static<A: Algorithm>(
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
    let algorithm = A::new();
    let win = select_window(window_type, n_fft);

    let (a, sr1, tmp1) = read_audio_file(input1)?;
    let (b, sr2, tmp2) = read_audio_file(input2)?;

    if sr1 != sr2 {
        return Err("Sample rates must match".into());
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        let a_mono: Vec<f32> = a[0].iter().zip(a[1].iter()).map(|(l, r)| (l + r) / 2.0).collect();
        let b_mono: Vec<f32> = b[0].iter().zip(b[1].iter()).map(|(l, r)| (l + r) / 2.0).collect();
        (vec![a_mono], vec![b_mono])
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

    let mut output_channels = Vec::new();

    for ch in 0..a.len() {
        let pb = ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = crate::utils::stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = crate::utils::stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        let y = crate::utils::istft(&result_spec, &win, hop, ifft.as_ref());
        output_channels.push(y);
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
    
    write_wav_24bit(output, &final_output_channels, sr1)?;

    cleanup_temp_files(vec![tmp1, tmp2]);

    Ok(())
}