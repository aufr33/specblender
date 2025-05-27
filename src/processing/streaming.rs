use std::time::Instant;
use indicatif::ProgressBar;
use rustfft::FftPlanner;

use crate::algorithms::Algorithm;
use crate::audio_io::{read_audio_file_with_mode, write_wav_24bit_streaming, write_wav_32bit_float_streaming, cleanup_temp_files};
use crate::utils::{select_window, PhaseSource};
use super::check_interrupted;

const CHUNK_SIZE_SECONDS: f32 = 120.0;
const OVERLAP_SECONDS: f32 = 1.0;

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
    use_float32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let algorithm = A::new();

    println!("Algorithm: {} (streaming mode)", algorithm.name());
    
    crate::algorithms::print_phase_usage_info(algorithm.name(), phase_source);
    
    println!("Reading audio file headers...");

    let (a_full, sr1, tmp1) = read_audio_file_with_mode(input1, mono_mode)?;
    let (b_full, sr2, tmp2) = read_audio_file_with_mode(input2, mono_mode)?;

    if sr1 != sr2 {
        cleanup_temp_files(vec![tmp1, tmp2]);
        return Err("Sample rates must match".into());
    }

    let sample_rate = sr1;
    let total_samples = a_full[0].len().min(b_full[0].len());
    let chunk_samples = (CHUNK_SIZE_SECONDS * sample_rate as f32) as usize;
    let overlap_samples = (OVERLAP_SECONDS * sample_rate as f32) as usize;
    let step_samples = chunk_samples - overlap_samples;
    
    let num_chunks = (total_samples + step_samples - 1) / step_samples;

    println!("Total duration: {:.1}s", total_samples as f32 / sample_rate as f32);
    println!("Processing in {:.1}s chunks with {:.1}s overlap", CHUNK_SIZE_SECONDS, OVERLAP_SECONDS);
    println!("Chunk samples: {}, Step samples: {}, Overlap samples: {}", chunk_samples, step_samples, overlap_samples);

    let win = select_window(window_type, n_fft);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    // Determine output channel count
    let output_channels = if mono_mode || mono_post { 1 } else { 2 };
    
    // Create appropriate writer based on PCM type
    let mut wav_writer = if use_float32 {
        write_wav_32bit_float_streaming(output, sample_rate, output_channels)?
    } else {
        write_wav_24bit_streaming(output, sample_rate, output_channels)?
    };

    let overall_pb = ProgressBar::new(num_chunks as u64);
    overall_pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("Overall: [{wide_bar}] {pos}/{len} chunks ({eta})")
            .unwrap()
    );

    let mut total_written_samples = 0usize;
    let mut prev_chunk_tail: Vec<Vec<f32>> = Vec::new();

    for chunk_idx in 0..num_chunks {
        check_interrupted()?;
        
        let start_sample = total_written_samples;
        let end_sample = (start_sample + chunk_samples).min(total_samples);
        let current_chunk_size = end_sample - start_sample;

        if current_chunk_size < n_fft {
            break;
        }

        let a_chunk = extract_chunk(&a_full, start_sample, current_chunk_size, mono_mode);
        let b_chunk = extract_chunk(&b_full, start_sample, current_chunk_size, mono_mode);

        let mut processed_chunk = process_chunk::<A>(
            &algorithm,
            &a_chunk,
            &b_chunk,
            &win,
            hop,
            fft.as_ref(),
            ifft.as_ref(),
            phase_source,
            chunk_idx == 0, // is_first_chunk
            sample_rate,    // sample_rate
        )?;

        // Apply crossfade with previous chunk tail if exists
        if chunk_idx > 0 && !prev_chunk_tail.is_empty() {
            let fade_len = overlap_samples.min(prev_chunk_tail[0].len()).min(processed_chunk[0].len());
            
            for ch in 0..processed_chunk.len() {
                if ch < prev_chunk_tail.len() {
                    for i in 0..fade_len {
                        let fade_factor = i as f32 / fade_len as f32;
                        processed_chunk[ch][i] = prev_chunk_tail[ch][i] * (1.0 - fade_factor) + 
                                               processed_chunk[ch][i] * fade_factor;
                    }
                }
            }
        }

        // For next iteration: save tail if not last chunk
        if chunk_idx < num_chunks - 1 && processed_chunk[0].len() > overlap_samples {
            let tail_start = processed_chunk[0].len() - overlap_samples;
            prev_chunk_tail = processed_chunk.iter()
                .map(|ch| ch[tail_start..].to_vec())
                .collect();
        } else {
            prev_chunk_tail.clear();
        }

        // Write the chunk (but skip the tail part if not last chunk)
        let samples_to_write = if chunk_idx == num_chunks - 1 {
            processed_chunk[0].len()
        } else {
            processed_chunk[0].len() - overlap_samples
        };

        let chunk_to_write: Vec<Vec<f32>> = processed_chunk.into_iter()
            .map(|ch| ch[..samples_to_write].to_vec())
            .collect();

        // Apply mono conversion if needed
        let output_chunk = if mono_post && chunk_to_write.len() > 1 {
            let mono_chunk: Vec<f32> = chunk_to_write[0].iter()
                .zip(chunk_to_write[1].iter())
                .map(|(l, r)| (l + r) / 2.0)
                .collect();
            vec![mono_chunk]
        } else {
            chunk_to_write
        };

        wav_writer.write_chunk(&output_chunk)?;
        total_written_samples += output_chunk[0].len();
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
        // Already mono from ffmpeg
        vec![full_data[0][start..start + size].to_vec()]
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
    is_first_chunk: bool, 
    sample_rate: u32, 
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let mut output_channels = Vec::new();

    for ch in 0..a_chunk.len() {
        let pb = ProgressBar::new(((a_chunk[ch].len() - win.len()) / hop + 1) as u64);
        pb.set_draw_target(indicatif::ProgressDrawTarget::hidden());

        let spec_a = crate::utils::stft(&a_chunk[ch], win, hop, fft, &pb)?;
        let spec_b = crate::utils::stft(&b_chunk[ch], win, hop, fft, &pb)?;
        
        let result_spec = algorithm.process(&spec_a, &spec_b, phase_source);
        let mut y = crate::utils::istft(&result_spec, win, hop, ifft);
        
        if is_first_chunk {
            let samples_to_mute = (sample_rate as f32 * 0.006) as usize; // 6ms
            let mute_count = samples_to_mute.min(y.len());
            for i in 0..mute_count {
                y[i] = 0.0;
            }
        }
        
        output_channels.push(y);
    }

    Ok(output_channels)
}
