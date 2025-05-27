use std::time::Instant;
use std::process::Command;
use std::fs;
use std::path::Path;
use indicatif::ProgressBar;
use rustfft::FftPlanner;

use crate::algorithms::Algorithm;
use crate::audio_io::{read_audio_file_with_mode, write_wav_24bit, write_wav_32bit_float, cleanup_temp_files};
use crate::utils::{select_window, PhaseSource};

pub fn process_multi_stft_static<A: Algorithm>(
    input1: &str,
    input2: &str,
    output: &str,
    mono_mode: bool,
    mono_post: bool,
    window_type: &str,
    phase_source: PhaseSource,
    use_float32: bool,
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
        use_float32,
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
        use_float32,
    )?;

    println!("Step 3/3: Blending frequency bands with ffmpeg...");
    
    let crossover_freq = 1000;
    
	let filter_complex = if mono_mode {
		format!(
			"[0:a]highpass=f={}:poles=2[hp_low];[0:a][hp_low]amerge,pan=mono|c0=c0-c1[lowpass_result];[1:a]highpass=f={}:poles=2[hp_high];[lowpass_result][hp_high]amix=inputs=2:duration=longest:normalize=0,volume=1.0",
			crossover_freq, crossover_freq
		)
	} else if mono_post {
		format!(
			"[0:a]highpass=f={}:poles=2[hp_low];[0:a][hp_low]amerge,pan=mono|c0=c0-c1[lowpass_result];[1:a]highpass=f={}:poles=2[hp_high];[lowpass_result][hp_high]amix=inputs=2:duration=longest:normalize=0,volume=1.0",
			crossover_freq, crossover_freq
		)
	} else {
		format!(
			"[0:a]highpass=f={}:poles=2[hp_low];[0:a][hp_low]amerge,pan=stereo|c0=c0-c2|c1=c1-c3[lowpass_result];[1:a]highpass=f={}:poles=2[hp_high];[lowpass_result][hp_high]amix=inputs=2:duration=longest:normalize=0,volume=1.0",
			crossover_freq, crossover_freq
		)
	};
    
    let ffmpeg_result = Command::new("ffmpeg")
        .args([
            "-y",
            "-v", "error",
            "-i", &temp_low,
            "-i", &temp_high,
            "-filter_complex", &filter_complex,
            "-c:a", if use_float32 { "pcm_f32le" } else { "pcm_s24le" },
            output
        ])
        .output();

    // Clean up temporary files
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
                if mono_mode {
                    println!("Mode: Mono processing");
                } else if mono_post {
                    println!("Mode: Post-process mono");
                } else {
                    println!("Mode: Stereo processing");
                }
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

    let (audio_data, sample_rate, tmp_file) = read_audio_file_with_mode(input1, mono_mode)?;
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
    use_float32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let algorithm = A::new();
    let win = select_window(window_type, n_fft);

    let (a, sr1, tmp1) = read_audio_file_with_mode(input1, mono_mode)?;
    let (b, sr2, tmp2) = read_audio_file_with_mode(input2, mono_mode)?;

    if sr1 != sr2 {
        return Err("Sample rates must match".into());
    }

    let (mut a, mut b): (Vec<Vec<f32>>, Vec<Vec<f32>>) = if mono_mode {
        // Уже моно из ffmpeg
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

    let mut output_channels = Vec::new();

    for ch in 0..a.len() {
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
    
    let final_output_channels = if mono_post {
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
    } else {
        write_wav_24bit(output, &final_output_channels, sr1)?;
    }

    cleanup_temp_files(vec![tmp1, tmp2]);

    Ok(())
}
