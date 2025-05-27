use crate::algorithms::Algorithm;
use crate::audio_io::{read_audio_file_with_mode, cleanup_temp_files};
use crate::utils::{PhaseSource, StftMode};

mod static_mode;
mod streaming;
mod multi_stft;

pub use static_mode::process_audio_static;
pub use streaming::process_audio_streaming;
pub use multi_stft::process_multi_stft_static;

const AUTO_STREAMING_THRESHOLD_MINUTES: f32 = 10.0;

fn check_interrupted() -> Result<(), Box<dyn std::error::Error>> {
    if crate::is_interrupted() {
        return Err("Processing interrupted by user".into());
    }
    Ok(())
}

pub fn process_with_streaming_detection<A: Algorithm>(
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
    streaming_mode: &str,
    use_float32: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check compatibility first
    if matches!(stft_mode, StftMode::Multi) && (streaming_mode == "on" || streaming_mode == "auto") {
        if streaming_mode == "on" {
            return Err("Streaming mode is not compatible with multi-STFT mode".into());
        }
        // For auto mode with multi-STFT, force static processing regardless of file length
        println!("Multi-STFT mode detected, forcing static processing");
        return process_audio_static::<A>(input1, input2, output, mono_mode, mono_post, n_fft, hop, window_type, stft_mode, phase_source, use_float32);
    }

    let use_streaming = match streaming_mode {
        "off" => false,
        "on" => true,
        "auto" => {
            // Quick check of file duration for auto mode
            let (audio_data, sample_rate, tmp_file) = read_audio_file_with_mode(input1, mono_mode)?;
            let duration_minutes = audio_data[0].len() as f32 / sample_rate as f32 / 60.0;
            cleanup_temp_files(vec![tmp_file]);
            
            if duration_minutes > AUTO_STREAMING_THRESHOLD_MINUTES {
                println!("Auto-detected long audio ({:.1} min), using streaming mode", duration_minutes);
                true
            } else {
                println!("Audio duration: {:.1} min, using static mode", duration_minutes);
                false
            }
        },
        _ => false,
    };

    if use_streaming {
        process_audio_streaming::<A>(input1, input2, output, mono_mode, mono_post, n_fft, hop, window_type, phase_source, use_float32)
    } else {
        process_audio_static::<A>(input1, input2, output, mono_mode, mono_post, n_fft, hop, window_type, stft_mode, phase_source, use_float32)
    }
}