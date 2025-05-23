use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::default::{get_codecs, get_probe};
use std::fs::File;
use std::path::PathBuf;
use std::env;
use std::f32::consts::PI;
use std::io::{BufWriter, Write};
use std::time::Instant;
use indicatif::ProgressBar;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::HashMap;

fn print_help() {
    println!("min-mag v0.2.0");
    println!("https://github.com/aufr33/min-mag");
    println!(" ");
    println!("Usage:");
    println!("  min_mag input1.wav input2.wav output.wav [--mono] [--n_fft=N] [--hop=H] [--window=TYPE]\n");
    println!("Options:");
    println!("  --mono           Export mono output (average of both channels)");
    println!("  --mono-post      Process stereo, then mix to mono (matches manual mixing)");
    println!("  --n_fft=N        FFT window size (default: 2048)");
    println!("  --hop=H          Hop length (default: 512)");
    println!("  --window=TYPE    Window function: hann or hamming (default: hann)");
    println!("  --help           Show this message");
    println!("  --version        Show version");
}

fn select_window(name: &str, size: usize) -> Vec<f32> {
    match name.to_lowercase().as_str() {
        "hamming" => (0..size).map(|i| 0.54 - 0.46 * ((2.0 * PI * i as f32) / (size - 1) as f32).cos()).collect(),
        _ => (0..size).map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / size as f32).cos()).collect(),
    }
}

fn read_audio_file(path: &str) -> Result<(Vec<Vec<f32>>, u32, Option<PathBuf>), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let probed = get_probe().format(&Default::default(), mss, &FormatOptions::default(), &MetadataOptions::default())?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("no default track")?;
    let dec_opts = DecoderOptions { verify: true, ..Default::default() };
    let mut decoder = get_codecs().make(&track.codec_params, &dec_opts)?;

    let mut channels_data: Vec<Vec<f32>> = Vec::new();
    let mut sample_rate = 0;

    while let Ok(packet) = format.next_packet() {
        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F32(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend_from_slice(chan);
                }
            }
            AudioBufferRef::S8(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| s as f32 / 128.0));
                }
            }
            AudioBufferRef::U8(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            AudioBufferRef::S16(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| s as f32 / 32768.0));
                }
            }
            AudioBufferRef::U16(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                }
            }
            AudioBufferRef::S24(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| s.inner() as f32 / 8388608.0));
                }
            }
            AudioBufferRef::U24(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| (s.inner() as f32 - 8388608.0) / 8388608.0));
                }
            }
            AudioBufferRef::S32(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| s as f32 / 2147483648.0));
                }
            }
            AudioBufferRef::U32(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| (s as f32 - 2147483648.0) / 2147483648.0));
                }
            }
            AudioBufferRef::F64(buf) => {
                if sample_rate == 0 {
                    sample_rate = buf.spec().rate;
                    channels_data = vec![Vec::new(); buf.spec().channels.count()];
                }
                for ch_idx in 0..buf.spec().channels.count() {
                    let chan = buf.chan(ch_idx);
                    channels_data[ch_idx].extend(chan.iter().map(|&s| s as f32));
                }
            }
        }
    }

    if channels_data.len() == 1 {
        channels_data.push(channels_data[0].clone());
    } else if channels_data.len() > 2 {
        channels_data.truncate(2);
    }

    Ok((channels_data, sample_rate, None))
}

fn resample_if_needed(channels: Vec<Vec<f32>>, from_rate: u32, to_rate: u32) -> Vec<Vec<f32>> {
    if from_rate == to_rate {
        return channels;
    }
    
    println!("Resampling from {}Hz to {}Hz...", from_rate, to_rate);
    
    let ratio = to_rate as f32 / from_rate as f32;
    let new_len = (channels[0].len() as f32 * ratio) as usize;
    
    let mut resampled = vec![Vec::with_capacity(new_len); channels.len()];
    
    for ch in 0..channels.len() {
        for i in 0..new_len {
            let src_pos = i as f32 / ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = src_pos - src_idx as f32;
            
            if src_idx + 1 < channels[ch].len() {
                let sample = channels[ch][src_idx] * (1.0 - frac) + channels[ch][src_idx + 1] * frac;
                resampled[ch].push(sample);
            } else if src_idx < channels[ch].len() {
                resampled[ch].push(channels[ch][src_idx]);
            }
        }
    }
    
    resampled
}

fn main() {
    let start_time = Instant::now();
    let args: Vec<String> = env::args().collect();

    if args.iter().any(|a| a == "--help") {
        print_help();
        return;
    }
    if args.iter().any(|a| a == "--version") {
        println!("min-mag v0.2.0");
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

    if positional.len() != 3 {
        print_help();
        return;
    }

    let input1 = &positional[0];
    let input2 = &positional[1];
    let output = &positional[2];
    let mono_mode = opts.get("mono").map(|v| v == "true").unwrap_or(false);
    let mono_post = opts.get("mono-post").map(|v| v == "true").unwrap_or(false);
    let n_fft = opts.get("n_fft").and_then(|v| v.parse::<usize>().ok()).unwrap_or(2048);
    let hop = opts.get("hop").and_then(|v| v.parse::<usize>().ok()).unwrap_or(512);
    let window_type = opts.get("window").map(|s| s.as_str()).unwrap_or("hann");

    if mono_mode && mono_post {
        eprintln!("Error: Cannot use both --mono and --mono-post at the same time");
        return;
    }

    let win = select_window(window_type, n_fft);

    if mono_mode {
        println!("Mode: Pre-processing mono (mix first, then process)");
    } else if mono_post {
        println!("Mode: Post-processing mono (process stereo, then mix)");
    } else {
        println!("Mode: Stereo processing");
    }

    println!("Reading audio files...");
    let read_start = Instant::now();

    let (mut a, sr1, _) = read_audio_file(input1).expect("failed to read first file");
    let (mut b, sr2, _) = read_audio_file(input2).expect("failed to read second file");

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

        let spec_a = stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        let min_spec = min_mag_spec(&spec_a, &spec_b);
        output_channels[ch] = istft(&min_spec, &win, hop, ifft.as_ref());

        pb.finish();
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
    
    let output_data = write_wav_24bit(output, &final_output_channels, target_sr);

    if let Err(e) = output_data {
        eprintln!("Error writing output file: {}", e);
        return;
    }

    let total_duration = start_time.elapsed();
    println!("Done! Output written to {}", output);
    println!("Total execution time: {:.2}s", total_duration.as_secs_f32());

    let audio_length = a[0].len() as f32 / target_sr as f32;
    let realtime_factor = audio_length / total_duration.as_secs_f32();
    println!("Audio length: {:.2}s, Realtime factor: {:.2}x", audio_length, realtime_factor);
}

fn stft(signal: &[f32], win: &[f32], hop: usize, fft: &dyn rustfft::Fft<f32>, pb: &ProgressBar) -> Vec<Vec<Complex<f32>>> {
    let frame_size = win.len();
    let mut output = Vec::new();

    for start in (0..=signal.len().saturating_sub(frame_size)).step_by(hop) {
        let mut frame: Vec<Complex<f32>> = signal[start..start + frame_size]
            .iter().zip(win.iter())
            .map(|(&x, &w)| Complex { re: x * w, im: 0.0 })
            .collect();
        fft.process(&mut frame);
        output.push(frame);
        pb.inc(1);
    }

    output
}

fn istft(spec: &[Vec<Complex<f32>>], win: &[f32], hop: usize, ifft: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let frame_size = win.len();
    let signal_len = (spec.len() - 1) * hop + frame_size;
    let mut signal = vec![0.0; signal_len];
    let mut window_sum = vec![0.0; signal_len];
    
    let win_norm: Vec<f32> = win.iter().map(|&w| w * w).collect();

    for (i, frame) in spec.iter().enumerate() {
        let mut frame = frame.clone();
        ifft.process(&mut frame);
        let start_idx = i * hop;
        
        for (j, &window_val) in win.iter().enumerate() {
            if start_idx + j < signal_len {
                signal[start_idx + j] += frame[j].re * window_val / frame_size as f32;
                window_sum[start_idx + j] += win_norm[j];
            }
        }
    }

    for (sample, &norm) in signal.iter_mut().zip(window_sum.iter()) {
        if norm > f32::EPSILON {
            *sample /= norm;
        }
    }
    
    signal
}

fn min_mag_spec(a: &[Vec<Complex<f32>>], b: &[Vec<Complex<f32>>]) -> Vec<Vec<Complex<f32>>> {
    a.iter().zip(b.iter()).map(|(arow, brow)| {
        arow.iter().zip(brow.iter()).map(|(a, b)| {
            if a.norm() <= b.norm() { *a } else { *b }
        }).collect()
    }).collect()
}

fn write_wav_24bit(path: &str, ch: &[Vec<f32>], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let num_channels = ch.len() as u32;
    let bits_per_sample = 24;
    let byte_rate = sample_rate * num_channels * 3;
    let block_align = num_channels * 3;
    let num_samples = ch[0].len();

    let data_size = num_samples as u32 * block_align as u32;
    let mut writer = BufWriter::new(File::create(path)?);

    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_size).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    writer.write_all(b"fmt ")?;
    writer.write_all(&(16u32).to_le_bytes())?;
    writer.write_all(&(1u16).to_le_bytes())?; // PCM
    writer.write_all(&(num_channels as u16).to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&(block_align as u16).to_le_bytes())?;
    writer.write_all(&(bits_per_sample as u16).to_le_bytes())?;

    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    for i in 0..num_samples {
        for c in 0..num_channels as usize {
            let sample = (ch[c][i] * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
            let bytes = sample.to_le_bytes();
            writer.write_all(&bytes[0..3])?;
        }
    }

    Ok(())
}
