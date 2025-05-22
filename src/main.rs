use hound::{WavReader, SampleFormat};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{env, f32::consts::PI, fs::File, io::{BufWriter, Write}, process::Command, path::PathBuf};
use indicatif::ProgressBar;

fn decode_with_ffmpeg(input: &str) -> PathBuf {
    let tmp_path = PathBuf::from(format!("{}.tmp_converted.wav", input));
    let status = Command::new("ffmpeg")
        .args(["-y", "-i", input, "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", "-f", "wav"])
        .arg(&tmp_path)
        .status()
        .expect("Failed to run ffmpeg");

    if !status.success() {
        panic!("ffmpeg failed to convert input");
    }

    tmp_path
}

fn read_wav_stereo(path: &str) -> (Vec<Vec<f32>>, u32, Option<PathBuf>) {
    let (actual_path, tmp_file): (PathBuf, Option<PathBuf>) = {
		let tmp = decode_with_ffmpeg(path);
		(tmp.clone(), Some(tmp))
    };

    let mut reader = WavReader::open(&actual_path).expect("Cannot open WAV");
    let spec = reader.spec();
    let sr = spec.sample_rate;

    let samples: Vec<f32> = match (spec.bits_per_sample, spec.sample_format) {
        (8, _) => reader.samples::<i8>().map(|s| s.unwrap() as f32 / 128.0).collect(),
        (16, _) => reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect(),
        (24, _) => reader.samples::<i32>().map(|s| (s.unwrap() >> 8) as f32 / 8388608.0).collect(),
        (32, SampleFormat::Float) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        (32, SampleFormat::Int) => reader.samples::<i32>().map(|s| s.unwrap() as f32 / 2147483648.0).collect(),
        _ => panic!("Unsupported format: {} {:?}", spec.bits_per_sample, spec.sample_format),
    };

    let mut ch1 = Vec::new();
    let mut ch2 = Vec::new();
    let channels = spec.channels.min(2);
    for i in 0..(samples.len() / channels as usize) {
        ch1.push(samples[i * channels as usize]);
        if channels > 1 {
            ch2.push(samples[i * channels as usize + 1]);
        } else {
            ch2.push(ch1[i]);
        }
    }

    (vec![ch1, ch2], sr, tmp_file)
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / n as f32).cos()).collect()
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

    pb.finish();
    output
}

fn istft(spec: &[Vec<Complex<f32>>], win: &[f32], hop: usize, ifft: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let frame_size = win.len();
    let mut signal = vec![0.0; (spec.len() - 1) * hop + frame_size];
    let mut norm = vec![0.0; signal.len()];

    for (i, frame) in spec.iter().enumerate() {
        let mut frame = frame.clone();
        ifft.process(&mut frame);
        for j in 0..frame_size {
            let val = frame[j].re * win[j] / frame_size as f32;
            signal[i * hop + j] += val;
            norm[i * hop + j] += win[j] * win[j];
        }
    }

    for i in 0..signal.len() {
        if norm[i] > 1e-6 {
            signal[i] /= norm[i];
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

fn write_wav_24bit(path: &str, ch: &[Vec<f32>], sample_rate: u32) {
    let num_channels = 2;
    let bits_per_sample = 24;
    let byte_rate = sample_rate * num_channels * 3;
    let block_align = num_channels * 3;
    let num_samples = ch[0].len();

    let data_size = num_samples as u32 * block_align as u32;
    let mut writer = BufWriter::new(File::create(path).unwrap());

    writer.write_all(b"RIFF").unwrap();
    writer.write_all(&(36 + data_size).to_le_bytes()).unwrap();
    writer.write_all(b"WAVE").unwrap();

    writer.write_all(b"fmt ").unwrap();
    writer.write_all(&(16u32).to_le_bytes()).unwrap();
    writer.write_all(&(1u16).to_le_bytes()).unwrap();
    writer.write_all(&(num_channels as u16).to_le_bytes()).unwrap();
    writer.write_all(&sample_rate.to_le_bytes()).unwrap();
    writer.write_all(&byte_rate.to_le_bytes()).unwrap();
    writer.write_all(&(block_align as u16).to_le_bytes()).unwrap();
    writer.write_all(&(bits_per_sample as u16).to_le_bytes()).unwrap();

    writer.write_all(b"data").unwrap();
    writer.write_all(&data_size.to_le_bytes()).unwrap();

    for i in 0..num_samples {
        for c in 0..num_channels as usize {
            let sample = (ch[c][i] * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
            let bytes = sample.to_le_bytes();
            writer.write_all(&bytes[0..3]).unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} input1 input2 output.wav", args[0]);
        return;
    }

    let (mut a, sr1, tmp1) = read_wav_stereo(&args[1]);
    let (mut b, sr2, tmp2) = read_wav_stereo(&args[2]);

    assert_eq!(sr1, sr2, "Sample rates must match");

    for i in 0..2 {
        let len = a[i].len().min(b[i].len());
        a[i].truncate(len);
        b[i].truncate(len);
    }

    let n_fft = 2048;
    let hop = 512;
    let win = hann_window(n_fft);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    let mut output = Vec::new();

    for ch in 0..2 {
        println!("Processing channel {}...", ch + 1);
        let pb = ProgressBar::new(((a[ch].len() - n_fft) / hop + 1) as u64);
        let spec_a = stft(&a[ch], &win, hop, fft.as_ref(), &pb);
        let spec_b = stft(&b[ch], &win, hop, fft.as_ref(), &pb);
        let min_spec = min_mag_spec(&spec_a, &spec_b);
        let y = istft(&min_spec, &win, hop, ifft.as_ref());
        output.push(y);
    }

    write_wav_24bit(&args[3], &output, sr1);

    if let Some(p1) = tmp1 { let _ = std::fs::remove_file(p1); }
    if let Some(p2) = tmp2 { let _ = std::fs::remove_file(p2); }
}
