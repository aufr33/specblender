// src/utils/mod.rs
use std::f32::consts::PI;
use rustfft::num_complex::Complex;
use indicatif::ProgressBar;

#[derive(Debug, Clone, Copy)]
pub enum PhaseSource {
    MinMagnitude,  // --phase=auto
    Input1,        // --phase=input1
    Input2,        // --phase=input2
}

#[derive(Debug, Clone, Copy)]
pub enum StftMode {
    Single,
    Multi,
}

pub fn parse_phase_source(phase_str: &str) -> PhaseSource {
    match phase_str.to_lowercase().as_str() {
        "input1" | "1" => PhaseSource::Input1,
        "input2" | "2" => PhaseSource::Input2,
        "auto" | "min" | _ => PhaseSource::MinMagnitude,  // default to auto
    }
}

pub fn parse_stft_mode(mode_str: &str) -> StftMode {
    match mode_str.to_lowercase().as_str() {
        "multi" | "multi-stft" => StftMode::Multi,
        _ => StftMode::Single,
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.5 - 0.5 * ((2.0 * PI * i as f32) / n as f32).cos()).collect()
}

fn hamming_window(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.54 - 0.46 * ((2.0 * PI * i as f32) / (n - 1) as f32).cos()).collect()
}

pub fn select_window(name: &str, size: usize) -> Vec<f32> {
    match name.to_lowercase().as_str() {
        "hamming" => hamming_window(size),
        _ => hann_window(size),
    }
}

pub fn stft(signal: &[f32], win: &[f32], hop: usize, fft: &dyn rustfft::Fft<f32>, pb: &ProgressBar) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
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
    Ok(output)
}

pub fn istft(spec: &[Vec<Complex<f32>>], win: &[f32], hop: usize, ifft: &dyn rustfft::Fft<f32>) -> Vec<f32> {
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