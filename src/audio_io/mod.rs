// src/audio_io/mod.rs
use hound::{WavReader, SampleFormat};
use std::fs::File;
use std::io::{BufWriter, Write, Seek, SeekFrom};
use std::path::PathBuf;
use std::process::Command;
use std::{thread, time::Duration};

fn decode_with_ffmpeg(input: &str) -> PathBuf {
    let tmp_path = PathBuf::from(format!("{}.tmp_converted.wav", input));
    let status = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-i", input, "-ac", "2", "-ar", "44100", "-c:a", "pcm_f32le", "-f", "wav"])
        .arg(&tmp_path)
        .status()
        .expect("Failed to run ffmpeg");

    if !status.success() {
        panic!("ffmpeg failed to convert input");
    }

    tmp_path
}

pub fn read_audio_file(path: &str) -> Result<(Vec<Vec<f32>>, u32, Option<PathBuf>), Box<dyn std::error::Error>> {
    let tmp_file = decode_with_ffmpeg(path);

    let mut reader = WavReader::open(&tmp_file)?;
    let spec = reader.spec();
    let sr = spec.sample_rate;

    let samples: Vec<f32> = match (spec.bits_per_sample, spec.sample_format) {
        (8, _) => reader.samples::<i8>().map(|s| s.unwrap() as f32 / 128.0).collect(),
        (16, _) => reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect(),
        (24, _) => reader.samples::<i32>().map(|s| (s.unwrap() >> 8) as f32 / 8388608.0).collect(),
        (32, SampleFormat::Float) => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        (32, SampleFormat::Int) => reader.samples::<i32>().map(|s| s.unwrap() as f32 / 2147483648.0).collect(),
        _ => return Err(format!("Unsupported format: {} {:?}", spec.bits_per_sample, spec.sample_format).into()),
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

    Ok((vec![ch1, ch2], sr, Some(tmp_file)))
}

pub struct StreamingWavWriter {
    writer: BufWriter<File>,
    samples_written: u32,
    num_channels: u16,
    sample_rate: u32,
}

impl StreamingWavWriter {
    fn new(path: &str, sample_rate: u32, num_channels: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let mut writer = BufWriter::new(File::create(path)?);
        
        writer.write_all(b"RIFF")?;
        writer.write_all(&(36u32 + 0).to_le_bytes())?;
        writer.write_all(b"WAVE")?;

        writer.write_all(b"fmt ")?;
        writer.write_all(&(16u32).to_le_bytes())?;
        writer.write_all(&(1u16).to_le_bytes())?;
        writer.write_all(&num_channels.to_le_bytes())?;
        writer.write_all(&sample_rate.to_le_bytes())?;
        writer.write_all(&(sample_rate * num_channels as u32 * 3).to_le_bytes())?;
        writer.write_all(&(num_channels * 3).to_le_bytes())?;
        writer.write_all(&(24u16).to_le_bytes())?;

        writer.write_all(b"data")?;
        writer.write_all(&(0u32).to_le_bytes())?;

        Ok(StreamingWavWriter {
            writer,
            samples_written: 0,
            num_channels,
            sample_rate,
        })
    }

    pub fn write_chunk(&mut self, chunk: &[Vec<f32>]) -> Result<(), Box<dyn std::error::Error>> {
        let num_samples = chunk[0].len();
        
        for i in 0..num_samples {
            for c in 0..self.num_channels as usize {
                let sample = (chunk[c][i] * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
                let bytes = sample.to_le_bytes();
                self.writer.write_all(&bytes[0..3])?;
            }
        }
        
        self.samples_written += num_samples as u32;
        Ok(())
    }

    pub fn finalize(mut self) -> Result<(), Box<dyn std::error::Error>> {
        let data_size = self.samples_written * self.num_channels as u32 * 3;
        let file_size = 36 + data_size;
        
        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_all(&file_size.to_le_bytes())?;
        
        self.writer.seek(SeekFrom::Start(40))?;
        self.writer.write_all(&data_size.to_le_bytes())?;
        
        self.writer.flush()?;
        Ok(())
    }
}

pub fn write_wav_24bit_streaming(path: &str, sample_rate: u32, num_channels: u16) -> Result<StreamingWavWriter, Box<dyn std::error::Error>> {
    StreamingWavWriter::new(path, sample_rate, num_channels)
}

pub fn write_wav_24bit(path: &str, ch: &[Vec<f32>], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
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
    writer.write_all(&(1u16).to_le_bytes())?;
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

pub fn cleanup_temp_files(files: Vec<Option<PathBuf>>) {
    for file_opt in files {
        if let Some(file_path) = file_opt {
            if !file_path.to_string_lossy().ends_with(".tmp_converted.wav") {
                continue;
            }
            
            for attempt in 0..3 {
                match std::fs::remove_file(&file_path) {
                    Ok(_) => break,
                    Err(e) => {
                        if attempt == 2 {
                            eprintln!("Warning: Failed to remove temp file {}: {}", file_path.display(), e);
                        } else {
                            thread::sleep(Duration::from_millis(100));
                        }
                    }
                }
            }
        }
    }
}