// src/audio_io/mod.rs
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::default::{get_codecs, get_probe};
use std::fs::File;
use std::path::PathBuf;
use std::io::{BufWriter, Write};

pub fn read_audio_file(path: &str) -> Result<(Vec<Vec<f32>>, u32, Option<PathBuf>), Box<dyn std::error::Error>> {
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

    // Convert to stereo if mono
    if channels_data.len() == 1 {
        channels_data.push(channels_data[0].clone());
    } else if channels_data.len() > 2 {
        channels_data.truncate(2);
    }

    Ok((channels_data, sample_rate, None))
}

pub fn resample_if_needed(channels: Vec<Vec<f32>>, from_rate: u32, to_rate: u32) -> Vec<Vec<f32>> {
    if from_rate == to_rate {
        return channels;
    }
    
    println!("Resampling from {}Hz to {}Hz...", from_rate, to_rate);
    
    // Linear resampling
    let ratio = to_rate as f32 / from_rate as f32;
    let new_len = (channels[0].len() as f32 * ratio) as usize;
    
    let mut resampled = vec![Vec::with_capacity(new_len); channels.len()];
    
    for ch in 0..channels.len() {
        for i in 0..new_len {
            let src_pos = i as f32 / ratio;
            let src_idx = src_pos.floor() as usize;
            let frac = src_pos - src_idx as f32;
            
            if src_idx + 1 < channels[ch].len() {
                // Linear interpolation
                let sample = channels[ch][src_idx] * (1.0 - frac) + channels[ch][src_idx + 1] * frac;
                resampled[ch].push(sample);
            } else if src_idx < channels[ch].len() {
                resampled[ch].push(channels[ch][src_idx]);
            }
        }
    }
    
    resampled
}

pub fn write_wav_24bit(path: &str, ch: &[Vec<f32>], sample_rate: u32) -> Result<(), Box<dyn std::error::Error>> {
    let num_channels = ch.len() as u32;
    let bits_per_sample = 24;
    let byte_rate = sample_rate * num_channels * 3;
    let block_align = num_channels * 3;
    let num_samples = ch[0].len();

    let data_size = num_samples as u32 * block_align as u32;
    let mut writer = BufWriter::new(File::create(path)?);

    // WAV header
    writer.write_all(b"RIFF")?;
    writer.write_all(&(36 + data_size).to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&(16u32).to_le_bytes())?;
    writer.write_all(&(1u16).to_le_bytes())?; // PCM
    writer.write_all(&(num_channels as u16).to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&(block_align as u16).to_le_bytes())?;
    writer.write_all(&(bits_per_sample as u16).to_le_bytes())?;

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // Writing audio data
    for i in 0..num_samples {
        for c in 0..num_channels as usize {
            let sample = (ch[c][i] * 8388607.0).clamp(-8388608.0, 8388607.0) as i32;
            let bytes = sample.to_le_bytes();
            writer.write_all(&bytes[0..3])?;
        }
    }

    Ok(())
}