// src/algorithms/mod.rs
use rustfft::num_complex::Complex;
use crate::utils::PhaseSource;

pub trait Algorithm {
    fn new() -> Self where Self: Sized;
    fn name(&self) -> &'static str;
    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        phase_source: PhaseSource,
    ) -> Vec<Vec<Complex<f32>>>;
}

pub struct MinMag;

impl Algorithm for MinMag {
    fn new() -> Self { MinMag }
    
    fn name(&self) -> &'static str {
        "min-mag"
    }

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        phase_source: PhaseSource,
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(a, b)| {
                let mag_a = a.norm();
                let mag_b = b.norm();
                let min_magnitude = mag_a.min(mag_b);
                
                match phase_source {
                    PhaseSource::MinMagnitude => {
                        if mag_a <= mag_b { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        Complex::from_polar(min_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        Complex::from_polar(min_magnitude, b.arg())
                    }
                }
            }).collect()
        }).collect()
    }
}

pub struct MaxMag;

impl Algorithm for MaxMag {
    fn new() -> Self { MaxMag }
    
    fn name(&self) -> &'static str {
        "max-mag"
    }

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        phase_source: PhaseSource,
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(a, b)| {
                let mag_a = a.norm();
                let mag_b = b.norm();
                let max_magnitude = mag_a.max(mag_b);
                
                match phase_source {
                    PhaseSource::MinMagnitude => {
                        if mag_a >= mag_b { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        Complex::from_polar(max_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        Complex::from_polar(max_magnitude, b.arg())
                    }
                }
            }).collect()
        }).collect()
    }
}

pub struct Sub;

impl Algorithm for Sub {
    fn new() -> Self { Sub }
    
    fn name(&self) -> &'static str {
        "sub"
    }

    #[allow(unused_variables)]
    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        phase_source: PhaseSource, 
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(x, y)| {
                let x_mag = x.norm();
                let y_mag = y.norm();
                let max_mag = if x_mag >= y_mag { x_mag } else { y_mag };
                
                let max_complex = Complex::from_polar(max_mag, x.arg());
                let result = y - max_complex;
                -result
            }).collect()
        }).collect()
    }
}

pub struct CopyPhase;

impl Algorithm for CopyPhase {
    fn new() -> Self { CopyPhase }
    
    fn name(&self) -> &'static str {
        "copy-phase"
    }

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        _phase_source: PhaseSource, 
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(a, b)| {
                let magnitude_from_input2 = b.norm();
                let phase_from_input1 = a.arg();
                Complex::from_polar(magnitude_from_input2, phase_from_input1)
            }).collect()
        }).collect()
    }
}

pub fn validate_phase_compatibility(algorithm_name: &str, phase_source: PhaseSource) -> Result<(), String> {
    match algorithm_name {
        "min-mag" | "max-mag" => {
            Ok(())
        }
        "sub" => {
            match phase_source {
                PhaseSource::MinMagnitude => Ok(()), 
                PhaseSource::Input1 | PhaseSource::Input2 => {
                    Err(format!(
                        "Algorithm '{}' does not support --phase=input1/input2. \
                        Phase is always taken from input1 due to spectral subtraction mathematics. \
                        Use --phase=auto or omit the --phase argument.",
                        algorithm_name
                    ))
                }
            }
        }
        "copy-phase" => {
            match phase_source {
                PhaseSource::MinMagnitude => Ok(()), 
                PhaseSource::Input1 | PhaseSource::Input2 => {
                    Err(format!(
                        "Algorithm '{}' does not support --phase=input1/input2. \
                        This algorithm always copies magnitude from input2 and phase from input1. \
                        Use --phase=auto or omit the --phase argument.",
                        algorithm_name
                    ))
                }
            }
        }
        _ => {
            Ok(())
        }
    }
}

pub fn print_phase_usage_info(algorithm_name: &str, phase_source: PhaseSource) {
    match algorithm_name {
        "min-mag" | "max-mag" => {
            match phase_source {
                PhaseSource::MinMagnitude => {
                    println!("Phase handling: Automatic (from signal with {} magnitude)", 
                             if algorithm_name == "min-mag" { "minimum" } else { "maximum" });
                }
                PhaseSource::Input1 => {
                    println!("Phase handling: Forced from input1 (magnitude from {})", 
                             if algorithm_name == "min-mag" { "minimum" } else { "maximum" });
                }
                PhaseSource::Input2 => {
                    println!("Phase handling: Forced from input2 (magnitude from {})", 
                             if algorithm_name == "min-mag" { "minimum" } else { "maximum" });
                }
            }
        }
        "sub" => {
            println!("Phase handling: Fixed (always from input1 due to spectral subtraction math)");
        }
        "copy-phase" => {
            println!("Phase handling: Fixed (magnitude from input2, phase from input1)");
        }
        _ => {}
    }
}
