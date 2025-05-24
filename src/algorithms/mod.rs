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
                match phase_source {
                    PhaseSource::MinMagnitude => {
                        if a.norm() <= b.norm() { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        let min_magnitude = a.norm().min(b.norm());
                        Complex::from_polar(min_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        let min_magnitude = a.norm().min(b.norm());
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
                match phase_source {
                    PhaseSource::MinMagnitude => {
                        if a.norm() >= b.norm() { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        let max_magnitude = a.norm().max(b.norm());
                        Complex::from_polar(max_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        let max_magnitude = a.norm().max(b.norm());
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

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],
        spec_b: &[Vec<Complex<f32>>],
        _phase_source: PhaseSource,
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