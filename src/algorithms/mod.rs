// src/algorithms/mod.rs
use rustfft::num_complex::Complex;
use crate::utils::PhaseSource;

pub trait Algorithm {
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
                        // Original behavior: use phase from signal with minimum magnitude
                        if a.norm() <= b.norm() { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        // Use minimum magnitude, but always phase from first input
                        let min_magnitude = a.norm().min(b.norm());
                        Complex::from_polar(min_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        // Use minimum magnitude, but always phase from second input
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
                        // Use phase from signal with maximum magnitude
                        if a.norm() >= b.norm() { *a } else { *b }
                    }
                    PhaseSource::Input1 => {
                        // Use maximum magnitude, but always phase from first input
                        let max_magnitude = a.norm().max(b.norm());
                        Complex::from_polar(max_magnitude, a.arg())
                    }
                    PhaseSource::Input2 => {
                        // Use maximum magnitude, but always phase from second input
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
    fn name(&self) -> &'static str {
        "sub"
    }

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],  // X - main signal 
        spec_b: &[Vec<Complex<f32>>],  // y - signal to subtract
        _phase_source: PhaseSource,    // Ignore, use original logic
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(x, y)| {
                // Exact replica of Python code:
                // X_mag = np.abs(specs[0])
                // y_mag = np.abs(specs[1])            
                // max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)  
                // v_spec = specs[1] - max_mag * np.exp(1.j * np.angle(specs[0]))
                
                let x_mag = x.norm();        // |X|
                let y_mag = y.norm();        // |y|
                
                // max_mag = max(|X|, |y|)
                let max_mag = if x_mag >= y_mag { x_mag } else { y_mag };
                
                // max_mag * e^(i * angle(X)) = max_mag * e^(i * phase_of_X)
                let max_complex = Complex::from_polar(max_mag, x.arg());
                
                // v_spec = y - max_mag * e^(i * angle(X))
                let result = y - max_complex;
                
                // Fix result phase: multiply by -1 (180° rotation)
                -result
            }).collect()
        }).collect()
    }
}

pub struct CopyPhase;

impl Algorithm for CopyPhase {
    fn name(&self) -> &'static str {
        "copy-phase"
    }

    fn process(
        &self,
        spec_a: &[Vec<Complex<f32>>],  // input1 - phase source
        spec_b: &[Vec<Complex<f32>>],  // input2 - amplitude source
        _phase_source: PhaseSource,    // Ignore, always use phase from input1
    ) -> Vec<Vec<Complex<f32>>> {
        spec_a.iter().zip(spec_b.iter()).map(|(arow, brow)| {
            arow.iter().zip(brow.iter()).map(|(a, b)| {
                // Use amplitude (magnitude) from input2 (b) and phase from input1 (a)
                let magnitude_from_input2 = b.norm();
                let phase_from_input1 = a.arg();
                
                // Create new complex signal: magnitude(input2) * e^(i * phase(input1))
                Complex::from_polar(magnitude_from_input2, phase_from_input1)
            }).collect()
        }).collect()
    }
}