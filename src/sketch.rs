// The two-stage output head. The output matrix (vocabulary × n_embd) is the largest matrix of
// a small model, and reading it is what a generated token waits for most. Stage 1 scores every
// token with a 2-bit copy of the matrix (the "sketch"); stage 2 recomputes the best-scoring
// ~4000 tokens from the real weights. So the logits of every likely token are exactly those of
// the plain head, and only the far tail — which top-k / top-p sampling never reaches — carries
// the sketch's rounding noise. `fastgguf bench` compares the two heads on the model at hand;
// FASTGGUF_HEAD=exact turns the sketch off.
//
// The sketch: W[r][c] ≈ row[r] · col[c] · (2u − 3), u in 0..=3. The column scales go into x, so
// stage 1 is integer only: Σ u·xq per row, 8 rows side by side as in repack.rs.

use crate::gguf::Tensor;
use crate::matvec::{act_kind, dot_row, matvec, prefetch_row, QAct};
use crate::pool::{par_for, SendPtr};
use crate::quant::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub struct Sketch {
    cols: usize,
    /// per group of 8 rows and per 16 columns: 32 bytes [row][i]; byte i holds the codes of
    /// columns 4k+i in bits 2k (k = 0..4)
    qs: Vec<u8>,
    /// per row: half the step of its codes
    row: Vec<f32>,
    /// per column: the RMS of the column
    col: Vec<f32>,
}

/// the step between two codes of a unit-variance value (the best uniform 2-bit quantizer)
const STEP: f32 = 0.996;

impl Sketch {
    /// None when the matrix is small, already about 2 bits per weight, or of a type that
    /// cannot be decoded
    pub fn build(t: &Tensor) -> Option<Sketch> {
        let (cols, rows) = (t.cols(), t.rows());
        if t.n_dims != 2 || t.typ == T_Q2_K || cols % 16 != 0 || rows < 16384 {
            return None;
        }
        let stride = row_stride(t.typ, cols as u64) as usize;
        let data = t.data;
        if !dequantize_row(t.typ, &data[..stride], &mut vec![0.0f32; cols]) {
            return None;
        }
        let ng = rows / 8;
        // column scales: RMS over a sample of the rows
        let sample: Vec<usize> = (0..ng * 8).step_by(16).collect();
        let sums = sample
            .par_iter()
            .fold(
                || (vec![0.0f32; cols], vec![0.0f64; cols]),
                |(mut row, mut acc), &r| {
                    dequantize_row(t.typ, &data[r * stride..][..stride], &mut row);
                    acc.iter_mut().zip(&row).for_each(|(a, &v)| *a += v as f64 * v as f64);
                    (row, acc)
                },
            )
            .map(|(_, acc)| acc)
            .reduce(|| vec![0.0f64; cols], |mut a, b| {
                a.iter_mut().zip(&b).for_each(|(a, b)| *a += b);
                a
            });
        let col: Vec<f32> = sums.iter().map(|&s| ((s / sample.len() as f64).sqrt() as f32).max(1e-20)).collect();
        let icol: Vec<f32> = col.iter().map(|c| 1.0 / c).collect();
        let nq = cols / 16;
        let mut qs = vec![0u8; ng * nq * 32];
        let mut row = vec![0.0f32; ng * 8];
        qs.par_chunks_mut(nq * 32).zip(row.par_chunks_mut(8)).enumerate().for_each_init(
            || vec![0.0f32; cols],
            |z, (g, (dst, rs))| {
                for r in 0..8 {
                    dequantize_row(t.typ, &data[(g * 8 + r) * stride..][..stride], z);
                    let mut ss = 0.0f32;
                    for (v, c) in z.iter_mut().zip(&icol) {
                        *v *= c;
                        ss += *v * *v;
                    }
                    let thr = (ss / cols as f32).sqrt() * STEP;
                    // codes 0..=3 around -thr, 0, thr; then the scale that fits them best:
                    // Σ z·q / Σ q², q = 2u − 3
                    let (mut zq, mut qq) = (0.0f32, 0i32);
                    for (q, zc) in z.chunks_exact(16).enumerate() {
                        let mut code = [0u8; 16];
                        for (u, &v) in code.iter_mut().zip(zc) {
                            *u = (v > -thr) as u8 + (v > 0.0) as u8 + (v > thr) as u8;
                            let w = 2 * *u as i32 - 3;
                            zq += v * w as f32;
                            qq += w * w;
                        }
                        for i in 0..4 {
                            dst[q * 32 + r * 4 + i] = code[i] | code[4 + i] << 2 | code[8 + i] << 4 | code[12 + i] << 6;
                        }
                    }
                    rs[r] = zq / qq as f32;
                }
            },
        );
        Some(Sketch { cols, qs, row, col })
    }

    pub fn bytes(&self) -> usize {
        self.qs.len() + 4 * (self.row.len() + self.col.len())
    }

    /// out = W · x for the rows the sketch covers, exact for the likely tokens (see the top)
    pub fn matvec(&self, t: &Tensor, x: &[f32], out: &mut [f32]) -> Result<(), String> {
        let (cols, rows) = (self.cols, t.rows());
        let x = &x[..cols];
        let out = &mut out[..rows];
        let ng = self.row.len() / 8;
        let nq = cols / 16;

        // x · column scales as int8 with one scale for the whole vector
        let xc: Vec<f32> = x.iter().zip(&self.col).map(|(v, c)| v * c).collect();
        let amax = xc.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let xs = amax / 127.0;
        let id = if xs > 0.0 { 1.0 / xs } else { 0.0 };
        let xq: Vec<i8> = xc.iter().map(|&v| rnd(v * id) as i8).collect();
        let sx: i32 = xq.iter().map(|&v| v as i32).sum();

        let optr = SendPtr(out.as_mut_ptr());
        par_for(ng, (ng / (crate::pool::threads() * 8)).max(1), |g0, g1| {
            for g in g0..g1 {
                // SAFETY: group g is handed to one thread only
                let o = unsafe { std::slice::from_raw_parts_mut(optr.get().add(g * 8), 8) };
                let mut si = [0i32; 8];
                dot_sk2x8(&self.qs[g * nq * 32..(g + 1) * nq * 32], &xq, &mut si);
                for r in 0..8 {
                    o[r] = self.row[g * 8 + r] * xs * (2 * si[r] - 3 * sx) as f32;
                }
            }
        });

        // the score that about `want` rows reach, from a sample of every step-th row
        let want = (rows / 8).min(4096);
        let step = (ng * 8 / 4096).max(1);
        let mut sample: Vec<f32> = out[..ng * 8].iter().step_by(step).copied().collect();
        let k = (sample.len() * want / (ng * 8)).clamp(1, sample.len() - 1);
        sample.select_nth_unstable_by(k, |p, q| q.total_cmp(p));
        let thr = sample[k];

        // stage 2: every row at or above it, from the real weights
        let kind = act_kind(t.typ, cols);
        let a = QAct::new(kind, x);
        let stride = row_stride(t.typ, cols as u64) as usize;
        let (fail, n_exact) = (AtomicBool::new(false), AtomicUsize::new(0));
        let a = &a;
        par_for(ng * 8, 4096, |r0, r1| {
            let mut n = 0;
            for r in r0..r1 {
                // SAFETY: row r is handed to one thread only
                let o = unsafe { &mut *optr.get().add(r) };
                if *o >= thr {
                    let rb = &t.data[r * stride..][..stride];
                    prefetch_row(rb);
                    match dot_row(t.typ, kind, rb, a, x) {
                        Some(d) => *o = d,
                        None => fail.store(true, Ordering::Relaxed),
                    }
                    n += 1;
                }
            }
            n_exact.fetch_add(n, Ordering::Relaxed);
        });
        // rows past the last full group have no sketch
        for r in ng * 8..rows {
            match dot_row(t.typ, kind, &t.data[r * stride..][..stride], a, x) {
                Some(d) => out[r] = d,
                None => fail.store(true, Ordering::Relaxed),
            }
        }
        // not a usable ranking (NaN scores, or most of the vocabulary tied): the plain head
        let n = n_exact.load(Ordering::Relaxed);
        if fail.load(Ordering::Relaxed) || n < want / 8 || n > rows / 2 {
            return matvec(t, x, out);
        }
        Ok(())
    }
}

/// si[r] = Σ code(r, c) · xq[c] for the 8 rows of one group
pub fn dot_sk2x8(qs: &[u8], xq: &[i8], si: &mut [i32; 8]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        // SAFETY: the CPU has AVX2; qs holds 32 bytes per 16 values of xq
        return unsafe { crate::simd::simd_call!(dot_sk2x8(qs, xq, si)) };
    }
    *si = [0; 8];
    for (q, blk) in qs.chunks_exact(32).enumerate() {
        for r in 0..8 {
            for i in 0..4 {
                let b = blk[r * 4 + i];
                for k in 0..4 {
                    si[r] += ((b >> (2 * k)) & 3) as i32 * xq[q * 16 + 4 * k + i] as i32;
                }
            }
        }
    }
}
