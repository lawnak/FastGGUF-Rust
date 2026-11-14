// Multi-threaded MatVec: out[r] = dot(W[r], x). x is quantized once (Q8 or Q8_K, matching
// the weight type) and every row is a call to one of the integer kernels.

use crate::gguf::Tensor;
use crate::quant::*;
use crate::pool::SendPtr;
use crate::repack::{self, Repacked};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

/// how x must be prepared for a weight type
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Act {
    F32, // no integer kernel: float path
    Q8,  // 32-value blocks
    Q8K, // 256-value blocks (K-quants)
}

pub fn act_kind(typ: u32, cols: usize) -> Act {
    match typ {
        T_Q4_K | T_Q5_K | T_Q6_K if cols % 256 == 0 => Act::Q8K,
        T_Q4_0 | T_Q8_0 | T_IQ4_NL | T_Q4_1 | T_Q5_0 | T_Q5_1 if cols % 32 == 0 => Act::Q8,
        T_Q2_K | T_Q3_K | T_IQ4_XS if cols % 256 == 0 => Act::Q8,
        _ => Act::F32,
    }
}

/// x quantized for one kind (the other buffer stays empty)
#[derive(Default)]
pub struct QAct {
    pub q8: Vec<Q8Blk>,
    pub q8k: Vec<Q8KBlk>,
}

impl QAct {
    pub fn new(kind: Act, x: &[f32]) -> QAct {
        let mut a = QAct::default();
        a.add(kind, x);
        a
    }
    /// also quantize for another kind (no-op when it is already there)
    pub fn add(&mut self, kind: Act, x: &[f32]) {
        match kind {
            Act::Q8 if self.q8.is_empty() => {
                self.q8 = vec![Q8Blk::default(); x.len() / 32];
                quantize_q8(x, &mut self.q8);
            }
            Act::Q8K if self.q8k.is_empty() => {
                self.q8k = vec![Q8KBlk::default(); x.len() / 256];
                quantize_q8k(x, &mut self.q8k);
            }
            _ => {}
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// dot(W row, x) — None for a type without a kernel or dequantizer
#[inline]
pub fn dot_row(typ: u32, kind: Act, rb: &[u8], a: &QAct, x: &[f32]) -> Option<f32> {
    let cols = x.len();
    let (nb32, nb256) = (cols / 32, cols / 256);
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        use crate::simd::simd_call;
        // SAFETY: the CPU has AVX2/FMA/F16C, and rb holds a whole row of this type
        unsafe {
            match (kind, typ) {
                (Act::Q8K, T_Q4_K) => return Some(simd_call!(dot_q4_k(rb, &a.q8k, nb256))),
                (Act::Q8K, T_Q5_K) => return Some(simd_call!(dot_q5_k(rb, &a.q8k, nb256))),
                (Act::Q8K, T_Q6_K) => return Some(simd_call!(dot_q6_k(rb, &a.q8k, nb256))),
                (Act::Q8, T_Q4_0) => return Some(simd_call!(dot_q4_0(rb, &a.q8, nb32))),
                (Act::Q8, T_Q8_0) => return Some(simd_call!(dot_q8_0(rb, &a.q8, nb32))),
                (Act::Q8, T_Q4_1) => return Some(simd_call!(dot_q4_1(rb, &a.q8, nb32))),
                (Act::Q8, T_Q5_0) => return Some(simd_call!(dot_q5_0(rb, &a.q8, nb32))),
                (Act::Q8, T_Q5_1) => return Some(simd_call!(dot_q5_1(rb, &a.q8, nb32))),
                (Act::Q8, T_Q2_K) => return Some(simd_call!(dot_q2_k(rb, &a.q8, nb256))),
                (Act::Q8, T_Q3_K) => return Some(simd_call!(dot_q3_k(rb, &a.q8, nb256))),
                (Act::Q8, T_IQ4_NL) => return Some(simd_call!(dot_iq4_nl(rb, &a.q8, nb32))),
                (Act::Q8, T_IQ4_XS) => return Some(simd_call!(dot_iq4_xs(rb, &a.q8, nb256))),
                (_, T_F32) => return Some(simd_call!(dot_f32_row(rb, x))),
                (_, T_F16) => return Some(simd_call!(dot_f16(rb, x))),
                _ => {}
            }
        }
    }
    Some(match (kind, typ) {
        (Act::Q8K, T_Q4_K) => dot_q4_k(rb, &a.q8k, nb256),
        (Act::Q8K, T_Q5_K) => dot_q5_k(rb, &a.q8k, nb256),
        (Act::Q8K, T_Q6_K) => dot_q6_k(rb, &a.q8k, nb256),
        (Act::Q8, T_Q4_0) => dot_q4_0(rb, &a.q8, nb32),
        (Act::Q8, T_Q8_0) => dot_q8_0(rb, &a.q8, nb32),
        (Act::Q8, T_Q4_1) => dot_q4_1(rb, &a.q8, nb32),
        (Act::Q8, T_Q5_0) => dot_q5_0(rb, &a.q8, nb32),
        (Act::Q8, T_Q5_1) => dot_q5_1(rb, &a.q8, nb32),
        (Act::Q8, T_Q2_K) => dot_q2_k(rb, &a.q8, nb256),
        (Act::Q8, T_Q3_K) => dot_q3_k(rb, &a.q8, nb256),
        (Act::Q8, T_IQ4_NL) => dot_iq4_nl(rb, &a.q8, nb32),
        (Act::Q8, T_IQ4_XS) => dot_iq4_xs(rb, &a.q8, nb256),
        (_, T_F32) => quant::dot_f32_row(rb, x),
        (_, T_F16) => quant::dot_f16(rb, x),
        (_, T_BF16) => quant::dot_bf16(rb, x),
        _ => {
            // no integer kernel (e.g. a row length that is not a block multiple): dequantize
            return SCRATCH.with(|s| {
                let mut s = s.borrow_mut();
                s.resize(cols, 0.0);
                dequantize_row(typ, rb, &mut s).then(|| dot_f32(&s, x))
            });
        }
    })
}

use crate::quant;

/// float dot product (attention, float weights)
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { crate::simd::simd_call!(dot_f32(a, b)) };
    }
    quant::dot_f32(a, b)
}

/// y += w · x
#[inline]
pub fn axpy(y: &mut [f32], w: f32, x: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        return unsafe { crate::simd::simd_call!(axpy(y, w, x)) };
    }
    for (o, v) in y.iter_mut().zip(x) {
        *o += w * v;
    }
}

#[inline(always)]
pub fn prefetch_row(p: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if crate::simd::avx2() {
        // ask for the bytes 2 KB ahead: with compute between the loads the hardware
        // prefetcher alone leaves one core well below the memory bandwidth
        let base = p.as_ptr().wrapping_add(2048);
        let mut c = 0;
        while c < p.len() {
            // SAFETY: prefetch never faults, even past the end of the mapping
            unsafe { crate::simd::avx2::prefetch(base.wrapping_add(c)) };
            c += 64;
        }
    }
}

/// o = rows 8g .. 8g+8 of a repacked matrix times the quantized x (bsp: q4kx8_bsums for Q4_K)
#[inline]
fn group_rp(rp: &Repacked, g: usize, cols: usize, a: &QAct, bsp: &[i32], o: &mut [f32]) {
    match rp {
        Repacked::Q4K(v) => {
            let nb = cols / 256;
            repack::dot_q4kx8(&v[g * nb..(g + 1) * nb], &a.q8k, bsp, o);
        }
        Repacked::Q40(v) => {
            let nb = cols / 32;
            repack::dot_q40x8(&v[g * nb..(g + 1) * nb], &a.q8, o);
        }
        Repacked::Q50(v) => {
            let nb = cols / 32;
            repack::dot_q50x8(&v[g * nb..(g + 1) * nb], &a.q8, o);
        }
    }
}

/// out = W · x  (rows of W in parallel)
pub fn matvec(t: &Tensor, x: &[f32], out: &mut [f32]) -> Result<(), String> {
    matvec_multi(x, &mut [(t, out)])
}

/// out_i = W_i · x for several matrices with the same input (Q/K/V, FFN gate/up): x is
/// quantized once and all the rows go through one parallel loop
pub fn matvec_multi(x: &[f32], jobs: &mut [(&Tensor, &mut [f32])]) -> Result<(), String> {
    struct Seg<'a> {
        t: &'a Tensor,
        out: SendPtr<f32>,
        rows: usize,
        stride: usize,
        kind: Act,
        first: usize, // first unit (8 rows) of this matrix in the shared loop
    }
    let cols = jobs.first().map(|j| j.0.cols()).unwrap_or(0);
    let x = &x[..cols];
    let mut a = QAct::default();
    let mut bsp = Vec::new();
    let mut segs: Vec<Seg> = Vec::with_capacity(jobs.len());
    let mut units = 0;
    for (t, out) in jobs.iter_mut() {
        let t: &Tensor = t;
        let rows = t.rows();
        if t.cols() != cols {
            return Err(format!("matvec: {} does not take {cols} inputs", t.name));
        }
        let out = &mut out[..rows];
        // weights in VRAM are multiplied there — any failure turns the GPU off and continues on the CPU
        if let Some(m) = &t.gpu {
            if crate::gpu::ready() {
                match crate::gpu::matvec(m, t.typ, x, out) {
                    Ok(()) => continue,
                    Err(e) => {
                        eprintln!("[gpu] matvec failed on {}: {e} - falling back to CPU", t.name);
                        crate::gpu::disable();
                    }
                }
            }
        }
        let kind = match &t.rp {
            Some(Repacked::Q4K(_)) => Act::Q8K,
            Some(_) => Act::Q8,
            None => act_kind(t.typ, cols),
        };
        a.add(kind, x);
        if matches!(t.rp, Some(Repacked::Q4K(_))) && bsp.is_empty() {
            bsp = repack::q4kx8_bsums(&a.q8k, cols / 256);
        }
        segs.push(Seg { t, out: SendPtr(out.as_mut_ptr()), rows, stride: row_stride(t.typ, cols as u64) as usize, kind, first: units });
        units += rows.div_ceil(8);
    }
    let fail = AtomicBool::new(false);
    // pieces of at least ~16k weights, and about 4 per thread so a slower core does not hold the rest up
    let chunk = (units / (crate::pool::threads() * 4)).max(16384 / (8 * cols.max(1))).max(1);
    let (a, bsp) = (&a, &bsp);
    crate::pool::par_for(units, chunk, |u0, u1| {
        for s in &segs {
            let (g0, g1) = (u0.max(s.first) - s.first, u1.min(s.first + s.rows.div_ceil(8)).saturating_sub(s.first));
            for g in g0..g1 {
                let n = (s.rows - g * 8).min(8);
                // SAFETY: every unit is handed to one thread only, and it covers rows g*8 .. g*8+n
                let o = unsafe { std::slice::from_raw_parts_mut(s.out.get().add(g * 8), n) };
                match &s.t.rp {
                    Some(rp) => group_rp(rp, g, cols, a, bsp, o),
                    None => {
                        for (i, v) in o.iter_mut().enumerate() {
                            let r = g * 8 + i;
                            let rb = &s.t.data[r * s.stride..(r + 1) * s.stride];
                            prefetch_row(rb);
                            match dot_row(s.t.typ, s.kind, rb, a, x) {
                                Some(d) => *v = d,
                                None => fail.store(true, Ordering::Relaxed),
                            }
                        }
                    }
                }
            }
        }
    });
    if fail.load(Ordering::Relaxed) {
        let names: Vec<String> = segs.iter().map(|s| format!("{} ({})", s.t.name, type_name(s.t.typ))).collect();
        return Err(format!("matvec: unsupported type in {}", names.join(", ")));
    }
    Ok(())
}
