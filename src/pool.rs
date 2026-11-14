// A small fork-join pool for single-token generation. One token is a few hundred short
// parallel loops (one per matrix), each tens of microseconds long, so what matters is how fast
// the other threads pick a loop up: here they spin between loops for a moment instead of going
// to sleep, and take work in large chunks from one shared counter. (rayon splits a loop into
// many tiny tasks and lets idle threads sleep; that costs more than these loops take.)
// The prompt path keeps using rayon. Thread count: the same as rayon's (RAYON_NUM_THREADS).

use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::*};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Body<'a> = dyn Fn(usize, usize) + Sync + 'a;

struct Job {
    f: *const Body<'static>,
    n: usize,
    chunk: usize,
}

/// state = epoch << 32 | CLOSED | number of workers inside the job.
/// A worker may enter (count + 1, with a compare-exchange) only while the job is open, and the
/// caller closes it only when the count is 0, so the job is never read after `par_for` returns.
const CLOSED: u64 = 1 << 31;
/// how long an idle worker spins before it sleeps
const SPIN: Duration = Duration::from_micros(400);

struct Pool {
    state: AtomicU64,
    job: UnsafeCell<Job>,
    next: AtomicUsize,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
    run: Mutex<()>, // one parallel loop at a time
    workers: usize,
}

// SAFETY: `job` is written only by the thread holding `run` while the state is CLOSED with no
// worker inside, and read only by threads that entered the open job
unsafe impl Sync for Pool {}

impl Pool {
    fn work(&self, job: &Job) {
        // SAFETY: the closure outlives the open job (see CLOSED)
        let f = unsafe { &*job.f };
        loop {
            let i = self.next.fetch_add(job.chunk, Relaxed);
            if i >= job.n {
                break;
            }
            f(i, (i + job.chunk).min(job.n));
        }
    }

    fn worker(&'static self) {
        let mut seen = 0u64; // epoch of the last job this worker took part in
        let mut idle = 0u32;
        let mut since = Instant::now();
        loop {
            let s = self.state.load(SeqCst);
            if s & CLOSED == 0 && s >> 32 != seen {
                if self.state.compare_exchange_weak(s, s + 1, SeqCst, Relaxed).is_ok() {
                    // SAFETY: entered while open: the caller cannot close or replace the job
                    self.work(unsafe { &*self.job.get() });
                    seen = s >> 32;
                    self.state.fetch_sub(1, SeqCst);
                    idle = 0;
                }
                continue;
            }
            if idle == 0 {
                since = Instant::now();
            }
            idle += 1;
            std::hint::spin_loop();
            if idle % 256 == 0 && since.elapsed() > SPIN {
                self.sleepers.fetch_add(1, SeqCst);
                let mut g = self.lock.lock().unwrap();
                loop {
                    let s = self.state.load(SeqCst);
                    if s & CLOSED == 0 && s >> 32 != seen {
                        break;
                    }
                    g = self.cv.wait(g).unwrap();
                }
                drop(g);
                self.sleepers.fetch_sub(1, SeqCst);
                idle = 0;
            }
        }
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<&'static Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let workers = rayon::current_num_threads().max(1) - 1;
        let p: &'static Pool = Box::leak(Box::new(Pool {
            state: AtomicU64::new(CLOSED),
            job: UnsafeCell::new(Job { f: &(|_: usize, _: usize| {}) as *const Body<'static>, n: 0, chunk: 1 }),
            next: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
            run: Mutex::new(()),
            workers,
        }));
        for i in 0..workers {
            let _ = std::thread::Builder::new().name(format!("fastgguf-{i}")).spawn(move || p.worker());
        }
        p
    })
}

/// threads a loop is shared between
pub fn threads() -> usize {
    pool().workers + 1
}

/// f(start, end) over 0..n in pieces of `chunk`, on all threads; returns when all are done.
/// When another loop is already running (two models at once, or a nested call) rayon runs it.
pub fn par_for<F: Fn(usize, usize) + Sync>(n: usize, chunk: usize, f: F) {
    let chunk = chunk.max(1);
    if n <= chunk {
        if n > 0 {
            f(0, n);
        }
        return;
    }
    let p = pool();
    if p.workers == 0 {
        return f(0, n);
    }
    let guard = p.run.try_lock().ok();
    if guard.is_none() {
        let pieces = n.div_ceil(chunk);
        (0..pieces).into_par_iter().for_each(|i| f(i * chunk, ((i + 1) * chunk).min(n)));
        return;
    }
    let body: &Body = &f;
    // SAFETY: only the lifetime changes; the job is closed before this function returns
    let body: *const Body<'static> = unsafe { std::mem::transmute(body) };
    // SAFETY: we hold `run` and the state is CLOSED with no worker inside
    unsafe { *p.job.get() = Job { f: body, n, chunk } };
    p.next.store(0, SeqCst);
    let open = ((p.state.load(SeqCst) >> 32).wrapping_add(1)) << 32;
    p.state.store(open, SeqCst);
    if p.sleepers.load(SeqCst) > 0 {
        let _g = p.lock.lock().unwrap();
        p.cv.notify_all();
    }
    // SAFETY: as above; the caller is one of the threads doing the work
    p.work(unsafe { &*p.job.get() });
    // wait for the workers still inside; if one of them lost its core, give ours away
    let mut spins = 0u32;
    while p.state.compare_exchange_weak(open, open | CLOSED, SeqCst, Relaxed).is_err() {
        spins += 1;
        if spins > 2000 {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
    drop(guard);
}

/// a pointer to a buffer whose disjoint parts are written by different threads
#[derive(Clone, Copy)]
pub struct SendPtr<T>(pub *mut T);
impl<T> SendPtr<T> {
    /// the pointer (a method, so closures capture the whole wrapper)
    pub fn get(self) -> *mut T {
        self.0
    }
}
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
