use anyhow::{Result, ensure};
use deepseek_ultrafast_cuda::{self as cu, Args, Buffer, Launch, Stream};
use deepseek_ultrafast_engine::model::Meta;
use deepseek_ultrafast_kernels::Kernels;
use half::bf16;

const DIM: usize = 512;
const RATIO: usize = 2;
const SENTINEL: f32 = 7.0;
const EPS: f32 = 1e-6;

fn group_value(group: usize, channel: usize) -> f32 {
    if ((channel >> group) & 1) == 0 {
        1.0
    } else {
        -1.0
    }
}

fn check_pool(
    kernels: &Kernels,
    stream: &Stream,
    start: usize,
    width: usize,
    ring_rows: Option<usize>,
) -> Result<()> {
    let end = start + width;
    let history_rows = ring_rows.unwrap_or(end);
    let mut keys = vec![0.0f32; history_rows * DIM];
    let mut scores = vec![0.0f32; history_rows * DIM];
    for position in end.saturating_sub(history_rows)..end {
        let physical_row = position % history_rows;
        let first = position % RATIO == 0;
        for channel in 0..DIM {
            let index = physical_row * DIM + channel;
            let magnitude = if first || channel < DIM / 2 {
                2.0
            } else {
                -2.0
            };
            keys[index] = magnitude * group_value(position / RATIO, channel);
            scores[index] = if first { 3.0f32.ln() } else { 0.0 };
        }
    }
    let meta = Buffer::from_host(&[Meta {
        pos: start as i32,
        m: width as i32,
        ..Default::default()
    }])?;
    let keys = Buffer::from_host(&keys)?;
    let scores = Buffer::from_host(&scores)?;
    let norm = Buffer::from_host(&[bf16::from_f32(1.0); DIM])?;
    let output = Buffer::from_host(&vec![bf16::from_f32(SENTINEL); width * DIM])?;
    let mut args = Args::new();
    args.ptr(meta.ptr)
        .ptr(keys.ptr)
        .ptr(scores.ptr)
        .i32(RATIO as i32)
        .ptr(norm.ptr)
        .f32(EPS)
        .ptr(output.ptr)
        .i32(DIM as i32);
    let kernel = match ring_rows {
        Some(rows) => {
            args.i32(rows as i32);
            "compress_pool_ring"
        }
        None => "compress_pool",
    };
    cu::launch(
        kernels.get(kernel)?,
        &Launch::new(width as u32, 128).pdl(true),
        stream,
        &mut args,
    )?;
    stream.synchronize()?;

    // The softmax weights are 3/4 and 1/4. The pooled magnitudes are 2 in
    // the first half and 1 in the second, giving a mean square of 2.5.
    let inverse_rms = (2.5 + EPS).sqrt().recip();
    let values = output.download::<bf16>(width * DIM)?;
    for (local, row) in values.chunks_exact(DIM).enumerate() {
        let position = start + local;
        for (channel, value) in row.iter().enumerate() {
            let expected = if (position + 1) % RATIO == 0 {
                let magnitude = if channel < DIM / 2 { 2.0 } else { 1.0 };
                group_value(position / RATIO, channel) * magnitude * inverse_rms
            } else {
                SENTINEL
            };
            let expected = bf16::from_f32(expected);
            ensure!(
                *value == expected,
                "{kernel} position {position}, channel {channel}: expected {expected}, got {value}",
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA GPU matching the kernel build"]
fn compression_pool_handles_fresh_and_wrapped_history() -> Result<()> {
    cu::init()?;
    let _device = cu::Device::new(0)?;
    let kernels = Kernels::load()?;
    let stream = Stream::new()?;
    check_pool(&kernels, &stream, 0, 16, None)?;
    check_pool(&kernels, &stream, 5, 2, None)?;
    // The group spans logical rows 4 and 5 and wraps to physical rows 4 and 0.
    check_pool(&kernels, &stream, 5, 2, Some(5))
}
