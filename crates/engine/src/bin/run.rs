use anyhow::{Context, Result};
use deepseek_ultrafast_cuda as cu;
use deepseek_ultrafast_engine::model::{DecodeSlot, DecodeToken, MAX_PACKED_DECODE_WIDTH, Meta, Rank, SpecDecoded};
use deepseek_ultrafast_engine::weights::{Arena, Loader};
use deepseek_ultrafast_model::config::Config;
use deepseek_ultrafast_model::st::Checkpoint;
use std::path::PathBuf;
use std::sync::{Barrier, Mutex};
use std::time::Instant;

const USAGE: &str = "run (--tokens <json> | --batch-tokens <json>) [--ckpt DIR] [--gpus 0,1,..] [--decode N] [--max-seq N] \
[--chunk N] [--seed N] [--runs N] [--fixed-block N] [--packed-drafts N] [--batch-logit-max-abs F] [--dump-tokens FILE] [--samp-test N] [--score FILE]";

struct Args {
    ckpt: PathBuf,
    gpus: Vec<i32>,
    prompt: Vec<i32>,
    batch_prompts: Option<Vec<Vec<i32>>>,
    decode: usize,
    max_seq: usize,
    chunk: usize,
    seed: u64,
    dump_tokens: Option<PathBuf>,
    samp_test: Option<usize>,
    score: Option<Vec<i32>>,
    runs: usize,
    fixed_block: Option<usize>,
    packed_drafts: usize,
    batch_logit_max_abs: Option<f32>,
}

fn parse_args(v: &[String]) -> Result<Args> {
    let mut a = Args {
        ckpt: PathBuf::from("/models"),
        gpus: vec![0, 1, 2, 3],
        prompt: vec![],
        batch_prompts: None,
        decode: 8,
        max_seq: 8192,
        chunk: 8192,
        seed: 0,
        dump_tokens: None,
        samp_test: None,
        score: None,
        runs: 1,
        fixed_block: None,
        packed_drafts: 0,
        batch_logit_max_abs: None,
    };
    let ids = |v: &serde_json::Value| -> Vec<i32> { v.as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap() as i32).collect()).unwrap_or_default() };
    let mut i = 1;
    while i < v.len() {
        let val = || v.get(i + 1).cloned().ok_or_else(|| anyhow::anyhow!("{} needs a value\nusage: {USAGE}", v[i]));
        match v[i].as_str() {
            "--ckpt" => a.ckpt = val()?.into(),
            "--gpus" => a.gpus = val()?.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--decode" => a.decode = val()?.parse()?,
            "--max-seq" => a.max_seq = val()?.parse()?,
            "--chunk" => a.chunk = val()?.parse()?,
            "--seed" => a.seed = val()?.parse()?,
            "--tokens" => a.prompt = serde_json::from_str(&std::fs::read_to_string(val()?)?)?,
            "--batch-tokens" => a.batch_prompts = Some(serde_json::from_str(&std::fs::read_to_string(val()?)?)?),
            "--dump-tokens" => a.dump_tokens = Some(val()?.into()),
            "--samp-test" => a.samp_test = Some(val()?.parse()?),
            "--runs" => a.runs = val()?.parse()?,
            "--fixed-block" => a.fixed_block = Some(val()?.parse()?),
            "--packed-drafts" => a.packed_drafts = val()?.parse()?,
            "--batch-logit-max-abs" => {
                let threshold: f32 = val()?.parse()?;
                anyhow::ensure!(threshold.is_finite() && threshold >= 0., "--batch-logit-max-abs requires a finite, nonnegative value");
                a.batch_logit_max_abs = Some(threshold);
            }
            "--score" => {
                let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(val()?)?)?;
                a.prompt = ids(&j["input_ids"]);
                a.score = Some(ids(&j["output_ids"]));
            }
            x => anyhow::bail!("unknown argument {x}\nusage: {USAGE}"),
        }
        i += 2;
    }
    if let Some(prompts) = &a.batch_prompts {
        anyhow::ensure!(a.prompt.is_empty() && a.score.is_none() && a.samp_test.is_none() && a.fixed_block.is_none() && a.dump_tokens.is_none() && a.runs == 1,
            "--batch-tokens runs the standalone packed decode parity probe");
        let max_lanes = MAX_PACKED_DECODE_WIDTH / a.packed_drafts.saturating_add(1);
        anyhow::ensure!((2..=max_lanes).contains(&prompts.len()),
            "--batch-tokens needs at least two prompts within the {MAX_PACKED_DECODE_WIDTH}-row verifier limit");
        anyhow::ensure!(a.decode >= 2 && prompts.iter().all(|p| !p.is_empty() && p.len().saturating_add(a.decode).saturating_add(a.packed_drafts) <= a.max_seq),
            "--batch-tokens needs nonempty prompts with room for output and the requested draft rows");
    } else {
        anyhow::ensure!(a.batch_logit_max_abs.is_none(), "--batch-logit-max-abs requires --batch-tokens");
        anyhow::ensure!(a.packed_drafts == 0, "--packed-drafts requires --batch-tokens");
        anyhow::ensure!(!a.prompt.is_empty(), "no prompt (use --tokens or --score)\nusage: {USAGE}");
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args(&std::env::args().collect::<Vec<_>>())?;
    deepseek_ultrafast_model::st::require_checkpoint(&args.ckpt)?;
    cu::init()?;
    cu::profile::init(cu::device_cc(args.gpus[0])?, args.gpus.len());
    let _ = rayon::ThreadPoolBuilder::new().stack_size(64 << 20).build_global();
    let cfg = deepseek_ultrafast_engine::load_config(&args.ckpt)?;
    let t0 = Instant::now();
    let ck = Checkpoint::open(&args.ckpt)?;
    let engram = deepseek_ultrafast_engine::engram::EngramCell::new(&args.ckpt, &cfg);
    let tp = args.gpus.len();
    let prompt = &args.prompt;

    let barrier = Barrier::new(tp);
    let peers: Mutex<Vec<u64>> = Mutex::new(vec![0; tp]);
    let shards: Mutex<Vec<Vec<f32>>> = Mutex::new(vec![vec![]; tp]);
    let ctxs: Mutex<Vec<usize>> = Mutex::new(vec![0; tp]);
    std::thread::scope(|sc| -> Result<()> {
        sc.spawn(|| engram.get().map(|_| ()));
        let mut hs = Vec::new();
        for r in 0..tp {
            let (ck, engram, cfg, args, barrier, peers, ctxs, shards) = (&ck, &engram, &cfg, &args, &barrier, &peers, &ctxs, &shards);
            hs.push(std::thread::Builder::new().stack_size(256 << 20).spawn_scoped(sc, move || -> Result<()> {
                let res = (move || -> Result<()> {
                    let dev = cu::Device::new(args.gpus[r])?;
                    ctxs.lock().unwrap()[r] = dev.ctx as usize;
                    barrier.wait();
                    for (q, c) in ctxs.lock().unwrap().clone().into_iter().enumerate() {
                        if q != r {
                            let res = unsafe { cu::sys::cuCtxEnablePeerAccess(c as cu::sys::CUcontext, 0) };
                            if res != cu::sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED {
                                cu::check(res, "peer access")?;
                            }
                        }
                    }
                    let mut arena = Arena::default();
                    let lt = Instant::now();
                    let w = Loader { ck, cfg, rank: r, tp, tc: Default::default() }.load(&mut arena, cfg.n_layers, true)?;
                    let engram = engram.get()?.upload(&mut arena)?;
                    println!("rank {r}: weights {:.1} GB in {:.1}s", arena.bytes as f64 / 1e9, lt.elapsed().as_secs_f32());
                    let constructor = if args.batch_prompts.is_some() { Rank::new_packed } else { Rank::new };
                    let mut rk = constructor(r, tp, cfg.clone(), dev, w, arena, args.max_seq, args.chunk, engram)?;
                    rk.set_sampling(args.seed)?;
                    peers.lock().unwrap()[r] = rk.comm.recv;
                    barrier.wait();
                    rk.peers = peers.lock().unwrap().clone();
                    deepseek_ultrafast_engine::model::setup_nvls(&mut rk, &args.gpus, barrier)?;
                    let (free, _) = rk.dev.mem_info()?;
                    if r == 0 {
                        println!("ready after {:.1}s, {:.1} GB free per GPU", t0.elapsed().as_secs_f32(), free as f64 / 1e9);
                    }
                    barrier.wait();

                    if let Some(prompts) = &args.batch_prompts {
                        return batch_probe(&mut rk, prompts, args);
                    }
                    let ptime = Instant::now();
                    let tok = rk.prefill(prompt, 0, args.chunk)?;
                    if let Some(cont) = &args.score {
                        return score(&mut rk, r, tp, cfg, barrier, shards, prompt.len(), cont);
                    }
                    if r == 0 {
                        println!("prefill {} tokens: {:.1} ms (first token {tok})", prompt.len(), ptime.elapsed().as_secs_f64() * 1e3);
                    }
                    if args.decode == 0 {
                        return Ok(());
                    }
                    if let Some(n) = args.samp_test {
                        return samp_test(&mut rk, r, cfg, barrier, shards, prompt.len(), tok, n, args.seed);
                    }
                    rk.spec_begin(tok, prompt.len())?;
                    let g = match args.fixed_block {
                        Some(b) => rk.capture_spec_blk(b)?,
                        None => rk.capture_spec()?,
                    };
                    let (mut all_steps, mut all_committed, mut all_dt) = (0u64, 0u64, 0f64);
                    for run in 0..args.runs {
                        let seed = args.seed + run as u64;
                        if run > 0 {
                            rk.set_sampling(seed)?;
                        }
                        rk.spec_begin(tok, prompt.len())?;
                        barrier.wait();
                        let st = Instant::now();
                        let mut launched = 0;
                        loop {
                            for _ in 0..8 {
                                g.launch(&rk.stream)?;
                            }
                            launched += 8;
                            let (_, _, committed) = rk.spec_stats()?;
                            if committed as usize + 1 >= args.decode || launched > 4 * args.decode {
                                break;
                            }
                        }
                        let dt = st.elapsed().as_secs_f64();
                        let (steps, acc, committed) = rk.spec_stats()?;
                        (all_steps, all_committed, all_dt) = (all_steps + steps as u64, all_committed + committed as u64, all_dt + dt);
                        if r != 0 {
                            continue;
                        }
                        let out = rk.read_out(committed as usize)?;
                        let mut generated = vec![tok];
                        generated.extend_from_slice(&out);
                        let n = generated.len();
                        let label = if args.runs > 1 { format!("graph spec decode (seed {seed})") } else { "graph spec decode".to_string() };
                        println!("{label}: {n} tokens in {steps} steps ({:.2} tokens/step, {:.2} accepted drafts/step), {:.1} ms ({:.1} tok/s, {:.3} ms/step)", committed as f64 / steps as f64, acc as f64 / steps as f64, dt * 1e3, (committed as f64) / dt, dt * 1e3 / steps as f64);
                        if run + 1 < args.runs {
                            continue;
                        }
                        println!("generated: {:?}", &generated[..n.min(40)]);

                        if let Some(path) = &args.dump_tokens {
                            std::fs::write(path, serde_json::to_string(&generated)?)?;
                        }
                        if args.runs > 1 {
                            println!("all {} runs: {all_committed} tokens in {all_steps} steps ({:.3} tokens/step), {:.1} tok/s, {:.4} ms/step", args.runs, all_committed as f64 / all_steps as f64, all_committed as f64 / all_dt, all_dt * 1e3 / all_steps as f64);
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = &res {
                    eprintln!("[rank {r}] error: {e:#}");
                }
                res
            }));
        }
        for h in hs {
            h?.join().unwrap()?;
        }
        Ok(())
    })?;
    Ok(())
}

fn probe_phase<T>(rk: &mut Rank, phase: &str, run: impl FnOnce(&mut Rank) -> Result<T>) -> Result<T> {
    let rank = rk.rank;
    if rank == 0 {
        eprintln!("kiln_packed_probe_phase phase={phase} status=started");
    }
    let output = run(rk).with_context(|| format!("packed probe rank {rank} phase {phase}"))?;
    if rank == 0 {
        eprintln!("kiln_packed_probe_phase phase={phase} status=returned");
    }
    Ok(output)
}

fn batch_probe(rk: &mut Rank, prompts: &[Vec<i32>], args: &Args) -> Result<()> {
    let phase_sync = matches!(std::env::var("DSUF_BATCH_PROBE_SYNC").as_deref(), Ok("1"));
    if phase_sync {
        probe_phase(rk, "initial_stream_fence", |rank| rank.synchronize_packed_compute())?;
    }
    rk.set_packed_target_rows(prompts.len() * (1 + args.packed_drafts))?;
    anyhow::ensure!(prompts.iter().flatten().all(|&token| (0..rk.cfg.vocab_size as i32).contains(&token)), "batch probe prompt token exceeds vocabulary");
    if args.packed_drafts > 0 {
        rk.validate_packed_spec(args.packed_drafts)?;
    }
    let mut first_tokens = Vec::with_capacity(prompts.len());
    let mut reference_steps = Vec::with_capacity(prompts.len());
    let mut isolated_elapsed = std::time::Duration::ZERO;
    let verify_rows = args.packed_drafts + 1;
    let mut reference_slot = probe_phase(rk, "dense_reference_reserve", |rank| rank.new_dense_reference_slot())?;
    for (sequence, prompt) in prompts.iter().enumerate() {
        let slot = &mut reference_slot;
        let mut token = probe_phase(rk, &format!("dense_reference_prefill_{sequence}"), |rank| {
            probe_prefill(rank, slot, prompt, args, sequence, phase_sync)
        })?;
        first_tokens.push(token);
        let mut emitted = 1;
        let mut steps = Vec::new();
        let start = Instant::now();
        while emitted < args.decode {
            let input = DecodeToken { position: prompt.len() + emitted - 1, token };
            let result = probe_phase(rk, &format!("width_one_forward_{sequence}_{}", steps.len()), |rank| {
                probe_map(rank, slot, input.position + verify_rows, phase_sync)?;
                Ok(probe_decode(rank, &mut [(slot, input)], args.packed_drafts)?.remove(0))
            })?;
            token = *result.tokens.last().expect("a decode step emits a token");
            emitted += result.tokens.len();
            steps.push(result);
        }
        isolated_elapsed += start.elapsed();
        probe_phase(rk, &format!("dense_reference_release_{sequence}"), |rank| rank.release_decode_slot(slot))?;
        reference_steps.push(steps);
    }

    drop(reference_slot);
    let mut slots = probe_phase(rk, "paged_slots_reserve", |rank| {
        (0..prompts.len()).map(|_| rank.new_decode_slot()).collect::<Result<Vec<_>>>()
    })?;
    for (sequence, (prompt, slot)) in prompts.iter().zip(&mut slots).enumerate() {
        let first = probe_phase(rk, &format!("paged_prefill_{sequence}"), |rank| {
            probe_prefill(rank, slot, prompt, args, sequence, phase_sync)
        })?;
        anyhow::ensure!(first == first_tokens[sequence], "paged prefill diverged from the dense reference at lane {sequence}");
    }
    let mut anchors: Vec<_> = prompts.iter().zip(&first_tokens).map(|(prompt, &token)| DecodeToken {
        position: prompt.len(), token,
    }).collect();
    let step_count = reference_steps.iter().map(Vec::len).max().unwrap();
    let mut accepted = 0;
    let mut steps = 0;
    let mut emitted = 0;
    let mut verifier_rows = 0;
    let mut anchor_logits = Vec::new();
    let mut packed_elapsed = std::time::Duration::ZERO;
    let mut start = Instant::now();
    for step in 0..step_count {
        let mut lanes = Vec::new();
        let mut sequences = Vec::new();
        for (sequence, slot) in slots.iter_mut().enumerate() {
            if step < reference_steps[sequence].len() {
                let anchor = anchors[sequence];
                probe_map(rk, slot, anchor.position + verify_rows, phase_sync)
                    .with_context(|| format!("packed step {step} sequence {sequence} map"))?;
                lanes.push((slot, anchor));
                sequences.push(sequence);
            }
        }
        if step % 2 == 1 {
            lanes.reverse();
            sequences.reverse();
        }
        let rotate = step % lanes.len();
        lanes.rotate_left(rotate);
        sequences.rotate_left(rotate);
        verifier_rows += lanes.len() * verify_rows;
        let generated = probe_phase(rk, &format!("packed_forward_step_{step}_width_{}", lanes.len()), |rank| {
            probe_decode(rank, &mut lanes, args.packed_drafts)
        })?;
        if step == 0 {
            // Exclude this diagnostic D2H copy from packed decode timing. Step zero is
            // in manifest order and row zero of each lane scores the teacher anchor.
            packed_elapsed += start.elapsed();
            let vocab_shard = rk.cfg.vocab_size / rk.tp;
            let logits = probe_phase(rk, "packed_anchor_logits", |rank| rank.read_logits(lanes.len() * verify_rows))?;
            anchor_logits = logits.chunks_exact(verify_rows * vocab_shard)
                .map(|lane| lane[..vocab_shard].to_vec()).collect();
            start = Instant::now();
        }
        for (sequence, output) in sequences.into_iter().zip(generated) {
            anyhow::ensure!(output == reference_steps[sequence][step],
                "packed decode diverged from isolated seeded decode at sequence {sequence}, step {step}");
            anchors[sequence].position += output.tokens.len();
            anchors[sequence].token = *output.tokens.last().expect("a decode step emits a token");
            emitted += output.tokens.len();
            accepted += output.accepted_drafts;
            steps += 1;
        }
    }
    packed_elapsed += start.elapsed();
    probe_phase(rk, "packed_slots_release", |rank| {
        for slot in &mut slots {
            rank.release_decode_slot(slot)?;
        }
        Ok(())
    })?;
    if rk.rank == 0 {
        println!("packed target graph ready = {}", rk.packed_target_graph_ready(args.packed_drafts > 0));
        println!("packed width {}, {} drafts: {emitted} emitted tokens in {step_count} launches and {verifier_rows} verifier rows, 0 token mismatches, {:.2} useful tokens/launch, {:.2} accepted drafts per sequence step, {:.1} isolated tokens/s, {:.1} packed tokens/s",
            prompts.len(), args.packed_drafts, emitted as f64 / step_count as f64, accepted as f64 / steps as f64,
            emitted as f64 / isolated_elapsed.as_secs_f64(), emitted as f64 / packed_elapsed.as_secs_f64());
    }
    drop(slots);
    probe_phase(rk, "incumbent_logit_comparison", |rank| {
        compare_incumbent_logits(rank, args, prompts, &first_tokens, &anchor_logits, phase_sync)
    })
}

fn compare_incumbent_logits(
    rk: &mut Rank,
    args: &Args,
    prompts: &[Vec<i32>],
    anchors: &[i32],
    packed_logits: &[Vec<f32>],
    phase_sync: bool,
) -> Result<()> {
    anyhow::ensure!(prompts.len() == anchors.len() && anchors.len() == packed_logits.len());
    let mut dense_slot = rk.new_dense_reference_slot()?;
    let (mut max_abs, mut squared, mut count, mut nonfinite) = (0f64, 0f64, 0usize, 0usize);
    let mut worst = (0usize, 0usize);
    for (sequence, ((prompt, &anchor), packed)) in prompts.iter().zip(anchors).zip(packed_logits).enumerate() {
        probe_prefill(rk, &mut dense_slot, prompt, args, sequence, phase_sync)
            .with_context(|| format!("incumbent prefill sequence {sequence}"))?;
        probe_map(rk, &mut dense_slot, prompt.len() + 1, phase_sync)
            .with_context(|| format!("incumbent anchor map sequence {sequence}"))?;
        let dense = rk.with_decode_slot(&mut dense_slot, |rank| {
            let mut meta = Meta { pos: prompt.len() as i32, m: 1, ..Default::default() };
            meta.tokens[0] = anchor;
            rank.set_meta(&meta)?;
            rank.forward(1, rank.cfg.n_layers, None)?;
            rank.read_logits(1)
        }).with_context(|| format!("incumbent anchor forward sequence {sequence}"))?;
        rk.release_decode_slot(&mut dense_slot)
            .with_context(|| format!("incumbent release sequence {sequence}"))?;
        anyhow::ensure!(dense.len() == packed.len(), "teacher-forced vocabulary shard differs");
        for (id, (&reference, &candidate)) in dense.iter().zip(packed).enumerate() {
            let error = (f64::from(reference) - f64::from(candidate)).abs();
            if !error.is_finite() { nonfinite += 1; continue; }
            if error > max_abs {
                max_abs = error;
                worst = (sequence, rk.rank * dense.len() + id);
            }
            squared += error * error;
            count += 1;
        }
    }
    let rms = (squared / count.max(1) as f64).sqrt();
    println!("teacher-forced incumbent comparison rank {}: {count} finite logits, {nonfinite} nonfinite, max_abs {max_abs:.7}, rms {rms:.7}, worst sequence {} token {}", rk.rank, worst.0, worst.1);
    anyhow::ensure!(nonfinite == 0, "nonfinite teacher-forced logit difference on rank {}", rk.rank);
    if let Some(limit) = args.batch_logit_max_abs {
        anyhow::ensure!(max_abs <= f64::from(limit), "teacher-forced logit difference {max_abs} exceeds {limit} on rank {}", rk.rank);
    }
    Ok(())
}

fn probe_map(rk: &mut Rank, slot: &mut DecodeSlot, through: usize, phase_sync: bool) -> Result<()> {
    rk.ensure_decode_slot(slot, through)
        .with_context(|| format!("map probe slot through {through} tokens"))?;
    if phase_sync {
        rk.synchronize_packed_compute()
            .with_context(|| format!("map probe slot stream fence through {through} tokens"))?;
    }
    Ok(())
}

fn probe_prefill(
    rk: &mut Rank,
    slot: &mut DecodeSlot,
    prompt: &[i32],
    args: &Args,
    sequence: usize,
    phase_sync: bool,
) -> Result<i32> {
    let seed = args.seed.wrapping_add(sequence as u64);
    probe_map(rk, slot, prompt.len(), phase_sync)?;
    rk.with_decode_slot(slot, |rank| {
        rank.set_sampling(seed)?;
        rank.prefill(prompt, 0, args.chunk)
    })
}

fn probe_decode(rk: &mut Rank, lanes: &mut [(&mut DecodeSlot, DecodeToken)], drafts: usize) -> Result<Vec<SpecDecoded>> {
    if drafts > 0 {
        rk.spec_decode_batch(lanes, drafts)
    } else {
        Ok(rk.decode_batch(lanes)?.into_iter().map(|token| SpecDecoded { tokens: vec![token], accepted_drafts: 0 }).collect())
    }
}

#[allow(clippy::too_many_arguments)]
fn samp_test(rk: &mut Rank, r: usize, cfg: &Config, barrier: &Barrier, shards: &Mutex<Vec<Vec<f32>>>, plen: usize, tok: i32, n: usize, seed: u64) -> Result<()> {
    rk.spec_begin(tok, plen)?;
    let gs = rk.capture_spec()?;
    let mut spec: Vec<i32> = vec![];
    let mut stepsum = 0usize;
    for t in 0..n {
        rk.spec_begin(tok, plen)?;
        rk.set_sampling(seed * 1_000_003 + t as u64)?;
        gs.launch(&rk.stream)?;
        gs.launch(&rk.stream)?;
        let (_, acc, committed) = rk.spec_stats()?;
        stepsum += acc as usize;
        let o = rk.read_out_nosync(committed as usize)?;
        spec.push(o[0]);
    }
    let meta0 = Meta { pos: plen as i32, m: 1, tokens: { let mut t = [0i32; 16]; t[0] = tok; t } };
    rk.set_meta(&meta0)?;
    rk.forward(1, cfg.n_layers, None)?;
    shards.lock().unwrap()[r] = rk.read_logits(1)?;
    barrier.wait();
    if r == 0 {
        let z: Vec<f64> = shards.lock().unwrap().concat().iter().map(|&x| x as f64).collect();
        let mx = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lse = mx + z.iter().map(|&x| (x - mx).exp()).sum::<f64>().ln();
        let ex: Vec<f64> = z.iter().map(|&x| (x - lse).exp()).collect();
        let mut ord: Vec<usize> = (0..ex.len()).collect();
        ord.sort_by(|&a, &b| ex[b].partial_cmp(&ex[a]).unwrap());
        println!("exact: top probs {:?}", ord.iter().take(5).map(|&i| (i, (ex[i] * 1e4).round() / 1e4)).collect::<Vec<_>>());
        let mut h = vec![0usize; ex.len()];
        let mut bad = 0;
        for &a in &spec {
            if a >= 0 && (a as usize) < h.len() { h[a as usize] += 1; } else { bad += 1; }
        }
        let (mut chi, mut dof, mut re, mut ro) = (0f64, 0usize, 0f64, 0f64);
        let (mut te, mut to) = (0f64, 0usize);
        for i in 0..ex.len() {
            let e = ex[i] * n as f64;
            if e >= 5.0 { chi += (h[i] as f64 - e).powi(2) / e; dof += 1; } else { re += e; ro += h[i] as f64; }
            if ex[i] < 1e-3 { te += e; to += h[i]; }
        }
        if re > 0.0 { chi += (ro - re).powi(2) / re; dof += 1; }
        println!("spec vs exact (first token): chi2 {chi:.1} on {} dof (p > 0.01 needs < ~{:.0}); tokens with p < 1e-3: expected {te:.1}, got {to}; invalid ids: {bad}", dof.saturating_sub(1), dof as f64 + 2.33 * (2.0 * dof as f64).sqrt());
        println!("spec: {:.2} accepted drafts per step over {} trials of 2 steps", stepsum as f64 / (2 * n) as f64, n);
    }
    barrier.wait();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn score(rk: &mut Rank, r: usize, tp: usize, cfg: &Config, barrier: &Barrier, shards: &Mutex<Vec<Vec<f32>>>, plen: usize, cont: &[i32]) -> Result<()> {
    const BLOCK: usize = 8;
    let n = cont.len();
    anyhow::ensure!(n >= 1, "--score: empty continuation");
    let v = cfg.vocab_size / tp;
    let (mut nll_sum, mut agree) = (0f64, 0usize);
    let mut nlls: Vec<f64> = vec![];
    let st = Instant::now();
    let mut eval = |full: &[f32], tok: i32| {
        let mx = full.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let lse = mx + full.iter().map(|x| (*x as f64 - mx).exp()).sum::<f64>().ln();
        let am = full.iter().enumerate().fold((0usize, f32::MIN), |b, (i, x)| if *x > b.1 { (i, *x) } else { b }).0;
        let nll = lse - full[tok as usize] as f64;
        nll_sum += nll;
        nlls.push(nll);
        agree += (am as i32 == tok) as usize;
    };
    let lg = rk.read_last_logits()?;
    shards.lock().unwrap()[r] = lg[..v].to_vec();
    barrier.wait();
    if r == 0 {
        let full: Vec<f32> = shards.lock().unwrap().concat();
        eval(&full, cont[0]);
    }
    barrier.wait();
    let mut j = 0usize;
    while j + 1 < n {
        let b = BLOCK.min(n - 1 - j);
        let mut meta = Meta { pos: (plen + j) as i32, m: b as i32, tokens: [0; 16] };
        meta.tokens[..b].copy_from_slice(&cont[j..j + b]);
        rk.set_meta(&meta)?;
        rk.forward(b, cfg.n_layers, None)?;
        shards.lock().unwrap()[r] = rk.read_logits(b)?;
        barrier.wait();
        if r == 0 {
            let sh = shards.lock().unwrap();
            for i in 0..b {
                let full: Vec<f32> = (0..tp).flat_map(|q| sh[q][i * v..(i + 1) * v].iter().copied()).collect();
                eval(&full, cont[j + i + 1]);
            }
        }
        barrier.wait();
        j += b;
    }
    if r == 0 {
        let cnt = nlls.len();
        let mean = nll_sum / cnt as f64;
        let sd = (nlls.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (cnt as f64 - 1.0).max(1.0)).sqrt();
        println!("score: {cnt} tokens ({:.1} s): mean NLL {mean:.5} (+- {:.5} s.e.), ppl {:.4}, argmax == recorded {agree}/{cnt}", st.elapsed().as_secs_f64(), sd / (cnt as f64).sqrt(), mean.exp());
    }
    Ok(())
}
