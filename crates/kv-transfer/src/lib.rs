mod address;
mod align;
mod directory;
mod encoding;
#[cfg(feature = "local-kv")]
#[path = "local.rs"]
mod rdma;
#[cfg(not(feature = "local-kv"))]
mod rdma;
mod snapshot;

#[cfg(all(feature = "local-kv", feature = "rdma"))]
compile_error!("local-kv and rdma are mutually exclusive transports");
#[cfg(test)]
mod tests;
pub use address::{DEFAULT_KV_PORT, kv_address};
pub use align::{Aligned, TokenText, align};
pub use directory::{DEFAULT_DIRECTORY_PREFIX, DEFAULT_REDIS_URL, Revision};
pub use encoding::{Encoding, Segment};
pub use rdma::{RdmaRequired, check_rdma, is_rdma_required};
pub use snapshot::{ImageSpan, MAX_IMAGES, Manifest, Prompt, SAVED_TAIL, Snapshot, TAIL_READ, VERSION, digest, image_prefix, payload_digest};

use anyhow::{Result, ensure};
use deepseek_ultrafast_logging::{Event as LogEvent, EvictionReason};
use directory::Onboarded;
use futures::StreamExt;
use rdma::{DataPlane, Staged};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Weak, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use uuid::Uuid;
use velo::{DataHandle, Handler, InstanceId, PeerInfo, TypedContext, Velo};

const HANDLER: &str = "deepseek-ultrafast.kv.rdma.v2";
pub const MAX_LEASES: usize = 8;
const MAX_ENTRIES: usize = 1024;
const LEASE_CLOSE_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Clone)]
pub struct Config {
    pub redis_url: String,
    pub directory_prefix: String,
    pub fingerprint: String,
    pub bind: SocketAddr,
    pub max_bytes: usize,
    pub max_snapshot_bytes: usize,
    pub rdma_bytes: u64,
    pub max_seq: usize,
    pub timeout: Duration,
    pub token_text: Option<Arc<TokenText>>,
}

#[derive(Debug, Serialize, Deserialize)]
enum Fetch {
    Open { key: String, revision: Revision },
    Checkpoint { key: String, revision: Revision },
    Close { lease: Uuid },
}
#[derive(Debug, Serialize, Deserialize)]
enum Reply {
    Miss,
    Open {
        lease: Uuid,
        revision: Revision,
        manifest: Box<Manifest>,
        handle: DataHandle,
        #[serde(default)]
        checkpoint: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint_tokens: Option<usize>,
    },
    Closed,
}
struct Offer {
    lease: DonorLease,
    revision: Revision,
    manifest: Box<Manifest>,
    handle: DataHandle,
    common: usize,
    aligned: Option<Aligned>,
}

/// An application-level offer lease. Velo's separate read lease protects the
/// staged region if a cancelled GET is still in flight.
struct DonorLease {
    id: Uuid,
    peer: InstanceId,
    velo: Arc<Velo>,
    runtime: tokio::runtime::Handle,
    cleanup_slot: Option<OwnedSemaphorePermit>,
}

impl Drop for DonorLease {
    fn drop(&mut self) {
        let Some(cleanup_slot) = self.cleanup_slot.take() else {
            return;
        };
        let (lease, peer, velo) = (self.id, self.peer, self.velo.clone());
        self.runtime.spawn(async move {
            let deadline = Instant::now() + LEASE_CLOSE_TIMEOUT;
            let _ = peer_rpc(&velo, peer, Fetch::Close { lease }, deadline).await;
            drop(cleanup_slot);
        });
    }
}

async fn peer_rpc(velo: &Velo, peer: InstanceId, request: Fetch, deadline: Instant) -> Result<Reply> {
    ensure!(Instant::now() < deadline, "KV peer deadline elapsed");
    tokio::time::timeout_at(
        deadline.into(),
        velo.typed_unary::<Reply>(HANDLER)?.payload(request)?.instance(peer).send(),
    )
    .await
    .inspect_err(|_| deepseek_ultrafast_logging::kv_failure("control_timeout"))?
    .inspect_err(|_| deepseek_ultrafast_logging::kv_failure("control_request"))
}

struct Stored {
    snapshot: Arc<Snapshot>,
    staged: Arc<Staged>,
}
struct Entry {
    stored: Arc<Stored>,
    checkpoint: Option<Arc<Stored>>,
    revision: Revision,
    used: u64,
    last_used: Instant,
    session_id: Option<Arc<str>>,
}
impl Entry {
    #[cfg(test)]
    fn select(&self, prompt: Prompt) -> Option<&Arc<Stored>> {
        self.select_by(prompt.tokens.len(), |m| (m.reusable_prefix(prompt), None)).map(|(s, _)| s)
    }
    fn select_by(&self, prompt_len: usize, mut reuse: impl FnMut(&Manifest) -> (usize, Option<Aligned>))
        -> Option<(&Arc<Stored>, Option<Aligned>)> {
        let (common, aligned) = reuse(&self.stored.snapshot.manifest);
        if common < prompt_len.saturating_sub(1)
            && let Some(checkpoint) = &self.checkpoint
            && checkpoint.snapshot.manifest.tokens.len() > common {
            let (c, a) = reuse(&checkpoint.snapshot.manifest);
            if c > common {
                return Some((checkpoint, a));
            }
        }
        (common > 0).then_some((&self.stored, aligned))
    }
    fn bytes(&self) -> usize {
        self.stored.snapshot.manifest.bytes
            + self.checkpoint.as_ref().map_or(0, |s| s.snapshot.manifest.bytes)
    }
    fn checkpoint_for(&self, next: &Manifest, input_tokens: usize) -> Option<Arc<Stored>> {
        if input_tokens == next.tokens.len() { return None; }
        let input = next.prefix(input_tokens);
        std::iter::once(&self.stored).chain(self.checkpoint.as_ref())
            .filter(|s| s.snapshot.manifest.restore_start() < next.restore_start())
            .filter_map(|s| {
                let common = s.snapshot.manifest.reusable_prefix(input);
                (common > 0).then_some((common, s))
            })
            .max_by_key(|(common, _)| *common)
            .map(|(_, s)| s.clone())
    }
}
struct Lease {
    stored: Arc<Stored>,
    expires: Instant,
}
#[derive(Default)]
struct State {
    watching: bool,
    epoch: Option<Uuid>,
    entries: HashMap<String, Entry>,
    pending_claims: HashMap<String, Vec<Weak<AtomicU64>>>,
    leases: HashMap<Uuid, Lease>,
    clock: u64,
    evictions: Vec<LogEvent>,
}
impl State {
    fn remove(&mut self, key: &str, reason: EvictionReason) {
        if let Some(entry) = self.entries.remove(key)
            && let Some(session_id) = entry.session_id {
            self.evictions.push(LogEvent::KvEvicted { session_id, reason, idle_age_ms: deepseek_ultrafast_logging::ms(entry.last_used.elapsed()) });
        }
    }
    fn take_evictions(&mut self) -> Vec<LogEvent> { std::mem::take(&mut self.evictions) }
    fn expire(&mut self) {
        self.leases.retain(|_, l| l.expires > Instant::now());
    }
    fn clear_local(&mut self) {
        self.entries.clear();
        self.pending_claims.clear();
    }
    fn epoch(&mut self, epoch: Uuid) {
        if self.epoch != Some(epoch) {
            self.clear_local();
            self.epoch = Some(epoch);
        }
    }
    fn invalidate(&mut self, key: &str, revision: Revision) {
        if self.epoch == Some(revision.epoch)
            && let Some(claims) = self.pending_claims.get(key)
        {
            for claim in claims.iter().filter_map(Weak::upgrade) {
                claim.fetch_max(revision.sequence, Ordering::Relaxed);
            }
        }
        if self.entries.get(key).is_some_and(|e| {
            e.revision.epoch == revision.epoch && e.revision.sequence < revision.sequence
        }) {
            self.remove(key, EvictionReason::OtherReplica);
        }
    }
    fn start_claim(&mut self, key: &str) -> Arc<AtomicU64> {
        // A cancelled claimant drops its ticket without needing to take this
        // lock from a destructor.
        self.pending_claims.retain(|_, claims| {
            claims.retain(|claim| claim.strong_count() > 0);
            !claims.is_empty()
        });
        let ticket = Arc::new(AtomicU64::new(0));
        self.pending_claims
            .entry(key.to_owned())
            .or_default()
            .push(Arc::downgrade(&ticket));
        ticket
    }
    fn finish_claim(&mut self, key: &str, ticket: &Arc<AtomicU64>) -> Option<u64> {
        let claims = self.pending_claims.get_mut(key)?;
        let index = claims
            .iter()
            .position(|claim| claim.as_ptr() == Arc::as_ptr(ticket))?;
        claims.swap_remove(index);
        if claims.is_empty() {
            self.pending_claims.remove(key);
        }
        Some(ticket.load(Ordering::Relaxed))
    }
    #[cfg(test)]
    fn insert(&mut self, key: String, revision: Revision, stored: Arc<Stored>, budget: usize) {
        self.install(key, revision, stored, budget, None);
    }
    #[cfg(test)]
    fn install(&mut self, key: String, revision: Revision, stored: Arc<Stored>, budget: usize,
               predecessor: Option<(Revision, usize)>) {
        self.install_named(key, revision, stored, budget, predecessor, None);
    }
    fn install_named(&mut self, key: String, revision: Revision, stored: Arc<Stored>, budget: usize,
                     predecessor: Option<(Revision, usize)>, session_id: Option<Arc<str>>) {
        self.epoch(revision.epoch);
        let checkpoint = self.entries.remove(&key)
            .filter(|e| Some(e.revision) == predecessor.map(|(revision, _)| revision))
            .and_then(|e| e.checkpoint_for(&stored.snapshot.manifest, predecessor.unwrap().1))
            .filter(|s| s.snapshot.manifest.bytes.checked_add(stored.snapshot.manifest.bytes)
                .is_some_and(|bytes| bytes <= budget));
        let bytes = stored.snapshot.manifest.bytes
            + checkpoint.as_ref().map_or(0, |s| s.snapshot.manifest.bytes);
        while self.entries.len() >= MAX_ENTRIES
            || bytes > budget.saturating_sub(self.entries.values().map(Entry::bytes).sum::<usize>())
        {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.remove(&victim, EvictionReason::Lru);
        }
        self.clock += 1;
        self.entries.insert(
            key,
            Entry {
                stored,
                checkpoint,
                revision,
                used: self.clock,
                last_used: Instant::now(),
                session_id,
            },
        );
    }
    fn serve(&mut self, req: Fetch, config: &Config) -> Reply {
        self.expire();
        let checkpoint_request = matches!(req, Fetch::Checkpoint { .. });
        match req {
            Fetch::Open { key, revision } | Fetch::Checkpoint { key, revision } => {
                let Some(e) = self.entries.get_mut(&key).filter(|e| {
                    e.revision.epoch == revision.epoch && e.revision.sequence >= revision.sequence
                }) else {
                    return Reply::Miss;
                };
                let stored = if checkpoint_request {
                    let Some(stored) = &e.checkpoint else { return Reply::Miss };
                    stored
                } else { &e.stored };
                let leased = self
                    .leases
                    .values()
                    .map(|l| l.stored.snapshot.manifest.bytes)
                    .sum::<usize>();
                if self.leases.len() >= MAX_LEASES
                    || stored.snapshot.manifest.bytes > config.max_bytes.saturating_sub(leased)
                {
                    return Reply::Miss;
                }
                self.clock += 1;
                e.used = self.clock;
                e.last_used = Instant::now();
                let lease = Uuid::new_v4();
                let manifest = Box::new(stored.snapshot.manifest.clone());
                let handle = stored.staged.handle;
                self.leases.insert(
                    lease,
                    Lease {
                        stored: stored.clone(),
                        expires: Instant::now() + config.timeout,
                    },
                );
                Reply::Open {
                    lease,
                    revision: e.revision,
                    manifest,
                    handle,
                    checkpoint: !checkpoint_request && e.checkpoint.is_some(),
                    checkpoint_tokens: if checkpoint_request { None } else {
                        e.checkpoint.as_ref().map(|s| s.snapshot.manifest.tokens.len())
                    },
                }
            }
            Fetch::Close { lease } => {
                self.leases.remove(&lease);
                Reply::Closed
            }
        }
    }
}

#[derive(Clone)]
pub struct Candidate {
    pub revision: Revision,
    pub snapshot: Option<Arc<Snapshot>>,
    pub aligned: Option<Aligned>,
    staged: Option<Arc<Staged>>,
    local: bool,
    key: String,
}
impl Candidate {
    pub fn is_local(&self) -> bool {
        self.local && self.snapshot.is_some()
    }
    pub fn needs_onboard(&self) -> bool {
        !self.local && self.snapshot.is_some()
    }
}
fn report_prefix_miss(key: &str, manifest: &Manifest, prompt: Prompt) {
    if deepseek_ultrafast_logging::serving() { return; }
    let tokens = manifest.token_prefix(prompt.tokens);
    let matched = image_prefix(tokens, &manifest.images, prompt.images);
    let reason = if tokens == 0 { "token_prefix_mismatch" } else if matched == 0 { "image_mismatch" } else { "outside_saved_tail" };
    deepseek_ultrafast_logging::diagnostic!("KV cache miss: key={key} reason={reason} prompt={} stored={} matched={matched} images={}/{} tail_start={} required_tail={TAIL_READ}",
              prompt.tokens.len(), manifest.tokens.len(), prompt.images.len(), manifest.images.len(), manifest.tail_start);
}

pub struct PeerCache {
    config: Config,
    plane: DataPlane,
    dir: directory::Directory,
    scope: String,
    state: Arc<Mutex<State>>,
    fetch_slot: Semaphore,
    lease_cleanup_slots: Arc<Semaphore>,
    data_plane_failures: AtomicU64,
    closing: Arc<AtomicBool>,
    tasks: Vec<JoinHandle<()>>,
}
async fn subscribe(client: &redis::Client, channel: &str) -> Result<impl futures::Stream<Item = redis::Msg> + use<>> {
    let mut pubsub = client.get_async_pubsub().await?;
    pubsub.subscribe(channel).await?;
    Ok(pubsub.into_on_message())
}
impl PeerCache {
    pub async fn start(config: Config) -> Result<Self> {
        ensure!(
            !config.fingerprint.is_empty(),
            "KV profile required"
        );
        ensure!(
            config.max_bytes > 0
                && config.max_snapshot_bytes > 0
                && config.max_snapshot_bytes <= config.max_bytes
                && config.rdma_bytes > 0
                && config.max_seq > 0
                && !config.timeout.is_zero(),
            "invalid KV limits"
        );
        ensure!(
            !config.bind.ip().is_unspecified(),
            "KV bind must be a routable IP, not a wildcard"
        );
        directory::validate_prefix(&config.directory_prefix)?;
        let redis = redis::Client::open(config.redis_url.as_str())?;
        let plane = DataPlane::start(config.bind, config.rdma_bytes, config.timeout).await
            .inspect_err(|_| deepseek_ultrafast_logging::kv_failure("startup"))?;
        Self::connect(config, plane, redis).await
    }
    async fn connect(config: Config, plane: DataPlane, redis: redis::Client) -> Result<Self> {
        let transport_version = if cfg!(feature = "local-kv") { "local-v1" } else { "rdma-v4" };
        let scope = digest(
            format!("{transport_version}\0{}\0snapshot-v{VERSION}", config.fingerprint).as_bytes(),
        );
        let closing = Arc::new(AtomicBool::new(false));
        let (dir, epoch, messages) = tokio::time::timeout(config.timeout, async {
            loop {
                let attempt = async {
                    let dir = directory::Directory::connect(&redis, &config.directory_prefix, config.timeout).await?;
                    let epoch = dir.epoch().await?;
                    let messages = subscribe(&redis, &dir.events(&scope)).await?;
                    Ok::<_, anyhow::Error>((dir, epoch, messages))
                };
                match attempt.await {
                    Ok(connected) => break connected,
                    Err(e) => {
                        deepseek_ultrafast_logging::redis_failure("kv", "connect");
                        deepseek_ultrafast_logging::diagnostic!("KV Redis directory startup retry: {e:#}");
                    }
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .map_err(|_| {
            deepseek_ultrafast_logging::redis_failure("kv", "startup_timeout");
            anyhow::anyhow!("Redis directory unavailable within startup timeout")
        })?;
        let state = Arc::new(Mutex::new(State {
            watching: true,
            epoch: Some(epoch),
            ..Default::default()
        }));
        let hs = state.clone();
        let hc = config.clone();
        plane.velo.register_handler(
            Handler::typed_unary_async(HANDLER, move |ctx: TypedContext<Fetch>| {
                let (state, config) = (hs.clone(), hc.clone());
                async move { Ok(state.lock().await.serve(ctx.input, &config)) }
            })
            .build(),
        )?;
        let watched = state.clone();
        let own = plane.velo.instance_id();
        let (client, channel, stop) = (redis.clone(), dir.events(&scope), closing.clone());
        let watch = tokio::spawn(async move {
            let mut messages = Some(messages);
            loop {
                if let Some(mut stream) = messages.take() {
                    while let Some(message) = stream.next().await {
                        let Ok(payload) = message.get_payload::<Vec<u8>>() else { continue };
                        if let Ok(event) = serde_json::from_slice::<Onboarded>(&payload) {
                            let owner = event.owner();
                            let mut state = watched.lock().await;
                            if event.peer.instance_id() != own {
                                state.invalidate(&event.key, owner.revision);
                            }
                            let evictions = state.take_evictions();
                            drop(state);
                            evictions.into_iter().for_each(deepseek_ultrafast_logging::emit);
                        }
                    }
                    if stop.load(Ordering::Acquire) { return; }
                    let mut state = watched.lock().await;
                    state.watching = false;
                    state.clear_local();
                    drop(state);
                    deepseek_ultrafast_logging::redis_failure("kv", "ownership_subscription_closed");
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
                match subscribe(&client, &channel).await {
                    Ok(stream) => messages = Some(stream),
                    Err(e) => deepseek_ultrafast_logging::diagnostic!("KV ownership resubscribe failed: {e:#}"),
                }
            }
        });
        let checked = state.clone();
        let (check_dir, scope2) = (dir.clone(), scope.clone());
        let reconcile = tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            loop {
                tick.tick().await;
                let entries = {
                    let mut state = checked.lock().await;
                    state.expire();
                    state
                        .entries
                        .iter()
                        .map(|(k, e)| (k.clone(), e.revision))
                        .collect::<Vec<_>>()
                };
                let reply = check_dir.check(&scope2, &entries).await;
                let mut state = checked.lock().await;
                match reply {
                    Ok((epoch, invalid)) => {
                        state.epoch(epoch);
                        for index in invalid {
                            let (key, revision) = &entries[index];
                            if state
                                .entries
                                .get(key)
                                .is_some_and(|e| e.revision == *revision)
                            {
                                state.remove(key, EvictionReason::OwnershipInvalidated);
                            }
                        }
                        state.watching = true;
                    }
                    Err(error) => {
                        if state.watching {
                            deepseek_ultrafast_logging::diagnostic!(
                                "KV directory reconciliation failed; discarding local cache: {error:#}"
                            );
                        }
                        state.clear_local();
                        state.watching = false;
                    }
                }
                let evictions = state.take_evictions();
                drop(state);
                evictions.into_iter().for_each(deepseek_ultrafast_logging::emit);
            }
        });
        Ok(Self {
            config,
            plane,
            dir,
            scope,
            state,
            fetch_slot: Semaphore::new(1),
            lease_cleanup_slots: Arc::new(Semaphore::new(MAX_LEASES)),
            data_plane_failures: AtomicU64::new(0),
            closing,
            tasks: vec![watch, reconcile],
        })
    }
    pub fn fingerprint(&self) -> &str {
        &self.config.fingerprint
    }

    /// Cumulative count of failed data-plane registrations, GETs, and staging
    /// operations since this replica started its peer cache.
    pub fn data_plane_failures(&self) -> u64 {
        self.data_plane_failures.load(Ordering::Relaxed)
    }

    fn record_data_plane_failure(&self, operation: &'static str) {
        self.data_plane_failures.fetch_add(1, Ordering::Relaxed);
        deepseek_ultrafast_logging::kv_failure(operation);
    }
    #[allow(
        clippy::misnamed_getters,
        reason = "Capture callers require the per-image bound, not LRU capacity"
    )]
    pub fn max_bytes(&self) -> usize {
        self.config.max_snapshot_bytes
    }
    async fn owner(&self, key: &str) -> Result<directory::Owner> {
        let owner = self.dir.lookup(&self.scope, key).await?;
        let mut state = self.state.lock().await;
        state.epoch(owner.revision.epoch);
        Ok(owner)
    }
    pub async fn revalidate(&self, session: &str, prompt: Prompt<'_>, candidate: Candidate) -> Result<Candidate> {
        let key = digest(session.as_bytes());
        ensure!(candidate.key == key, "queued KV candidate belongs to another session");
        let owner = self.owner(&key).await?;
        if owner.peer.as_ref().is_some_and(|p| p.instance_id() == self.plane.velo.instance_id()) {
            return Ok(self.local_candidate(key, owner.revision, prompt).await);
        } else if !candidate.local && owner.peer.is_some() && owner.revision == candidate.revision {
            return Ok(candidate);
        }
        Ok(Candidate {
            revision: owner.revision, snapshot: None, aligned: None, staged: None, local: false,
            ..candidate
        })
    }

    async fn local_candidate(&self, key: String, revision: Revision, prompt: Prompt<'_>) -> Candidate {
        let mut candidate = Candidate {
            revision, snapshot: None, aligned: None, staged: None, local: true,
            key: key.clone(),
        };
        let tokens = prompt.tokens;
        if tokens.len() <= 1 { return candidate; }
        let mut state = self.state.lock().await;
        state.clock += 1;
        let clock = state.clock;
        if let Some(e) = state.entries.get_mut(&key).filter(|e| {
            e.revision.epoch == revision.epoch && e.revision.sequence >= revision.sequence
        }) {
            candidate.revision = e.revision;
            if let Some((stored, aligned)) = e.select_by(tokens.len(), |m| self.reuse(m, prompt)) {
                if !Arc::ptr_eq(stored, &e.stored) {
                    deepseek_ultrafast_logging::diagnostic!("KV input checkpoint reused: key={key} prompt={}", tokens.len());
                }
                candidate.snapshot = Some(stored.snapshot.clone());
                candidate.aligned = aligned;
                candidate.staged = Some(stored.staged.clone());
                e.used = clock;
                e.last_used = Instant::now();
            } else {
                report_prefix_miss(&key, &e.stored.snapshot.manifest, prompt);
            }
        } else {
            deepseek_ultrafast_logging::diagnostic!("KV cache miss: key={key} reason=local_entry_unavailable prompt={}", tokens.len());
        }
        candidate
    }

    pub async fn lookup(&self, session: &str, prompt: Prompt<'_>) -> Result<Candidate> {
        self.lookup_before(session, prompt, Instant::now() + self.config.timeout).await
    }

    /// Resolve and, when possible, retrieve the cached prefix by the caller's
    /// preparation deadline. Donor failures leave the observed ownership fence
    /// available for publication after recomputing the prefix.
    pub async fn lookup_before(
        &self,
        session: &str,
        prompt: Prompt<'_>,
        deadline: Instant,
    ) -> Result<Candidate> {
        let key = digest(session.as_bytes());
        let owner = tokio::time::timeout_at(deadline.into(), self.owner(&key)).await??;
        let tokens = prompt.tokens;
        let mut candidate = Candidate {
            revision: owner.revision,
            snapshot: None,
            aligned: None,
            staged: None,
            local: false,
            key: key.clone(),
        };
        if tokens.len() <= 1 { return Ok(candidate); }
        let Some(peer) = owner.peer else {
            deepseek_ultrafast_logging::diagnostic!("KV cache miss: key={key} reason=no_directory_owner prompt={}", tokens.len());
            return Ok(candidate);
        };
        if peer.instance_id() == self.plane.velo.instance_id() {
            return Ok(tokio::time::timeout_at(deadline.into(), self.local_candidate(key, owner.revision, prompt))
                .await.unwrap_or(candidate));
        }
        let invalidate = async {
            let mut state = self.state.lock().await;
            state.invalidate(&key, owner.revision);
            let evictions = state.take_evictions();
            drop(state);
            evictions.into_iter().for_each(deepseek_ultrafast_logging::emit);
        };
        if tokio::time::timeout_at(deadline.into(), invalidate).await.is_err() {
            return Ok(candidate);
        }

        // Ownerless requests and local hits never wait behind a donor. The remote
        // transfer still has one slot so concurrent preparations fit the same
        // registered-memory and retained-snapshot budget.
        let Ok(Ok(_slot)) = tokio::time::timeout_at(deadline.into(), self.fetch_slot.acquire())
            .await
            .inspect_err(|_| deepseek_ultrafast_logging::kv_failure("fetch_slot_timeout"))
        else {
            return Ok(candidate);
        };
        if let Err(error) = self.fetch_remote(&peer, prompt, &mut candidate, deadline).await {
            deepseek_ultrafast_logging::diagnostic!("KV donor cache miss: key={key} {error:#}");
        }
        Ok(candidate)
    }

    async fn fetch_remote(
        &self,
        peer: &PeerInfo,
        prompt: Prompt<'_>,
        candidate: &mut Candidate,
        deadline: Instant,
    ) -> Result<()> {
        let key = &candidate.key;
        let tokens = prompt.tokens;
        self.plane
            .register_peer(peer.clone())
            .inspect_err(|_| self.record_data_plane_failure("register_peer"))?;
        let mut expected = candidate.revision;
        let mut best: Option<Offer> = None;
        for checkpoint_request in [false, true] {
            // Reserve bounded background cleanup before asking a donor to
            // allocate a lease. If the reply is lost, only its TTL can reap it.
            let Ok(cleanup_slot) = self.lease_cleanup_slots.clone().try_acquire_owned() else {
                deepseek_ultrafast_logging::kv_failure("lease_cleanup_busy");
                break;
            };
            let request = if checkpoint_request {
                Fetch::Checkpoint { key: key.clone(), revision: expected }
            } else {
                Fetch::Open { key: key.clone(), revision: expected }
            };
            let reply = match peer_rpc(&self.plane.velo, peer.instance_id(), request, deadline).await {
                Ok(reply) => reply,
                Err(e) => {
                    deepseek_ultrafast_logging::diagnostic!("KV donor unavailable before transfer: key={key} {e:#}");
                    break;
                }
            };
            let Reply::Open { lease, revision, manifest, handle, checkpoint, checkpoint_tokens } = reply else {
                deepseek_ultrafast_logging::diagnostic!("KV cache miss: key={key} reason=donor_entry_unavailable prompt={}", tokens.len());
                break;
            };
            let lease = DonorLease {
                id: lease,
                peer: peer.instance_id(),
                velo: self.plane.velo.clone(),
                runtime: tokio::runtime::Handle::current(),
                cleanup_slot: Some(cleanup_slot),
            };
            if revision.epoch != expected.epoch || revision.sequence < expected.sequence {
                anyhow::bail!("donor returned a stale/different-epoch revision");
            }
            expected = revision;
            let (common, aligned) = match manifest.validate(&self.config.fingerprint, self.config.max_seq, self.config.max_snapshot_bytes) {
                Ok(()) => self.reuse(&manifest, prompt),
                Err(e) => { deepseek_ultrafast_logging::diagnostic!("KV incompatible candidate: key={key} {e:#}"); (0, None) }
            };
            if common > 0 && best.as_ref().is_none_or(|offer| common > offer.common) {
                if checkpoint_request {
                    deepseek_ultrafast_logging::diagnostic!("KV input checkpoint reused: key={key} prefix={common} prompt={}", tokens.len());
                }
                best = Some(Offer { lease, revision, manifest, handle, common, aligned });
            } else {
                if common == 0 && best.is_none() && !checkpoint { report_prefix_miss(key, &manifest, prompt); }
            }
            let limit = checkpoint_tokens.unwrap_or(usize::MAX).min(tokens.len() - 1);
            if !checkpoint || best.as_ref().is_some_and(|offer| offer.common >= limit) { break; }
        }
        if let Some(Offer { lease, revision, manifest, handle, aligned, .. }) = best {
            let result = self.receive(
                key,
                peer,
                *manifest,
                handle,
                aligned.as_ref().map_or(prompt, Aligned::prompt),
                deadline,
            ).await;
            drop(lease);
            if let Some((snapshot, staged)) = result? {
                candidate.revision = revision;
                candidate.snapshot = Some(snapshot);
                candidate.aligned = aligned;
                candidate.staged = Some(staged);
            }
        }
        Ok(())
    }
    fn reuse(&self, manifest: &Manifest, prompt: Prompt) -> (usize, Option<Aligned>) {
        let exact = manifest.reusable_prefix(prompt);
        if let Some(text) = &self.config.token_text
            && let Some(aligned) = align(&manifest.tokens, prompt, text) {
            let common = manifest.reusable_prefix(aligned.prompt());
            if common > exact {
                return (common, Some(aligned));
            }
        }
        (exact, None)
    }
    async fn receive(
        &self,
        key: &str,
        peer: &PeerInfo,
        manifest: Manifest,
        handle: DataHandle,
        prompt: Prompt<'_>,
        deadline: Instant,
    ) -> Result<Option<(Arc<Snapshot>, Arc<Staged>)>> {
        if let Err(e) = manifest.validate(
            &self.config.fingerprint,
            self.config.max_seq,
            self.config.max_snapshot_bytes,
        ) {
            deepseek_ultrafast_logging::diagnostic!("KV incompatible candidate: key={key} {e:#}");
            return Ok(None);
        }
        if manifest.reusable_prefix(prompt) == 0 {
            report_prefix_miss(key, &manifest, prompt);
            return Ok(None);
        }
        if handle.unpack().0 != peer.worker_id() {
            self.record_data_plane_failure("donor_handle");
            return Err(RdmaRequired("donor handle names another worker".into()).into());
        }
        ensure!(Instant::now() < deadline, "KV peer deadline elapsed");
        let data = tokio::time::timeout_at(
            deadline.into(),
            self.plane.receive(handle, manifest.bytes),
        )
        .await
        .inspect_err(|_| self.record_data_plane_failure("get_timeout"))?
        .inspect_err(|_| self.record_data_plane_failure("get"))?;
        let snapshot = Arc::new(Snapshot { manifest, data });
        snapshot
            .validate(
                &self.config.fingerprint,
                self.config.max_seq,
                self.config.max_snapshot_bytes,
            )
            .inspect_err(|_| deepseek_ultrafast_logging::kv_failure("snapshot_invalid"))?;
        ensure!(Instant::now() < deadline, "KV peer deadline elapsed");
        let staged = tokio::time::timeout_at(deadline.into(), self.plane.stage(&snapshot.data))
            .await
            .inspect_err(|_| self.record_data_plane_failure("registration_timeout"))?
            .inspect_err(|_| self.record_data_plane_failure("registration"))?;
        Ok(Some((snapshot, staged)))
    }
    pub async fn onboard_named(&self, session: &str, candidate: &Candidate, session_id: Option<Arc<str>>) -> Result<Revision> {
        let stored = Arc::new(Stored {
            snapshot: candidate
                .snapshot
                .clone()
                .ok_or_else(|| anyhow::anyhow!("cannot onboard a miss"))?,
            staged: candidate
                .staged
                .clone()
                .ok_or_else(|| anyhow::anyhow!("candidate is not RDMA staged"))?,
        });
        self.announce(session, candidate.revision, stored, None, session_id).await
    }
    pub async fn publish(
        &self,
        session: &str,
        revision: Revision,
        snapshot: Arc<Snapshot>,
    ) -> Result<Revision> {
        let input_tokens = snapshot.manifest.tokens.len();
        self.publish_with_input(session, revision, snapshot, input_tokens).await
    }
    pub async fn publish_with_input(
        &self,
        session: &str,
        revision: Revision,
        snapshot: Arc<Snapshot>,
        input_tokens: usize,
    ) -> Result<Revision> {
        self.publish_with_input_named(session, revision, snapshot, input_tokens, None).await
    }
    pub async fn publish_with_input_named(
        &self, session: &str, revision: Revision, snapshot: Arc<Snapshot>,
        input_tokens: usize, session_id: Option<Arc<str>>,
    ) -> Result<Revision> {
        ensure!(input_tokens > 0 && input_tokens <= snapshot.manifest.tokens.len(),
                "invalid KV publication input boundary");
        snapshot.validate(
            &self.config.fingerprint,
            self.config.max_seq,
            self.config.max_snapshot_bytes,
        )?;
        let staged = self.plane.stage(&snapshot.data).await
            .inspect_err(|_| self.record_data_plane_failure("registration"))?;
        self.announce(session, revision, Arc::new(Stored { snapshot, staged }), Some(input_tokens), session_id)
            .await
    }
    async fn announce(
        &self,
        session: &str,
        revision: Revision,
        stored: Arc<Stored>,
        input_tokens: Option<usize>,
        session_id: Option<Arc<str>>,
    ) -> Result<Revision> {
        let key = digest(session.as_bytes());
        let ticket = {
            let mut state = self.state.lock().await;
            ensure!(state.watching, "KV ownership subscriber/directory unavailable");
            ensure!(state.epoch == Some(revision.epoch), "local directory epoch advanced");
            ensure!(
                state.entries.get(&key).is_none_or(|e| e.revision.epoch == revision.epoch
                    && e.revision.sequence <= revision.sequence),
                "local ownership generation advanced"
            );
            state.start_claim(&key)
        };
        let result = self.dir.claim(&self.scope, &key, revision, &self.plane.velo.peer_info()).await;
        // Observe invalidations that landed during the CAS. Do not hold the
        // cache state lock across Redis I/O, including reconnects and timeouts.
        let mut state = self.state.lock().await;
        let invalidated_through = state.finish_claim(&key, &ticket);
        let outcome = match result {
            Ok(owner) => {
                ensure!(
                    state.watching && state.epoch == Some(owner.revision.epoch)
                        && invalidated_through.is_some_and(|seq| seq <= owner.revision.sequence)
                        && state.entries.get(&key).is_none_or(|e| e.revision.sequence <= owner.revision.sequence),
                    "ownership advanced during KV publication"
                );
                state.install_named(key, owner.revision, stored, self.config.max_bytes,
                    input_tokens.map(|input| (revision, input)), session_id);
                Ok(owner.revision)
            }
            result => {
                if state
                    .entries
                    .get(&key)
                    .is_some_and(|e| e.revision == revision)
                {
                    state.remove(&key, EvictionReason::OwnershipInvalidated);
                }
                result.map(|owner| owner.revision)
            }
        };
        let evictions = state.take_evictions();
        drop(state);
        evictions.into_iter().for_each(deepseek_ultrafast_logging::emit);
        if let Err(error) = &outcome { directory::log_failure(error); }
        outcome
    }
    pub async fn shutdown(&self) {
        self.closing.store(true, Ordering::Release);
        for task in &self.tasks {
            task.abort();
        }
        {
            let mut state = self.state.lock().await;
            state.watching = false;
            state.clear_local();
            state.leases.clear();
        }
        self.plane.shutdown().await;
    }
}
impl Drop for PeerCache {
    fn drop(&mut self) {
        self.closing.store(true, Ordering::Release);
        for task in &self.tasks {
            task.abort();
        }
    }
}
