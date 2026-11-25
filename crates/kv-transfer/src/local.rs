//! Single-host KV storage. This build has no peer-transfer transport.
//!
//! Keep upstream snapshot validation, Redis generation fences, LRU ownership,
//! and publication unchanged. Velo supplies local handle identity/storage;
//! all network listeners bind loopback and remote peer admission is refused.
use anyhow::{Result, ensure};
use bytes::Bytes;
use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};
use velo::{DataHandle, PeerInfo, ShutdownPolicy, Velo};

#[derive(Debug)]
pub struct RdmaRequired(pub String);
impl std::fmt::Display for RdmaRequired {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "single-host KV transport refused: {}", self.0)
    }
}
impl std::error::Error for RdmaRequired {}
pub fn is_rdma_required(error: &anyhow::Error) -> bool {
    error.downcast_ref::<RdmaRequired>().is_some()
}
pub(crate) fn required(error: impl std::fmt::Display) -> anyhow::Error {
    RdmaRequired(error.to_string()).into()
}

pub(crate) struct Staged {
    pub handle: DataHandle,
    store: Arc<velo::rendezvous::store::DataStore>,
}
impl Drop for Staged {
    fn drop(&mut self) {
        let id = self.handle.unpack().1;
        self.store.ref_decrement(id);
        self.store.try_free(id);
    }
}

pub(crate) struct DataPlane {
    pub velo: Arc<Velo>,
}
impl DataPlane {
    pub async fn start(_bind: SocketAddr, budget: u64, _timeout: Duration) -> Result<Self> {
        ensure!(budget > 0, "local KV budget must be positive");
        let velo = Velo::builder()
            .stream_bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
            .build()
            .await?;
        Ok(Self { velo })
    }
    pub fn register_peer(&self, _peer: PeerInfo) -> Result<()> {
        Err(required("remote cache owners are not supported"))
    }
    pub async fn stage(&self, data: &[u8]) -> Result<Arc<Staged>> {
        Ok(Arc::new(Staged {
            handle: self.velo.register_data(Bytes::copy_from_slice(data)),
            store: self.velo.rendezvous_manager().data_store().clone(),
        }))
    }
    pub async fn receive(&self, _handle: DataHandle, _len: usize) -> Result<Bytes> {
        Err(required("remote cache reads are not supported"))
    }
    pub async fn shutdown(&self) {
        self.velo
            .shutdown(ShutdownPolicy::Timeout(Duration::from_secs(5)))
            .await;
    }
}

pub async fn check_rdma(bind: SocketAddr, budget: u64, timeout: Duration) -> Result<()> {
    let plane = DataPlane::start(bind, budget, timeout).await?;
    let staged = plane.stage(b"local-kv-check").await?;
    let id = staged.handle.unpack().1;
    let store = staged.store.clone();
    ensure!(
        store.metadata(id).is_some(),
        "local KV registration missing"
    );
    ensure!(
        plane.register_peer(plane.velo.peer_info()).is_err(),
        "remote peers accepted"
    );
    ensure!(
        plane.receive(staged.handle, 14).await.is_err(),
        "remote reads accepted"
    );
    drop(staged);
    ensure!(store.metadata(id).is_none(), "local KV storage leaked");
    plane.shutdown().await;
    println!("KV local ready: storage/release passed; remote transfers refused");
    Ok(())
}

#[cfg(test)]
pub(crate) fn heap_stage(data: Bytes) -> Arc<Staged> {
    let store = Arc::new(velo::rendezvous::store::DataStore::new());
    let id = store.register(data, None);
    Arc::new(Staged {
        handle: DataHandle::pack(velo::InstanceId::new_v4().worker_id(), id),
        store,
    })
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn local_storage_lifecycle_refuses_peer_transfer() {
        super::check_rdma(
            "127.0.0.1:0".parse().unwrap(),
            64 << 20,
            std::time::Duration::from_secs(5),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn local_peer_cache_preserves_payload_and_generation() {
        use crate::{Config, PeerCache, Prompt, Snapshot};
        let redis_url = std::env::var("KILN_TEST_REDIS_URL").expect("private test Redis required");
        let cache = PeerCache::start(Config {
            redis_url,
            directory_prefix: "local-kv-build-test".into(),
            fingerprint: "test".into(),
            bind: "127.0.0.1:0".parse().unwrap(),
            max_bytes: 1024,
            max_snapshot_bytes: 1024,
            rdma_bytes: 1024,
            max_seq: 16,
            timeout: std::time::Duration::from_secs(5),
            token_text: None,
        })
        .await
        .unwrap();
        let prompt = [1, 2, 3];
        let miss = cache
            .lookup("session", Prompt::text(&prompt))
            .await
            .unwrap();
        assert!(miss.snapshot.is_none());
        let snapshot = std::sync::Arc::new(Snapshot::new(
            "test".into(),
            vec![1, 2, 3],
            0,
            [0, 3, 3],
            vec![8],
            bytes::Bytes::from_static(b"payload!"),
        ));
        let revision = cache
            .publish("session", miss.revision, snapshot)
            .await
            .unwrap();
        let hit = cache
            .lookup("session", Prompt::text(&prompt))
            .await
            .unwrap();
        assert!(hit.local);
        assert_eq!(hit.revision, revision);
        assert_eq!(&hit.snapshot.as_ref().unwrap().data[..], b"payload!");
        let hit = cache
            .revalidate("session", Prompt::text(&prompt), hit)
            .await
            .unwrap();
        assert!(hit.snapshot.is_some());
        assert!(
            cache
                .publish(
                    "session",
                    miss.revision,
                    std::sync::Arc::new(Snapshot::new(
                        "test".into(),
                        vec![1, 2, 3],
                        0,
                        [0, 3, 3],
                        vec![8],
                        bytes::Bytes::from_static(b"stale!!!")
                    ))
                )
                .await
                .is_err()
        );
        let invalid = std::sync::Arc::new(Snapshot::new(
            "wrong-profile".into(),
            vec![1, 2, 3],
            0,
            [0, 3, 3],
            vec![8],
            bytes::Bytes::from_static(b"invalid!"),
        ));
        assert!(cache.publish("session", revision, invalid).await.is_err());
        let retained = cache
            .lookup("session", Prompt::text(&prompt))
            .await
            .unwrap();
        assert_eq!(retained.revision, revision);
        assert_eq!(&retained.snapshot.unwrap().data[..], b"payload!");
        cache.shutdown().await;
    }
}
