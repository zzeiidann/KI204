use std::sync::Arc;

use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use quantized_llm_service::{AppConfig, ModelRegistry, build_router};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let config = Arc::new(AppConfig::from_env()?);
    tracing::info!(?config.listen_addr, "loading model artifacts");

    let registry = Arc::new(ModelRegistry::initialize(config.as_ref())?);
    let router = build_router(config.clone(), registry);

    let listener = TcpListener::bind(config.listen_addr).await?;
    let addr = listener.local_addr()?;
    tracing::info!(%addr, "REST server ready");

    axum::serve(listener, router).await?;

    Ok(())
}

fn init_tracing() {
    if tracing::dispatcher::has_been_set() {
        return;
    }
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,hyper=warn,axum::rejection=trace".into());
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_target(false)
        .compact();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();
}
