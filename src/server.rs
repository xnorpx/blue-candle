use crate::{
    api::{
        StatusUpdateResponse, VersionInfo, VisionCustomListResponse, VisionDetectionRequest,
        VisionDetectionResponse,
    },
    detector::DetectorConfig,
    server_stats::ServerStats,
    worker::DetectorDispatcher,
};
use axum::{
    body::{self, Body},
    extract::{DefaultBodyLimit, Multipart, State},
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use candle::utils::cuda_is_available;
use candle_core as candle;
use chrono::Utc;
use clap::ValueEnum;
use std::{
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Instant,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn, Level};

const MEGABYTE: usize = 1024 * 1024; // 1 MB = 1024 * 1024 bytes
const THIRTY_MEGABYTES: usize = 30 * MEGABYTE; // 30 MB in bytes

struct ServerState {
    detector_dispatcher: Arc<DetectorDispatcher>,
    stats: Mutex<ServerStats>,
}

pub async fn run_server(
    port: u16,
    blocking_threads: usize,
    detector_config: DetectorConfig,
    cancellation_token: CancellationToken,
) -> anyhow::Result<()> {
    // let (_, inference_time, processing_time) = detector.test_detection()?;
    // info!(
    //     "Server inference startup test, processing time: {:#?}, inference time: {:#?}",
    //     processing_time, inference_time
    // );

    let detector_dispatcher = DetectorDispatcher::new(blocking_threads, detector_config.clone())?;

    let server_state = Arc::new(ServerState {
        detector_dispatcher,
        stats: Mutex::new(ServerStats::default()),
    });

    let blue_candle = Router::new()
        .route(
            "/",
            get(|| async { (StatusCode::OK, "Blue Candle is alive and healthy") }),
        )
        .route(
            "/v1/status/updateavailable",
            get(v1_status_update_available),
        )
        .route("/v1/vision/detection", post(v1_vision_detection))
        .with_state(server_state.clone())
        .route("/v1/vision/custom/list", post(v1_vision_custom_list))
        .fallback(fallback_handler)
        .layer(DefaultBodyLimit::max(THIRTY_MEGABYTES));

    let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), port);
    info!("Starting server, listening on {}", addr);
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            error!("Looks like {port} is already in use either by Blue candle, CPAI or another application, please turn off the other application or pick another port with --port");
            return Err(e.into());
        }
        Err(e) => return Err(e.into()),
    };

    axum::serve(listener, blue_candle.into_make_service())
        .with_graceful_shutdown(async move {
            cancellation_token.cancelled().await;
        })
        .await?;

    Ok(())
}

async fn v1_vision_detection(
    State(server_state): State<Arc<ServerState>>,
    mut multipart: Multipart, // Note multipart needs to be last
) -> Result<Json<VisionDetectionResponse>, BlueCandleError> {
    let request_start_time = Instant::now();
    let mut vision_request = VisionDetectionRequest::default();

    while let Some(field) = multipart.next_field().await? {
        match field.name() {
            Some("min_confidence") => {
                vision_request.min_confidence = field.text().await?.parse::<f32>()?;
            }
            Some("image") => {
                if let Some(image_name) = field.file_name().map(|s| s.to_string()) {
                    vision_request.image_name = image_name;
                }
                vision_request.image_data = field.bytes().await?;
            }
            Some(&_) => {}
            None => {}
        }
    }

    let (sender, receiver) = tokio::sync::oneshot::channel();
    server_state
        .detector_dispatcher
        .dispatch_work(vision_request, sender);
    let mut vision_response = receiver.await?;

    vision_response.analysis_round_trip_ms = request_start_time.elapsed().as_millis() as i32;
    Ok(Json(vision_response))
}

async fn v1_status_update_available() -> Result<Json<StatusUpdateResponse>, BlueCandleError> {
    let response = StatusUpdateResponse {
        success: true,
        message: "".to_string(),
        version: None,
        current: VersionInfo::default(),
        latest: VersionInfo::default(),
        updateAvailable: false, // Always respond that no update is available
    };

    Ok(Json(response))
}

async fn v1_vision_custom_list() -> Result<Json<VisionCustomListResponse>, BlueCandleError> {
    let response = VisionCustomListResponse {
        success: true,
        models: vec![],
        moduleId: "".to_string(),
        moduleName: "".to_string(),
        command: "list".to_string(),
        statusData: None,
        inferenceDevice: "CPU".to_string(),
        analysisRoundTripMs: 0,
        processedBy: "BlueCandle".to_string(),
        timestampUTC: Utc::now().to_rfc3339(),
    };

    Ok(Json(response))
}

struct BlueCandleError(anyhow::Error);

impl IntoResponse for BlueCandleError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(VisionDetectionResponse {
                success: false,
                message: "".into(),
                error: Some(self.0.to_string()),
                predictions: vec![],
                count: 0,
                command: "".into(),
                module_id: "".into(),
                execution_provider: "".into(),
                can_useGPU: cuda_is_available(),
                inference_ms: 0_i32,
                process_ms: 0_i32,
                analysis_round_trip_ms: 0_i32,
            }),
        )
            .into_response()
    }
}

async fn fallback_handler(req: Request<Body>) -> impl IntoResponse {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let headers = req.headers().clone();

    let body_bytes = body::to_bytes(req.into_body(), usize::MAX)
        .await
        .unwrap_or_else(|_| body::Bytes::new());

    warn!(
        "Unimplemented endpoint called: Method: {}, URI: {}, Headers: {:?}, Body: {:?}",
        method, uri, headers, body_bytes
    );

    (StatusCode::NOT_FOUND, "Endpoint not implemented")
}

impl<E> From<E> for BlueCandleError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for Level {
    fn from(value: LogLevel) -> Self {
        match value {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }
}
