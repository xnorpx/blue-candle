use crate::{
    api::{
        Prediction, StatusUpdateResponse, VersionInfo, VisionCustomListResponse,
        VisionDetectionRequest, VisionDetectionResponse,
    },
    coco_classes,
    detector::{Bbox, Detector, InferenceTime, KeyPoint, ProcessingTime},
    server_stats::ServerStats,
    utils::{img_with_bbox, save_image},
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
use image::ImageReader;
use std::{
    io::Cursor,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Instant,
};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn, Level};
use uuid::Uuid;

const MEGABYTE: usize = 1024 * 1024; // 1 MB = 1024 * 1024 bytes
const THIRTY_MEGABYTES: usize = 30 * MEGABYTE; // 30 MB in bytes

struct ServerState {
    detector: Detector,
    stats: Mutex<ServerStats>,
}

pub async fn run_server(
    port: u16,
    detector: Detector,
    cancellation_token: CancellationToken,
) -> anyhow::Result<()> {
    let (_, inference_time, processing_time) = detector.test_detection()?;
    info!(
        "Server inference startup test, processing time: {:#?}, inference time: {:#?}",
        processing_time, inference_time
    );

    let server_state = Arc::new(ServerState {
        detector,
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

    let image_data = vision_request.image_data.clone();
    let state2 = server_state.detector.clone();
    // Detection will be slow, (100ms+) so we spawn a blocking task.
    let (predictions, inference_time, processing_time) = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(Vec<Prediction>, InferenceTime, ProcessingTime)> {
            let (bboxes, inference_time, processing_time) = state2.detect(image_data.as_ref())?;

            let predictions = from_bbox_to_predictions(
                bboxes,
                vision_request.min_confidence,
                &coco_classes::NAMES,
                state2.labels(),
            );

            Ok((predictions, inference_time, processing_time))
        },
    )
    .await??;

    if !predictions.is_empty() {
        if let Some(image_path) = server_state.detector.image_path() {
            let reader = ImageReader::new(Cursor::new(vision_request.image_data.as_ref()))
                .with_guessed_format()
                .expect("Cursor io never fails");
            let img = img_with_bbox(predictions.clone(), reader, 15)?;
            // The api doesn't provide a source id or a source name so we just generate a uuid here.
            let picture_name = format!("{}/{}.jpg", image_path, Uuid::new_v4());
            save_image(img, picture_name, "-od").await?;
        }
    }

    let request_time = Instant::now().duration_since(request_start_time);
    let count = predictions.len() as i32;

    let image = vision_request.image_name.split('.').next().unwrap_or("");

    debug!(
        "Image: {}, request time {:#?}, processing time: {:#?}, inference time: {:#?}",
        image, request_time, processing_time, inference_time
    );

    {
        let mut stats = server_state.stats.lock().await;
        stats.calculate_and_log_stats(
            request_start_time,
            request_time,
            processing_time,
            inference_time,
        );
    }

    let response = VisionDetectionResponse {
        success: true,
        message: "".into(),
        error: None,
        predictions,
        count,
        command: "detect".into(),
        module_id: "Yolo8".into(),
        execution_provider: if server_state.detector.is_gpu() {
            "GPU".to_string()
        } else {
            "CPU".to_string()
        },
        can_useGPU: cuda_is_available(),
        inference_ms: inference_time.as_millis() as i32,
        process_ms: processing_time.as_millis() as i32,
        analysis_round_trip_ms: request_time.as_millis() as i32,
    };
    Ok(Json(response))
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

pub fn from_bbox_to_predictions(
    bboxes: (Vec<Vec<Bbox<Vec<KeyPoint>>>>, f32, f32),
    confidence_threshold: f32,
    class_names: &[&str],
    labels: &[String],
) -> Vec<Prediction> {
    let mut predictions: Vec<Prediction> = Vec::new();
    let w_ratio = bboxes.1;
    let h_ratio = bboxes.2;

    for (class_index, class_bboxes) in bboxes.0.iter().enumerate() {
        if class_bboxes.is_empty() {
            continue;
        }

        let class_name = class_names
            .get(class_index)
            .unwrap_or(&"Unknown")
            .to_string();

        if !labels.is_empty() && !labels.contains(&class_name) {
            continue;
        }

        for bbox in class_bboxes.iter() {
            if bbox.confidence > confidence_threshold {
                let prediction = Prediction {
                    x_max: ((bbox.xmax * w_ratio) as usize),
                    x_min: ((bbox.xmin * w_ratio) as usize),
                    y_max: ((bbox.ymax * h_ratio) as usize),
                    y_min: ((bbox.ymin * h_ratio) as usize),
                    confidence: bbox.confidence.clamp(0.0, 1.0),
                    label: class_name.clone(),
                };
                debug!("Object detected: {:#?}", prediction);
                predictions.push(prediction);
            }
        }
    }

    predictions
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
