use axum::{
    extract::Multipart,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use blue_candle::{
    api::{Prediction, VisionDetectionRequest, VisionDetectionResponse},
    coco_classes,
    detector::{Bbox, Detector, InferenceTime, KeyPoint, ProcessingTime, BIKE_IMAGE_BYTES},
    utils::{download_models, ensure_directory_exists, img_with_bbox, read_jpeg_file, save_image},
};
use candle::utils::cuda_is_available;
use candle_core as candle;
use clap::Parser;
use image::io::Reader;
use std::{
    io::Cursor,
    net::{Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Instant,
};
use tracing::{debug, info};
use uuid::Uuid;

const MEGABYTE: usize = 1024 * 1024; // 1 MB = 1024 * 1024 bytes
const THIRTY_MEGABYTES: usize = 30 * MEGABYTE; // 30 MB in bytes

#[derive(Parser, Debug, Clone)]
#[command(author = "Marcus Asteborg", version=env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// Filters the results to include only the specified labels. Provide labels separated by spaces.
    /// Example: --labels "person,cup"
    #[arg(long, value_delimiter = ' ', num_args = 1..)]
    pub labels: Vec<String>,

    /// The port on which the server will listen for HTTP requests.
    /// Default is 32168. Example usage: --port 8080
    //#[arg(long, default_value_t = 32168)]
    #[arg(long, default_value_t = 32168)]
    pub port: u16,

    /// Forces the application to use the CPU for computations, even if a GPU is available.
    /// Useful to compare inference time between CPU and GPU. Use --cpu to enable.
    #[arg(long)]
    pub cpu: bool,

    /// Sets the confidence threshold for model predictions.
    /// Only detections with a confidence level higher than this threshold will be considered.
    /// This will override min_confidence from API calls.
    /// Default is 0.25. Example: --confidence-threshold 0.5
    #[arg(long, default_value_t = 0.25)]
    pub confidence_threshold: f32,

    /// The threshold for non-maximum suppression, used to filter out overlapping bounding boxes.
    /// Higher values can reduce the number of overlapping boxes. Default is 0.45.
    /// Example: --nms-threshold 0.5
    #[arg(long, default_value_t = 0.45)]
    pub nms_threshold: f32,

    /// Font size for the legend in saved images. Set to 0 to disable the legend.
    /// Default is 14. Example: --legend_size 16
    #[arg(long, default_value_t = 14)]
    pub legend_size: u32,

    /// This will download the other yolo8 models into model path directory
    /// Example: --model-path "./models"
    #[arg(long)]
    pub model_path: Option<String>,

    /// Path to the model weights file (in safetensors format).
    /// If not provided, the default model 'yolov8n.safetensors' will be used.
    /// Example: --model "/path/to/model.safetensors"
    #[arg(long)]
    pub model: Option<String>,

    /// Path to save images that have been processed with detected objects.
    /// If not specified, images will not be saved. Example: --image_path "/path/to/save/images"
    #[arg(long)]
    pub image_path: Option<String>,

    /// Path to a test image. The application will run object detection on this image
    /// and save a copy with the suffix '-od'. Only this image will be processed.
    /// Example: --image "/path/to/test.jpg"
    #[arg(long)]
    pub image: Option<String>,

    /// The application will run object detection on this image on the built in
    /// and save a copy with the suffix '-od'. Only this image will be processed.
    /// Example: --image "/path/to/test.jpg"
    #[arg(long)]
    pub test: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    setup_ansi_support();

    let args = Args::parse();

    // TODO(xnorpx): make this configurable
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    ensure_directory_exists(args.image_path.clone()).await?;

    let detector = Detector::new(
        args.cpu,
        args.model.clone(),
        args.confidence_threshold,
        args.nms_threshold,
        args.labels.clone(),
        args.image_path.clone(),
    )?;

    if let Some(model_path) = args.model_path {
        download_models(model_path).await?;
        return Ok(());
    }
    if args.test {
        return test(detector, args).await;
    }

    match args.image.clone() {
        None => run_server(args, detector).await?,
        Some(image) => test_image(image, args, detector).await?,
    };

    Ok(())
}

async fn run_server(args: Args, detector: Detector) -> anyhow::Result<()> {
    let (_, inference_time, processing_time) = detector.test_detection()?;
    info!(
        "Server inference startup test, processing time: {:#?}, inference time: {:#?}",
        processing_time, inference_time
    );

    let detector = Arc::new(detector);

    let blue_candle = Router::new()
        .route("/v1/vision/detection", post(v1_vision_detection))
        .with_state(detector.clone())
        .layer(DefaultBodyLimit::max(THIRTY_MEGABYTES));

    let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), args.port);
    info!("Starting server, listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, blue_candle.into_make_service()).await?;

    Ok(())
}

async fn v1_vision_detection(
    State(detector): State<Arc<Detector>>,
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
    let state2 = detector.clone();
    // Detection will be slow, (100ms+) so we spawn a blocking task.
    let (predictions, inference_time, processing_time) = tokio::task::spawn_blocking(
        move || -> anyhow::Result<(Vec<Prediction>, InferenceTime, ProcessingTime)> {
            let (bboxes, inference_time, processing_time) =
                state2.detect(image_data.as_ref()).unwrap();

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
        if let Some(image_path) = detector.image_path() {
            let reader = Reader::new(Cursor::new(vision_request.image_data.as_ref()))
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

    info!(
        "Image: {}, request time {:#?}, processing time: {:#?}, inference time: {:#?}",
        image, request_time, processing_time, inference_time
    );

    let response = VisionDetectionResponse {
        success: true,
        message: "".into(),
        error: None,
        predictions,
        count,
        command: "detect".into(),
        module_id: "Yolo8".into(),
        execution_provider: if detector.is_gpu() {
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
    labels: &Vec<String>,
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

async fn test_image(image: String, args: Args, detector: Detector) -> anyhow::Result<()> {
    let start_test_time = Instant::now();
    let contents = read_jpeg_file(image.clone()).await?;

    let (bboxes, inference_time, processing_time) = detector.detect(&contents)?;

    let predictions =
        from_bbox_to_predictions(bboxes, 0.5, &coco_classes::NAMES, detector.labels());

    let reader = Reader::new(Cursor::new(contents.as_ref()))
        .with_guessed_format()
        .expect("Cursor io never fails");
    let img = img_with_bbox(predictions, reader, args.legend_size)?;

    save_image(img, image, "-od").await?;
    let test_time = Instant::now().duration_since(start_test_time);

    info!(
        "Tested image in {:#?}, processing time: {:#?}, inference time: {:#?}",
        test_time, processing_time, inference_time
    );

    Ok(())
}

async fn test(detector: Detector, args: Args) -> anyhow::Result<()> {
    let start_test_time = Instant::now();
    let (bboxes, inference_time, processing_time) = detector.test_detection()?;
    let predictions = from_bbox_to_predictions(
        bboxes,
        args.confidence_threshold,
        &coco_classes::NAMES,
        detector.labels(),
    );
    let reader = Reader::new(Cursor::new(BIKE_IMAGE_BYTES))
        .with_guessed_format()
        .expect("Cursor io never fails");
    let img = img_with_bbox(predictions.clone(), reader, 30)?;
    // The api doesn't provide a source id or a source name so we just generate a uuid here.
    save_image(img, "test.jpg".into(), "").await?;
    let test_time = Instant::now().duration_since(start_test_time);

    info!(
        "Tested image in {:#?}, processing time: {:#?}, inference time: {:#?}",
        test_time, processing_time, inference_time
    );
    Ok(())
}

fn setup_ansi_support() {
    #[cfg(target_os = "windows")]
    if let Err(e) = ansi_term::enable_ansi_support() {
        eprintln!("Failed to enable ANSI support: {}", e);
    }
}
