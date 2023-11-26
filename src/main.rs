use crate::detector::Detector;
use axum::{
    body::Bytes,
    extract::Multipart,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use candle::utils::cuda_is_available;
use candle_core as candle;
use candle_examples::coco_classes;
use candle_transformers::object_detection::{Bbox, KeyPoint};
use clap::Parser;
use detector::BIKE_IMAGE_BYTES;
use image::{codecs::jpeg::JpegEncoder, io::Reader, DynamicImage, ImageFormat};
use serde::Serialize;
use std::{
    fs,
    io::Cursor,
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::{fs::File, io::AsyncReadExt, io::AsyncWriteExt};
use tracing::{debug, info};
use uuid::Uuid;

mod detector;
mod model;

#[derive(Parser, Debug, Clone)]
#[command(author = "Marcus Asteborg", version=env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// Filters the results to include only the specified labels. Provide labels separated by spaces.
    /// Example: --labels "person,cup"
    #[arg(long, value_delimiter = ' ', num_args = 1..)]
    pub labels: Vec<String>,

    /// The port on which the server will listen for HTTP requests.
    /// Default is 32168. Example usage: --port 8080
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
    let args = Args::parse();

    // TODO(xnorpx): make this configurable
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    ensure_directory_exists(args.image_path.clone())?;

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
    detector.test_detection()?;

    let detector = Arc::new(detector);

    let blue_candle = Router::new()
        .route("/v1/vision/detection", post(v1_vision_detection))
        .with_state(detector.clone());

    let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), args.port);
    info!("Starting server, listening on {}", addr);
    axum::Server::bind(&addr)
        .serve(blue_candle.into_make_service())
        .await?;

    Ok(())
}

async fn v1_vision_detection(
    State(state): State<Arc<Detector>>,
    mut multipart: Multipart, // Note multipart needs to be last
) -> Result<Json<VisionDetectionResponse>, BlueCandleError> {
    let process_start = Instant::now();
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

    let image_ref = vision_request.image_data.clone();
    let state2 = state.clone();
    // Detection will be slow, (100ms+) so we spawn a blocking task.
    let predictions = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<Prediction>> {
        let reader = Reader::new(Cursor::new(image_ref.as_ref()))
            .with_guessed_format()
            .expect("Cursor io never fails");
        let bboxes = state2.detect(reader)?;

        let predictions = from_bbox_to_predictions(
            bboxes,
            vision_request.min_confidence,
            &coco_classes::NAMES,
            state2.labels(),
        );
        Ok(predictions)
    })
    .await??;

    if !predictions.is_empty() {
        if let Some(image_path) = state.image_path() {
            let reader = Reader::new(Cursor::new(vision_request.image_data.as_ref()))
                .with_guessed_format()
                .expect("Cursor io never fails");
            let img = img_with_bbox(predictions.clone(), reader, 15)?;
            // The api doesn't provide a source id or a source name so we just generate a uuid here.
            let picture_name = format!("{}/{}.jpg", image_path, Uuid::new_v4());
            save_image(img, picture_name, "-od").await?;
        }
    }

    let process_time = Instant::now().duration_since(process_start);

    let count = predictions.len() as i32;

    let response = VisionDetectionResponse {
        success: true,
        message: "".into(),
        error: None,
        predictions,
        count,
        command: "detect".into(),
        module_id: "Yolo8".into(),
        execution_provider: "TODO".into(),
        can_useGPU: cuda_is_available(),
        // TODO(xnorpx): measure different times
        inference_ms: process_time.as_millis() as i32,
        process_ms: process_time.as_millis() as i32,
        analysis_round_trip_ms: process_time.as_millis() as i32,
    };
    Ok(Json(response))
}

#[derive(Default)]
struct VisionDetectionRequest {
    pub min_confidence: f32,
    pub image_data: Bytes,
    pub image_name: String,
}

#[allow(non_snake_case)]
#[derive(Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct VisionDetectionResponse {
    /// True if successful.
    success: bool,
    /// A summary of the inference operation.
    message: String,
    /// An description of the error if success was false.
    error: Option<String>,
    /// An array of objects with the x_max, x_min, max, y_min, label and confidence.
    predictions: Vec<Prediction>,
    /// The number of objects found.
    count: i32,
    /// The command that was sent as part of this request. Can be detect, list, status.
    command: String,
    /// The Id of the module that processed this request.
    module_id: String,
    /// The name of the device or package handling the inference. eg CPU, GPU
    execution_provider: String,
    /// True if this module can use the current GPU if one is present.
    can_useGPU: bool,
    // The time (ms) to perform the AI inference.
    inference_ms: i32,
    // The time (ms) to process the image (includes inference and image manipulation operations).
    process_ms: i32,
    // The time (ms) for the round trip to the analysis module and back.
    analysis_round_trip_ms: i32,
}

#[derive(Serialize, Debug, Clone)]
pub struct Prediction {
    x_max: usize,
    x_min: usize,
    y_max: usize,
    y_min: usize,
    confidence: f32,
    label: String,
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

fn img_with_bbox(
    predictions: Vec<Prediction>,
    original_img: Reader<Cursor<&[u8]>>,
    legend_size: u32,
) -> anyhow::Result<DynamicImage> {
    assert_eq!(original_img.format(), Some(ImageFormat::Jpeg));
    let original_image = original_img.decode().map_err(candle::Error::wrap)?;

    let mut img = original_image.to_rgb8();
    let font = if legend_size > 0 {
        let font = Vec::from(include_bytes!("./../assets/roboto-mono-stripped.ttf") as &[u8]);
        rusttype::Font::try_from_vec(font)
    } else {
        None
    };
    for prediction in predictions {
        let dx = prediction.x_max - prediction.x_min;
        let dy = prediction.y_max - prediction.y_min;

        if dx > 0 && dy > 0 {
            imageproc::drawing::draw_hollow_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(prediction.x_min as i32, prediction.y_min as i32)
                    .of_size(dx as u32, dy as u32),
                image::Rgb([255, 0, 0]),
            );
        }
        if let Some(font) = font.as_ref() {
            imageproc::drawing::draw_filled_rect_mut(
                &mut img,
                imageproc::rect::Rect::at(prediction.x_min as i32, prediction.y_min as i32)
                    .of_size(dx as u32, legend_size),
                image::Rgb([170, 0, 0]),
            );
            let legend = format!(
                "{}   {:.0}%",
                prediction.label,
                prediction.confidence * 100_f32
            );
            imageproc::drawing::draw_text_mut(
                &mut img,
                image::Rgb([255, 255, 255]),
                prediction.x_min as i32,
                prediction.y_min as i32,
                rusttype::Scale::uniform(legend_size as f32 - 1.),
                font,
                &legend,
            )
        }
    }
    Ok(DynamicImage::ImageRgb8(img))
}

async fn save_image(img: DynamicImage, image_path: String, suffix: &str) -> anyhow::Result<()> {
    let mut buffer = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 100);
    encoder.encode_image(&img)?;
    let file_path = append_suffix_to_filename(image_path.as_str(), suffix);
    let mut file = File::create(file_path.clone()).await?;
    file.write_all(&buffer).await?;
    debug!("Saved image: {:?}", file_path);
    Ok(())
}

fn append_suffix_to_filename(original_path: &str, suffix: &str) -> PathBuf {
    let path = Path::new(original_path);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

    let mut new_filename = String::from(stem);
    new_filename.push_str(suffix);
    if !extension.is_empty() {
        new_filename.push('.');
        new_filename.push_str(extension);
    }

    path.with_file_name(new_filename)
}

async fn test_image(image: String, args: Args, detector: Detector) -> anyhow::Result<()> {
    let contents = read_jpeg_file(image.clone()).await?;

    let reader = Reader::new(Cursor::new(contents.as_ref()))
        .with_guessed_format()
        .expect("Cursor io never fails");

    let bboxes = detector.detect(reader)?;

    let predictions =
        from_bbox_to_predictions(bboxes, 0.5, &coco_classes::NAMES, detector.labels());

    let reader = Reader::new(Cursor::new(contents.as_ref()))
        .with_guessed_format()
        .expect("Cursor io never fails");
    let img = img_with_bbox(predictions, reader, args.legend_size)?;

    save_image(img, image, "-od").await?;

    Ok(())
}

async fn read_jpeg_file(file_path: String) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(file_path).await?;
    let mut contents = vec![];
    file.read_to_end(&mut contents).await?;
    Ok(contents)
}

fn ensure_directory_exists(path: Option<String>) -> anyhow::Result<()> {
    if let Some(path) = path {
        let path = Path::new(&path);
        if !path.exists() {
            fs::create_dir_all(path)?;
        }
    }
    Ok(())
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

async fn download_models(model_path: String) -> anyhow::Result<()> {
    ensure_directory_exists(Some(model_path.clone()))?;
    let path = PathBuf::from(model_path);
    let api = hf_hub::api::tokio::Api::new()?;
    let api = api.model("lmz/candle-yolo-v8".to_string());
    for size in ["s", "m", "l", "x"] {
        let filename = format!("yolov8{size}.safetensors");
        let cached_file_path = api.get(&filename).await?;
        tokio::fs::copy(cached_file_path, path.join(filename)).await?;
    }
    Ok(())
}

async fn test(detector: Detector, args: Args) -> anyhow::Result<()> {
    let bboxes = detector.test_detection()?;
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
    Ok(())
}
