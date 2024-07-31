use blue_candle::{
    coco_classes,
    detector::{Detector, BIKE_IMAGE_BYTES},
    server::{from_bbox_to_predictions, run_server, LogLevel},
    system_info,
    utils::{download_models, ensure_directory_exists, img_with_bbox, read_jpeg_file, save_image},
};
use candle::utils::{cuda_is_available, with_avx};
use candle_core as candle;
use clap::Parser;
use image::ImageReader;
use std::{env, io::Cursor, time::Instant};
use tracing::{debug, info, Level};

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

    /// Sets a custom file path for logging
    #[clap(short, long, value_parser)]
    log_path: Option<String>,

    /// Sets the level of logging
    #[clap(short, long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,

    /// Max blocking threads, max will be number of cores of the system
    #[arg(long)]
    pub blocking_threads: Option<usize>,
}

fn main() -> anyhow::Result<()> {
    setup_ansi_support();

    let args = Args::parse();

    // Logging
    let _guard = if let Some(log_path) = args.log_path.clone() {
        println!(
            "Starting Blue Candle, logging into: {}/blue_candle.log",
            log_path
        );
        let file_appender = tracing_appender::rolling::daily(&log_path, "blue_candle.log");
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
        tracing_subscriber::fmt()
            .with_writer(non_blocking)
            .with_max_level(Level::from(args.log_level))
            .with_ansi(false)
            .init();
        Some(_guard)
    } else {
        tracing_subscriber::fmt()
            .with_max_level(Level::from(args.log_level))
            .init();
        None
    };

    info!("Starting Blue Candle object detection service");
    info!(avx = ?with_avx(), cuda = ?cuda_is_available(), "Compiled with");

    system_info::cpu_info()?;
    let num_cores = num_cpus::get();

    let mut blocking_threads = if !args.cpu && cuda_is_available() {
        system_info::cuda_gpu_info()?;
        args.blocking_threads.unwrap_or(1)
    } else {
        // Run CPU inference on one core
        env::set_var("RAYON_NUM_THREADS", "1");
        args.blocking_threads.unwrap_or(num_cores - 1)
    };
    blocking_threads = blocking_threads.clamp(1, num_cores - 1);

    // Configure Tokio
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1) // Number of request will be low so 1 thread is enough
        .max_blocking_threads(blocking_threads) // Number of request for processing
        .enable_all()
        .build()?;

    debug!("Tokio initilized with {blocking_threads} blocking threads.");

    let detector = Detector::new(
        args.cpu,
        args.model.clone(),
        args.confidence_threshold,
        args.nms_threshold,
        args.labels.clone(),
        args.image_path.clone(),
    )?;

    rt.block_on(async {
        ensure_directory_exists(args.image_path.clone()).await?;

        if let Some(model_path) = args.model_path {
            download_models(model_path).await?;
            return Ok(());
        }
        if args.test {
            test(detector, args).await?;
            return Ok(());
        }

        match args.image.clone() {
            None => run_server(args.port, detector).await?,
            Some(image) => test_image(image, args, detector).await?,
        };
        Ok(())
    })
}

async fn test_image(image: String, args: Args, detector: Detector) -> anyhow::Result<()> {
    let start_test_time = Instant::now();
    let contents = read_jpeg_file(image.clone()).await?;

    let (bboxes, inference_time, processing_time) = detector.detect(&contents)?;

    let predictions =
        from_bbox_to_predictions(bboxes, 0.5, &coco_classes::NAMES, detector.labels());

    let reader = ImageReader::new(Cursor::new(contents.as_ref()))
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
    let reader = ImageReader::new(Cursor::new(BIKE_IMAGE_BYTES))
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
