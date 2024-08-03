use blue_candle::server::LogLevel;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[command(author = "Marcus Asteborg", version=env!("CARGO_PKG_VERSION"))]
pub struct Args {
    /// Filters the results to include only the specified labels. Provide labels separated by spaces.
    /// Example: --labels "person cup"
    #[arg(long, value_delimiter = ' ', num_args = 1..)]
    pub labels: Vec<String>,

    /// The port on which the server will listen for HTTP requests.
    /// Default is 32168. Example usage: --port 1337
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

#[cfg(target_os = "windows")]
mod blue_candle_windows_service {
    use blue_candle::detector::Detector;
    use blue_candle::server::run_server;
    use blue_candle::system_info;
    use candle_core::utils::{cuda_is_available, with_avx};
    use clap::Parser;
    use std::path::PathBuf;
    use std::thread;
    use std::time::Duration;
    use std::{env, ffi::OsString};
    use tokio::runtime::Runtime;
    use tokio_util::sync::CancellationToken;
    use tracing::{debug, error, info};
    use windows_service::{
        define_windows_service,
        service::{
            ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl,
            ServiceExitCode, ServiceInfo, ServiceState, ServiceStatus, ServiceType,
        },
        service_control_handler::{self, ServiceControlHandlerResult},
        service_dispatcher,
        service_manager::{ServiceManager, ServiceManagerAccess},
        Result,
    };

    use crate::Args;

    define_windows_service!(ffi_service_main, my_service_main);

    pub fn my_service_main(arguments: Vec<OsString>) {
        let args = Args::parse_from(arguments.iter().map(|x| x.to_string_lossy().to_string()));

        // Init logging

        info!("Starting Blue Candle object detection service");
        info!(avx = ?with_avx(), cuda = ?cuda_is_available(), "Compiled with");

        system_info::cpu_info().expect("Failed to get CPU info");
        let num_cores = num_cpus::get();

        let mut blocking_threads = if !args.cpu && cuda_is_available() {
            system_info::cuda_gpu_info().expect("Failed to get CUDA GPU info");
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
            .build()
            .expect("Failed to build Tokio runtime.");

        debug!("Tokio initilized with {blocking_threads} blocking threads.");

        let detector = Detector::new(
            args.cpu,
            args.model.clone(),
            args.confidence_threshold,
            args.nms_threshold,
            args.labels.clone(),
            args.image_path.clone(),
        )
        .expect("Failed to create detector");

        if let Err(e) = run_service(args.port, detector, rt) {
            error!(?e, "Failed to run service");
        }
    }

    fn run_service(port: u16, detector: Detector, rt: Runtime) -> Result<()> {
        let cancellation_token = CancellationToken::new();
        let cancellation_token2 = cancellation_token.clone();

        let status_handle =
            service_control_handler::register("blue_candle_service", move |control_event| {
                match control_event {
                    ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
                    ServiceControl::Stop => {
                        cancellation_token2.cancel();
                        ServiceControlHandlerResult::NoError
                    }
                    _ => ServiceControlHandlerResult::NotImplemented,
                }
            })?;

        status_handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: ServiceState::Running,
            controls_accepted: ServiceControlAccept::STOP,
            exit_code: ServiceExitCode::Win32(0),
            checkpoint: 0,
            wait_hint: Duration::default(),
            process_id: None,
        })?;

        thread::spawn(move || {
            rt.block_on(async { run_server(port, detector, cancellation_token).await })
                .unwrap();

            status_handle
                .set_service_status(ServiceStatus {
                    service_type: ServiceType::OWN_PROCESS,
                    current_state: ServiceState::Stopped,
                    controls_accepted: ServiceControlAccept::empty(),
                    exit_code: ServiceExitCode::Win32(0),
                    checkpoint: 0,
                    wait_hint: Duration::default(),
                    process_id: None,
                })
                .unwrap();
        });

        Ok(())
    }

    pub fn run() -> Result<()> {
        service_dispatcher::start("blue_candle_service", ffi_service_main)
    }

    pub fn install_service(path: PathBuf, launch_args: Vec<String>) -> Result<()> {
        let manager_access = ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE;
        let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

        let service_binary_path = path.to_string_lossy();

        let service_info = ServiceInfo {
            name: OsString::from("blue_candle_service"),
            display_name: OsString::from("Blue Candle Service"),
            service_type: ServiceType::OWN_PROCESS,
            error_control: ServiceErrorControl::Normal,
            executable_path: OsString::from(service_binary_path.as_ref()).into(),
            launch_arguments: launch_args.iter().map(OsString::from).collect(),
            dependencies: vec![],
            account_name: None, // run as System
            account_password: None,
            start_type: windows_service::service::ServiceStartType::AutoStart,
        };

        let service = service_manager.create_service(&service_info, ServiceAccess::all())?;
        service.start(&launch_args)?;

        println!("Service installed and started successfully.");
        Ok(())
    }

    pub fn uninstall_service() -> Result<()> {
        let manager_access = ServiceManagerAccess::CONNECT;
        let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;
        let service = service_manager.open_service("blue_candle_service", ServiceAccess::all())?;

        service.stop()?;
        service.delete()?;

        println!("Service uninstalled successfully.");
        Ok(())
    }
}

#[derive(Parser)]
#[command(
    name = "blue_candle_service",
    version = "1.0",
    author = "Marcus Asteborg",
    about = "A Windows Service for Blue Candle"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Run,
    Install {
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        launch_args: Vec<String>,
    },
    Uninstall,
}

#[cfg(target_os = "windows")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;

    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Run) => {
            blue_candle_windows_service::run().context("Failed to run the service")
        }
        Some(Commands::Install { path, launch_args }) => {
            blue_candle_windows_service::install_service(path.clone(), launch_args.clone())
                .context("Failed to install the service")
        }
        Some(Commands::Uninstall) => blue_candle_windows_service::uninstall_service()
            .context("Failed to uninstall the service"),
        None => {
            eprintln!("No command specified. Use --help for more information.");
            Ok(())
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn main() {
    panic!("This service only runs on Windows.");
}
