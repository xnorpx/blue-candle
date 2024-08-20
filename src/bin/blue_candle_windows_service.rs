use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[cfg(target_os = "windows")]
mod windows_service {
    use super::*;
    use clap::{ArgEnum, Parser, Subcommand};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    use windows_service::{
        define_windows_service,
        service::{
            ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode,
            ServiceInfo, ServiceState, ServiceStatus, ServiceType,
        },
        service_control_handler::{self, ServiceControlHandlerResult},
        service_dispatcher,
        service_manager::{ServiceManager, ServiceManagerAccess},
        Result,
    };

    define_windows_service!(ffi_service_main, my_service_main);

    pub fn my_service_main(_arguments: Vec<OsString>) {
        // Init logging
        if let Err(_e) = run_service() {
            // Handle the error
        }
    }

    fn run_service() -> Result<()> {
        let (shutdown_sender, shutdown_receiver) = mpsc::channel();

        let status_handle =
            service_control_handler::register("blue_candle_service", move |control_event| {
                match control_event {
                    ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
                    ServiceControl::Stop => {
                        shutdown_sender.send(()).unwrap();
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
            run_server(shutdown_receiver);

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

    fn run_server(shutdown_receiver: std::sync::mpsc::Receiver<()>) {
        // This is where you implement your server logic
        println!("Running the server...");

        loop {
            // Perform server tasks here, e.g., handling requests

            // Check for shutdown signal
            if shutdown_receiver
                .recv_timeout(Duration::from_secs(1))
                .is_ok()
            {
                println!("Shutting down the server...");
                break;
            }
        }
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
            executable_path: OsString::from(service_binary_path.as_ref()),
            launch_arguments: launch_args.iter().map(OsString::from).collect(),
            dependencies: vec![],
            account_name: None, // run as System
            account_password: None,
        };

        let service = service_manager.create_service(&service_info, ServiceManagerAccess::all())?;
        service.start(&[])?;

        println!("Service installed and started successfully.");
        Ok(())
    }

    pub fn uninstall_service() -> Result<()> {
        let manager_access = ServiceManagerAccess::CONNECT;
        let service_manager = ServiceManager::local_computer(None::<&str>, manager_access)?;

        let service_access = ServiceManagerAccess::all();
        let service = service_manager.open_service("blue_candle_service", service_access)?;

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
    author = "Author Name",
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
        Some(Commands::Run) => windows_service::run().context("Failed to run the service"),
        Some(Commands::Install { path, launch_args }) => {
            windows_service::install_service(path.clone(), launch_args.clone())
                .context("Failed to install the service")
        }
        Some(Commands::Uninstall) => {
            windows_service::uninstall_service().context("Failed to uninstall the service")
        }
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
