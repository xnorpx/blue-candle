#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

pub mod api;
pub mod coco_classes;
pub mod detector;
pub mod model;
pub mod server_stats;
pub mod system_info;
pub mod utils;
