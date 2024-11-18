use axum::http::response;
use candle_core::utils::cuda_is_available;
use tokio::sync::oneshot;
use tracing::{debug, error};
use tracing_subscriber::field::debug;

use crate::{
    api::{Prediction, VisionDetectionRequest, VisionDetectionResponse},
    coco_classes,
    detector::{self, Bbox, Detector, DetectorConfig, KeyPoint},
};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc::{self, Receiver, Sender},
    Arc,
};

struct DetectorWorker {
    receiver: Receiver<(
        VisionDetectionRequest,
        oneshot::Sender<VisionDetectionResponse>,
    )>,
    labels: Vec<String>,
    detector: Detector,
    id: usize,
}

impl DetectorWorker {
    fn new(
        id: usize,
        receiver: Receiver<(
            VisionDetectionRequest,
            oneshot::Sender<VisionDetectionResponse>,
        )>,
        detector_config: DetectorConfig,
    ) -> anyhow::Result<Self> {
        Ok(DetectorWorker {
            id,
            receiver,
            labels: detector_config.labels.clone(),
            detector: Detector::new(detector_config)?,
        })
    }

    fn run(&mut self) {
        debug!("Worker ID: {}, Starting worker", self.id);
        while let Ok((vision_request, response_sender)) = self.receiver.recv() {
            debug!(
                "Worker ID: {}, Received image: {:#?}",
                self.id, vision_request.image_name
            );
            let (bboxes, inference_time, processing_time) =
                match self.detector.detect(&vision_request.image_data) {
                    Ok(result) => result,
                    Err(e) => {
                        error!("Detection error: {:?}", e);
                        continue;
                    }
                };
            let predictions = from_bbox_to_predictions(
                bboxes,
                vision_request.min_confidence,
                &coco_classes::NAMES,
                &self.labels,
            );

            // if !predictions.is_empty() {
            //     if let Some(image_path) = server_state.detector.image_path() {
            //         let reader = ImageReader::new(Cursor::new(vision_request.image_data.as_ref()))
            //             .with_guessed_format()
            //             .expect("Cursor io never fails");
            //         let img = img_with_bbox(predictions.clone(), reader, 15)?;
            //         // The api doesn't provide a source id or a source name so we just generate a uuid here.
            //         let picture_name = format!("{}/{}.jpg", image_path, Uuid::new_v4());
            //         save_image(img, picture_name, "-od").await?;
            //     }
            // }

            // let image = vision_request.image_name.split('.').next().unwrap_or("");
            // debug!(
            //     "Image: {}, request time {:#?}, processing time: {:#?}, inference time: {:#?}",
            //     image, request_time, processing_time, inference_time
            // );

            // {
            //     let mut stats = server_state.stats.lock().await;
            //     stats.calculate_and_log_stats(
            //         request_start_time,
            //         request_time,
            //         processing_time,
            //         inference_time,
            //     );
            // }

            let count = predictions.len() as i32;
            let response = VisionDetectionResponse {
                success: true,
                message: "".into(),
                error: None,
                predictions,
                count,
                command: "detect".into(),
                module_id: "Yolo8".into(),
                execution_provider: if self.detector.is_gpu() {
                    "GPU".to_string()
                } else {
                    "CPU".to_string()
                },
                can_useGPU: cuda_is_available(),
                inference_ms: inference_time.as_millis() as i32,
                process_ms: processing_time.as_millis() as i32,
                analysis_round_trip_ms: 0_i32,
            };
            //analysis_round_trip_ms: request_time.as_millis() as i32,
            if let Err(e) = response_sender.send(response) {
                error!("Failed to send response: {:?}", e);
            }
        }
        debug!("Worker ID: {}, Stop worker", self.id);
    }
}

pub struct DetectorDispatcher {
    senders: Vec<
        Sender<(
            VisionDetectionRequest,
            oneshot::Sender<VisionDetectionResponse>,
        )>,
    >,
    work_counter: AtomicU64,
}

impl DetectorDispatcher {
    pub fn new(size: usize, detector_config: DetectorConfig) -> anyhow::Result<Arc<Self>> {
        let mut senders = Vec::with_capacity(size);
        for id in 0..size {
            let (tx, rx) = mpsc::channel();
            senders.push(tx);
            let mut worker = DetectorWorker::new(id, rx, detector_config.clone())?;
            tokio::task::spawn_blocking(move || {
                worker.run();
            });
        }
        Ok(Arc::new(DetectorDispatcher {
            senders,
            work_counter: AtomicU64::new(0),
        }))
    }

    pub fn dispatch_work(
        &self,
        vision_request: VisionDetectionRequest,
        response: oneshot::Sender<VisionDetectionResponse>,
    ) {
        let work_number = self.work_counter.fetch_add(1, Ordering::SeqCst);
        let sender_index = (work_number % self.senders.len() as u64) as usize;
        debug!("Dispatching work to worker: {}", sender_index);
        let sender = &self.senders[sender_index];
        sender.send((vision_request, response)).unwrap();
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
