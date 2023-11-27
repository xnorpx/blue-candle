use crate::model::{Multiples, YoloV8};
use anyhow::{anyhow, bail};
use candle::{
    utils::{cuda_is_available, has_mkl, with_avx, with_f16c},
    DType, Device, IndexOp, Tensor,
};
use candle_core as candle;
use candle_nn::{Module, VarBuilder};
use image::{io::Reader, ImageFormat};
use std::{io::Cursor, time::Instant};
use tracing::{debug, info};

// For testing
pub static BIKE_IMAGE_BYTES: &[u8] = include_bytes!("../assets/crossing.jpg");

// Include a default model
static DEFAULT_MODEL: &[u8] = include_bytes!("../models/yolov8n.safetensors");
static DEFAULT_MODEL_MULTIPLES: Multiples = Multiples::n();

pub type Bboxes = Vec<Vec<Bbox<Vec<KeyPoint>>>>;

#[derive(Clone, Debug)]
pub struct Detector {
    device: Device,
    model: YoloV8,
    confidence_threshold: f32,
    nms_threshold: f32,
    labels: Vec<String>,
    image_path: Option<String>,
}

impl Detector {
    pub fn new(
        force_cpu: bool,
        model: Option<String>,
        confidence_threshold: f32,
        nms_threshold: f32,
        labels: Vec<String>,
        image_path: Option<String>,
    ) -> anyhow::Result<Self> {
        let device = if !force_cpu && cuda_is_available() {
            info!("Detector is initialized for GPU");
            Device::new_cuda(0)?
        } else {
            info!(
                "Detector is initialized for CPU with mkl: {:?}, with avx: {:?} with f16c: {:?}",
                has_mkl(),
                with_avx(),
                with_f16c()
            );
            Device::Cpu
        };

        let (vb, multiples) = if let Some(model) = model {
            let Some(multiplies) = Multiples::from_filename(&model) else {
                bail!("Failed to parse multiples from model filename: {:?}", model);
            };
            let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
            (vb, multiplies)
        } else {
            let multiplies = DEFAULT_MODEL_MULTIPLES;
            let vb =
                VarBuilder::from_buffered_safetensors(DEFAULT_MODEL.to_vec(), DType::F32, &device)?;
            (vb, multiplies)
        };

        let model = YoloV8::load(vb, multiples, 80)?;
        Ok(Self {
            model,
            confidence_threshold,
            nms_threshold,
            device,
            labels,
            image_path,
        })
    }

    pub fn test_detection(&self) -> anyhow::Result<(Bboxes, f32, f32)> {
        info!("Test detection");
        let start_detection_time = Instant::now();
        let reader = Reader::new(Cursor::new(BIKE_IMAGE_BYTES))
            .with_guessed_format()
            .expect("Cursor io never fails");
        let bboxes = self.detect_inner(reader)?;
        if bboxes.0.is_empty() {
            bail!("Detection failed");
        }
        info!(
            "Detection succeeded in: {:#?}",
            Instant::now().duration_since(start_detection_time)
        );
        Ok(bboxes)
    }

    pub fn detect(&self, reader: Reader<Cursor<&[u8]>>) -> anyhow::Result<(Bboxes, f32, f32)> {
        let start_detection_time = Instant::now();
        let res = self.detect_inner(reader)?;
        debug!(
            "Detection succeeded in: {:#?}",
            Instant::now().duration_since(start_detection_time)
        );
        Ok(res)
    }

    fn detect_inner(&self, reader: Reader<Cursor<&[u8]>>) -> anyhow::Result<(Bboxes, f32, f32)> {
        assert_eq!(reader.format(), Some(ImageFormat::Jpeg));
        let original_image = reader.decode().map_err(candle::Error::wrap)?;

        let w = original_image.width() as usize;
        let h = original_image.height() as usize;

        let (width, height) = {
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };
        let image_t = {
            let img = original_image.resize_exact(
                width as u32,
                height as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let data = img.to_rgb8().into_raw();
            Tensor::from_vec(
                data,
                (img.height() as usize, img.width() as usize, 3),
                &self.device,
            )?
            .permute((2, 0, 1))?
        };
        let image_t = (image_t.unsqueeze(0)?.to_dtype(DType::F32)? * (1. / 255.))?;
        let pred = self.model.forward(&image_t)?.squeeze(0)?;
        let bboxes = from_tensor_to_bbox(&pred, self.confidence_threshold, self.nms_threshold)?;
        Ok((
            bboxes,
            (w as f32 / width as f32),
            (h as f32 / height as f32),
        ))
    }

    pub fn labels(&self) -> &Vec<String> {
        &self.labels
    }

    pub fn image_path(&self) -> &Option<String> {
        &self.image_path
    }
}

pub fn from_tensor_to_bbox(
    pred: &Tensor,
    confidence_threshold: f32,
    nms_threshold: f32,
) -> anyhow::Result<Vec<Vec<Bbox<Vec<KeyPoint>>>>> {
    let pred = pred.to_device(&Device::Cpu)?;
    let (pred_size, npreds) = pred.dims2()?;
    let nclasses = pred_size - 4;
    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<Vec<Bbox<Vec<KeyPoint>>>> = (0..nclasses).map(|_| vec![]).collect();
    // Extract the bounding boxes for which confidence is above the threshold.
    for index in 0..npreds {
        let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
        let confidence = *pred[4..]
            .iter()
            .max_by(|x, y| x.total_cmp(y))
            .ok_or_else(|| anyhow!("No confidence interval"))?;
        if confidence > confidence_threshold {
            let mut class_index = 0;
            for i in 0..nclasses {
                if pred[4 + i] > pred[4 + class_index] {
                    class_index = i
                }
            }
            if pred[class_index + 4] > 0. {
                let bbox = Bbox {
                    xmin: pred[0] - pred[2] / 2.,
                    ymin: pred[1] - pred[3] / 2.,
                    xmax: pred[0] + pred[2] / 2.,
                    ymax: pred[1] + pred[3] / 2.,
                    confidence,
                    data: vec![],
                };
                bboxes[class_index].push(bbox)
            }
        }
    }

    non_maximum_suppression(&mut bboxes, nms_threshold);
    Ok(bboxes)
}

/// A bounding box around an object.
#[derive(Debug, Clone)]
pub struct Bbox<D> {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
    pub confidence: f32,
    pub data: D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub mask: f32,
}

/// Intersection over union of two bounding boxes.
pub fn iou<D>(b1: &Bbox<D>, b2: &Bbox<D>) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

pub fn non_maximum_suppression<D>(bboxes: &mut [Vec<Bbox<D>>], threshold: f32) {
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}
