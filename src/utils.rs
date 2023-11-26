use crate::api::Prediction;
use candle_core as candle;
use image::{codecs::jpeg::JpegEncoder, io::Reader, DynamicImage, ImageFormat};
use std::{
    io::Cursor,
    path::{Path, PathBuf},
};
use tokio::{
    fs::{self, File},
    io::{AsyncReadExt, AsyncWriteExt},
};
use tracing::debug;

pub async fn download_models(model_path: String) -> anyhow::Result<()> {
    ensure_directory_exists(Some(model_path.clone())).await?;
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

pub fn img_with_bbox(
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

pub async fn read_jpeg_file(file_path: String) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(file_path).await?;
    let mut contents = vec![];
    file.read_to_end(&mut contents).await?;
    Ok(contents)
}

pub async fn save_image(img: DynamicImage, image_path: String, suffix: &str) -> anyhow::Result<()> {
    let mut buffer = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut buffer, 100);
    encoder.encode_image(&img)?;
    let file_path = append_suffix_to_filename(image_path.as_str(), suffix);
    let mut file = File::create(file_path.clone()).await?;
    file.write_all(&buffer).await?;
    debug!("Saved image: {:?}", file_path);
    Ok(())
}

pub async fn ensure_directory_exists(path: Option<String>) -> anyhow::Result<()> {
    if let Some(path) = path {
        let path = Path::new(&path);
        if !path.exists() {
            fs::create_dir_all(path).await?;
        }
    }
    Ok(())
}

pub fn append_suffix_to_filename(original_path: &str, suffix: &str) -> PathBuf {
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
