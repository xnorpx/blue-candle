use blue_candle::{api::VisionDetectionResponse, detector::BIKE_IMAGE_BYTES};
use clap::Parser;
use reqwest::{multipart, Body, Client};
use std::time::{Duration, Instant};
use tokio::fs::File;
use tokio_util::codec::{BytesCodec, FramedRead};

/// Simple OB Service Test Application
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Origin for the requests
    #[clap(short, long, default_value = "http://127.0.0.1:32168")]
    origin: String,

    /// Min confidence
    #[arg(long, default_value_t = 0.60)]
    pub min_confidence: f32,

    /// Optional image input path
    #[clap(short, long)]
    image: Option<String>,

    /// Save image with boundary bbox
    #[clap(short = 'I', long)]
    image_ob: Option<String>,

    /// Number of requests to make
    #[clap(short, long, default_value_t = 1)]
    number_of_requests: u32,

    /// Interval in milliseconds for making requests
    #[clap(long, default_value_t = 1000)]
    interval: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut futures = Vec::with_capacity(args.number_of_requests as usize);

    let start_time = Instant::now();
    for i in 0..args.number_of_requests {
        let image = args.image.clone();
        let origin = args.origin.clone();
        let min_confidence = args.min_confidence;
        futures.push(tokio::task::spawn(send_vision_detection_request(
            origin,
            image,
            min_confidence,
        )));
        if i < args.number_of_requests - 1 {
            tokio::time::sleep(std::time::Duration::from_millis(args.interval)).await;
        }
    }
    let results = futures::future::join_all(futures).await;
    let runtime_duration = Instant::now().duration_since(start_time);
    let mut request_times: Vec<Duration> = Vec::with_capacity(args.number_of_requests as usize);

    let mut vision_detection_response = VisionDetectionResponse::default();
    results.into_iter().for_each(|result| {
        if let Ok(Ok(result)) = result {
            vision_detection_response = result.0;
            request_times.push(result.1);
        }
    });

    println!("{:#?}", vision_detection_response);

    println!(
        "Calling {}, {} times with {} ms interval",
        args.origin, args.number_of_requests, args.interval
    );
    println!("Runtime duration: {:?}", runtime_duration);
    if !request_times.is_empty() {
        let min_duration = request_times.iter().min().unwrap();
        let max_duration = request_times.iter().max().unwrap();
        let avg_duration = request_times.iter().sum::<Duration>() / request_times.len() as u32;

        println!("Minimum request time: {:?}", min_duration);
        println!("Maximum request time: {:?}", max_duration);
        println!("Average request time: {:?}", avg_duration);
    } else {
        println!("No request times to summarize");
    }

    Ok(())
}

async fn send_vision_detection_request(
    origin: String,
    image: Option<String>,
    min_confidence: f32,
) -> anyhow::Result<(VisionDetectionResponse, Duration)> {
    let url = reqwest::Url::parse(&origin)?.join("v1/vision/detection")?;
    let client = Client::new();

    let image_part = if let Some(image) = image {
        let file = File::open(image).await?;
        let stream = FramedRead::new(file, BytesCodec::new());
        let body = Body::wrap_stream(stream);
        multipart::Part::stream(body).file_name("image.jpg")
    } else {
        multipart::Part::bytes(BIKE_IMAGE_BYTES.to_vec()).file_name("image.jpg")
    };

    let form = multipart::Form::new()
        .text("min_confidence", min_confidence.to_string())
        .part("image", image_part);

    let request_start_time = Instant::now();
    let response = client.post(url).multipart(form).send().await?;
    let response = response.json::<VisionDetectionResponse>().await?;

    Ok((response, Instant::now().duration_since(request_start_time)))
}
