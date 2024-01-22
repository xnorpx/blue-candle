use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};
use tracing::{info, trace};

const STAT_SUMMARY_INTERVAL: Duration = Duration::from_secs(60);

pub struct ServerStats {
    request_times: VecDeque<Instant>,
    last_log_time: Instant,
    freq_map: HashMap<u64, usize>,
    total_requests: usize,
    min_request_time: Duration,
    min_inference_time: Duration,
    min_processing_time: Duration,
    max_request_time: Duration,
    max_inference_time: Duration,
    max_processing_time: Duration,
}

impl ServerStats {
    pub fn calculate_and_log_stats(
        &mut self,
        now: Instant,
        request_time: Duration,
        processing_time: Duration,
        inferece_time: Duration,
    ) {
        if now.duration_since(self.last_log_time) >= STAT_SUMMARY_INTERVAL
            && !self.request_times.is_empty()
        {
            for request_time in self.request_times.drain(..) {
                let second = now.duration_since(request_time).as_secs();
                *self.freq_map.entry(second).or_insert(0) += 1;
            }

            let mut max_requests_per_second = 0;
            for (_, count) in self.freq_map.drain() {
                max_requests_per_second = max_requests_per_second.max(count);
                trace!("Max Req/Sec: {}", count);
            }

            info!("Stats: Total Req: {}, Max Req/Sec: {}, Min Inference : {:#?}, Max Inference: {:#?}, Min Processing: {:#?}, Max Processing: {:#?}, Min Request: {:#?}, Max Request: {:#?}",
                self.total_requests,
                max_requests_per_second,
                self.min_inference_time,
                self.max_inference_time,
                self.min_processing_time,
                self.max_processing_time,
                self.min_request_time,
                self.max_request_time
            );

            self.last_log_time = now;
        }

        self.max_inference_time = self.max_inference_time.max(inferece_time);
        self.min_inference_time = self.min_inference_time.min(inferece_time);

        self.max_processing_time = self.max_processing_time.max(processing_time);
        self.min_processing_time = self.min_processing_time.min(processing_time);

        self.max_request_time = self.max_request_time.max(request_time);
        self.min_request_time = self.min_request_time.min(request_time);

        self.total_requests += 1;
        self.request_times.push_back(now);
    }
}

impl Default for ServerStats {
    fn default() -> Self {
        Self {
            request_times: VecDeque::with_capacity(10000),
            last_log_time: Instant::now(),
            total_requests: Default::default(),
            freq_map: HashMap::with_capacity(120),
            max_request_time: Duration::ZERO,
            max_inference_time: Duration::ZERO,
            max_processing_time: Duration::ZERO,
            min_request_time: Duration::from_secs_f64(1337.),
            min_inference_time: Duration::from_secs_f64(1337.),
            min_processing_time: Duration::from_secs_f64(1337.),
        }
    }
}
