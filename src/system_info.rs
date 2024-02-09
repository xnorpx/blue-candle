use nvml_wrapper::Nvml;
use tracing::info;

pub fn cpu_info() -> anyhow::Result<()> {
    use raw_cpuid::CpuId;
    let cpuid = CpuId::new();

    let cpu_vendor_info = match cpuid.get_vendor_info() {
        Some(vendor_info) => vendor_info.as_str().to_owned(),
        None => "Unknown".to_owned(),
    };

    let cpu_brand = match cpuid.get_processor_brand_string() {
        Some(cpu_brand) => cpu_brand.as_str().to_owned(),
        None => "Unknown".to_owned(),
    };

    info!(
        "CPU | {} | {} | {} Cores | {} Logical Cores",
        cpu_vendor_info,
        cpu_brand,
        num_cpus::get_physical(),
        num_cpus::get()
    );
    Ok(())
}

pub fn cuda_gpu_info() -> anyhow::Result<()> {
    let nvml = Nvml::init()?;
    let num_cuda_gpus = nvml.device_count()?;
    for index in 0..num_cuda_gpus {
        let device = nvml.device_by_index(0)?;
        let name = device.name()?;
        let memory = device.memory_info()?;
        let compute_capability = device.cuda_compute_capability()?;
        let cuda_cores = device.num_cores()?;
        info!(
            "Cuda GPU: [{}] | {} | CC {}.{} | {} Cores | {}/{} MB",
            index,
            name,
            compute_capability.major,
            compute_capability.minor,
            cuda_cores,
            bytes_to_megabytes(memory.used),
            bytes_to_megabytes(memory.total)
        );
    }
    Ok(())
}

fn bytes_to_megabytes(bytes: u64) -> u64 {
    (bytes as f64 / 1_048_576.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn print_cuda_gpu_info() {
        cuda_gpu_info().unwrap();
    }

    #[test]
    fn print_cpu_info() {
        cpu_info().unwrap()
    }
}
