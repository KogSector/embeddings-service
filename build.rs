fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build script simplified - no protobuf compilation needed for Kafka-only architecture
    println!("cargo:rerun-if-changed=build.rs");
    Ok(())
}
