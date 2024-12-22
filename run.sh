cp /c/Users/Jasmine/Downloads/meshlet_builder.rs /d/bevy/examples/3d/meshlet.rs
cargo run --jobs 8 --example meshlet --features meshlet,meshlet_processor --release > log.txt
cp /c/Users/Jasmine/Downloads/meshlet_runner.rs /d/bevy/examples/3d/meshlet.rs
cargo run --jobs 8 --example meshlet --features meshlet,meshlet_processor --release
