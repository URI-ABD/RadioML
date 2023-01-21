use radio_ml::reader;

fn main() {
    println!("Getting Started ...");
    let folders_from_root = [
        "Users",
        "najibishaq",
        "Documents",
        "research",
        "data",
        "rf_terry",
    ];

    let num_samples = 100;
    let samples = reader::RadioData::read(&folders_from_root, num_samples).join();
    println!("Joined samples together into shape {:?}", samples.shape());

    println!("Success!")
}
