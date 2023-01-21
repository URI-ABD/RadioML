pub use hdf5::{Dataset, File};
use ndarray::{concatenate, s, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::SamplingStrategy;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

/// .
///
/// # Panics
///
/// Panics if .
pub fn get_samples_processed(
    sample_ds: hdf5::File,
    number_samples: usize,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    // Unpack the data into dataset references.
    let mygroup: Result<hdf5::Group, hdf5::Error> = sample_ds.group("/");
    let f1: Result<Dataset, hdf5::Error> = mygroup.as_ref().unwrap().dataset("X");
    let f2: Result<Dataset, hdf5::Error> = mygroup.as_ref().unwrap().dataset("Y");
    let f3: Result<Dataset, hdf5::Error> = mygroup.as_ref().unwrap().dataset("Z");
    // Moving the datasets into Array's
    let xsubslice: Result<Array2<f64>, hdf5::Error> = f1.to_owned().unwrap().read();
    let ysubslice: Result<Array2<f64>, hdf5::Error> = f2.to_owned().unwrap().read();
    let zsubslice: Result<Array2<f64>, hdf5::Error> = f3.to_owned().unwrap().read();
    let mut sample_test = number_samples; //copying into a separate variable that we can lower
                                          // Reduces the value of the noise to a limit of the size of the dataset referenced.
    if sample_test > f1.as_ref().unwrap().shape()[0] {
        sample_test = f1.as_ref().unwrap().shape()[0];
    }
    // Selects a random sample based on the noise_no value which is the size of the same requested.
    let end_val: i32 = f1.unwrap().shape()[0] as i32; //determine length of dataset
                                                      //let mut rng = rand::thread_rng();  //random seed
    let mut rng = ChaCha8Rng::seed_from_u64(42); //selected seed.
    let mut idx: Vec<i32> = (0..end_val).collect::<Vec<i32>>(); //develop an array indices
    let mut idx: Vec<i32> = idx
        .choose_multiple(&mut rng, sample_test)
        .cloned()
        .collect(); //select the samples
    idx.sort();
    println!("{:?}", idx[0]);
    /*
    Generating the initial noise array's to build up with the samples selected. First val allows for easy selection
    of the sample from idx. Load the first sample into the array.
    */
    let first_val = idx[0];
    let mut _x_test: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
        xsubslice.clone().unwrap().sample_axis_using(
            Axis(0),
            200,
            SamplingStrategy::WithoutReplacement,
            &mut rng,
        );
    println!("x_test samples complete");
    let mut x_sample: Array2<f64> = xsubslice
        .clone()
        .unwrap()
        .slice(s![first_val..(first_val + 1), ..])
        .to_owned();
    let mut y_sample: Array2<f64> = ysubslice
        .clone()
        .unwrap()
        .slice(s![first_val..(first_val + 1), ..])
        .to_owned();
    let mut z_sample: Array2<f64> = zsubslice
        .clone()
        .unwrap()
        .slice(s![first_val..(first_val + 1), ..])
        .to_owned();
    // Iterate and build up the noise array, skip the first sample.
    idx.iter().skip(1).for_each(|&i| {
        /*
        General approach for sample collection is VERY slow. Need to determine how to perform a better approach than
        concatenate. Perhaps vec would perform better
        */
        x_sample = concatenate![
            Axis(0),
            x_sample,
            xsubslice
                .clone()
                .unwrap()
                .slice(s![i..(i + 1), ..])
                .to_owned()
        ];
        y_sample = concatenate![
            Axis(0),
            y_sample,
            ysubslice
                .clone()
                .unwrap()
                .slice(s![i..(i + 1), ..])
                .to_owned()
        ];
        z_sample = concatenate![
            Axis(0),
            z_sample,
            zsubslice
                .clone()
                .unwrap()
                .slice(s![i..(i + 1), ..])
                .to_owned()
        ];
    });
    println!("added {} samples.", x_sample.shape()[0]);
    return (x_sample, y_sample, z_sample);
}
