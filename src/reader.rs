use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn open_hdf5_file(folders_from_root: &[&str], data_name: &str) -> hdf5::Result<hdf5::File> {
    let mut data_dir = std::path::PathBuf::from("/");

    for &folder in folders_from_root {
        data_dir.push(folder)
    }

    let path = {
        let mut path = data_dir.clone();
        path.push(data_name);
        assert!(path.exists(), "{:?} does not exist.", &path);
        path
    };

    hdf5::File::open(path)
}

#[derive(Debug)]
pub struct RadioData {
    samples: Vec<Array3<f64>>,
    num_samples: usize,
}

impl RadioData {
    pub fn read(folders_from_root: &[&str], num_samples: usize) -> Self {
        let samples = ModulationMode::variants()
            .into_iter()
            .map(|m| {
                let handle = open_hdf5_file(folders_from_root, m.data_name()).unwrap();
                (m, handle)
            })
            .map(|(&m, handle)| RadioFile::new(handle, m, num_samples).unwrap().join().0)
            .collect::<Vec<_>>();
        Self {
            samples,
            num_samples,
        }
    }

    pub fn validate_sampled(&self) {
        for (_, s) in ModulationMode::variants().iter().zip(self.samples.iter()) {
            assert_eq!([26 * self.num_samples, 1024, 2], s.shape());
        }
    }

    pub fn join(self) -> Array3<f64> {
        let samples = self.samples.iter().map(|s| s.view()).collect::<Vec<_>>();
        ndarray::concatenate(Axis(0), &samples).unwrap()
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub enum ModulationMode {
    ASK_4,
    OOK,
    OOK_2,
}

impl ModulationMode {
    pub fn data_name(&self) -> &str {
        match self {
            Self::ASK_4 => "mod_4ASK.h5",
            Self::OOK => "mod_OOK.h5",
            Self::OOK_2 => "mod2_OOK.h5",
        }
    }

    pub fn variants<'a>() -> &'a [ModulationMode] {
        &[Self::ASK_4, Self::OOK, Self::OOK_2]
    }
}

#[derive(Debug)]
pub struct RadioFile {
    modulation: ModulationMode,
    levels: Vec<SingleSnR>, // len == 26
    sample_indices: Vec<usize>,
}

impl RadioFile {
    pub fn new(
        handle: hdf5::File,
        modulation: ModulationMode,
        num_samples: usize,
    ) -> Result<Self, String> {
        let mut all_iq: Array3<f64> = handle
            .dataset("X")
            .map_err(|reason| format!("Could not read `X` because {}", reason))?
            .read()
            .map_err(|reason| {
                format!(
                    "Could not convert from HDF5 to Array3<f64> because {}",
                    reason
                )
            })?;
        let mut all_iq = all_iq.view_mut();

        assert_eq!(
            all_iq.shape(),
            &[106_496, 1024, 2],
            "iq data had the wrong shape"
        );
        let mut levels = Vec::new();

        for snr in (-20..=30).step_by(2) {
            let (iq, rest) = all_iq.split_at(Axis(0), 4096);
            all_iq = rest;
            let iq = iq.to_owned();
            levels.push(SingleSnR { iq, snr });
        }

        let sample_indices = {
            let mut indices =
                (0..4096).choose_multiple(&mut ChaCha8Rng::seed_from_u64(42), num_samples);
            indices.sort();
            indices
        };

        let levels = levels
            .into_iter()
            .map(|s| s.subsample(&sample_indices))
            .collect::<Vec<_>>();

        Ok(Self {
            modulation,
            levels,
            sample_indices,
        })
    }

    pub fn validate_full(&self) {
        println!("Validating full arrays ...");
        assert_eq!(26, self.levels.len());
        self.levels.iter().for_each(|s| s.validate_full());
        self.levels.iter().for_each(|s| s.print_summary());
    }

    pub fn validate_sampled(&self) {
        println!("Validating sampled arrays ...");
        assert_eq!(26, self.levels.len());
        self.levels
            .iter()
            .for_each(|s| s.validate_sampled(self.sample_indices.len()));
        self.levels.iter().for_each(|s| s.print_summary());
    }

    pub fn join(self) -> (Array3<f64>, ModulationMode) {
        let sub_iqs = self.levels.iter().map(|i| i.iq.view()).collect::<Vec<_>>();
        let iq = ndarray::concatenate(Axis(0), &sub_iqs).unwrap();
        assert_eq!([26 * self.sample_indices.len(), 1024, 2], iq.shape());
        (iq, self.modulation)
    }
}

#[derive(Debug)]
pub struct SingleSnR {
    iq: Array3<f64>, // starts with (4096, 1024, 2). Sampled to (num_samples, 1024, 2)
    snr: i32,        // -50dB for noise, otherwise one of (-20..=30).step_by(2)
}

impl SingleSnR {
    pub fn print_summary(&self) {
        println!("snr: {}dB, shape: {:?}", self.snr, self.iq.shape());
    }

    pub fn validate_full(&self) {
        assert_eq!([4096, 1024, 2], self.iq.shape());
    }

    pub fn validate_sampled(&self, num_samples: usize) {
        assert_eq!([num_samples, 1024, 2], self.iq.shape());
    }

    pub fn subsample(self, indices: &[usize]) -> Self {
        let iq_samples = indices
            .iter()
            .map(|&i| self.iq.index_axis(Axis(0), i))
            .collect::<Vec<_>>();
        let iq_samples = ndarray::stack(Axis(0), &iq_samples).unwrap();
        Self {
            iq: iq_samples,
            snr: self.snr,
        }
    }
}
