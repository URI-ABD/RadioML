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
            .inspect(|m| println!("Reading from file {:?}", m.data_name()))
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
    QAM_64,
    QPSK,
    GMSK,
    QAM_32,
    PSK_16,
    OOK,
    FM,
    APSK_32,
    PSK_32,
    AM_DSB_WC,
    APSK_64,
    OQPSK,
    QAM_128,
    ASK_4,
    AM_SSB_SC,
    AM_DSB_SC,
    PSK_8,
    ASK_8,
    QAM_256,
    APSK_128,
    APSK_16,
    OOK_2,
    BPSK,
    QAM_16,
    AM_SSB_WC,
    Noise_20220222,
}

impl ModulationMode {
    pub fn data_name(&self) -> &str {
        match self {
            Self::QAM_64 => "mod_64QAM.h5",
            Self::QPSK => "mod_QPSK.h5",
            Self::GMSK => "mod_GMSK.h5",
            Self::QAM_32 => "mod_32QAM.h5",
            Self::PSK_16 => "mod_16PSK.h5",
            Self::OOK => "mod_OOK.h5",
            Self::FM => "mod_FM.h5",
            Self::APSK_32 => "mod_32APSK.h5",
            Self::PSK_32 => "mod_32PSK.h5",
            Self::AM_DSB_WC => "mod_AM-DSB-WC.h5",
            Self::APSK_64 => "mod_64APSK.h5",
            Self::OQPSK => "mod_OQPSK.h5",
            Self::QAM_128 => "mod_128QAM.h5",
            Self::ASK_4 => "mod_4ASK.h5",
            Self::AM_SSB_SC => "mod_AM-SSB-SC.h5",
            Self::AM_DSB_SC => "mod_AM-DSB-SC.h5",
            Self::PSK_8 => "mod_8PSK.h5",
            Self::ASK_8 => "mod_8ASK.h5",
            Self::QAM_256 => "mod_256QAM.h5",
            Self::APSK_128 => "mod_128APSK.h5",
            Self::APSK_16 => "mod_16APSK.h5",
            Self::OOK_2 => "mod2_OOK.h5",
            Self::BPSK => "mod_BPSK.h5",
            Self::QAM_16 => "mod_16QAM.h5",
            Self::AM_SSB_WC => "mod_AM-SSB-WC.h5",
            Self::Noise_20220222 => "noise_20220222.h5",
        }
    }

    pub fn variants<'a>() -> &'a [ModulationMode] {
        &[
            Self::QAM_64,
            Self::QPSK,
            Self::GMSK,
            Self::QAM_32,
            Self::PSK_16,
            Self::OOK,
            Self::FM,
            Self::APSK_32,
            Self::PSK_32,
            Self::AM_DSB_WC,
            Self::APSK_64,
            Self::OQPSK,
            Self::QAM_128,
            Self::ASK_4,
            Self::AM_SSB_SC,
            Self::AM_DSB_SC,
            Self::PSK_8,
            Self::ASK_8,
            Self::QAM_256,
            Self::APSK_128,
            Self::APSK_16,
            Self::OOK_2,
            Self::BPSK,
            Self::QAM_16,
            Self::AM_SSB_WC,
            Self::Noise_20220222,
        ]
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

        let sample_indices = {
            let mut indices =
                (0..4096).choose_multiple(&mut ChaCha8Rng::seed_from_u64(42), num_samples);
            indices.sort();
            indices
        };

        let levels = if matches!(modulation, ModulationMode::Noise_20220222) {
            assert_eq!(
                all_iq.shape(),
                [16_384, 1024, 2],
                "noise data had the wrong shape"
            );

            vec![SingleSnR {
                iq: all_iq,
                snr: -50,
            }
            .subsample(&sample_indices)]
        } else {
            assert_eq!(
                all_iq.shape(),
                [106_496, 1024, 2],
                "iq data had the wrong shape"
            );

            let mut all_iq = all_iq.view_mut();
            let mut levels = Vec::new();
            for snr in (-20..=30).step_by(2) {
                let (iq, rest) = all_iq.split_at(Axis(0), 4096);
                all_iq = rest;
                let iq = iq.to_owned();
                levels.push(SingleSnR { iq, snr }.subsample(&sample_indices));
            }

            levels
        };

        Ok(Self {
            modulation,
            levels,
            sample_indices,
        })
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

        if matches!(self.modulation, ModulationMode::Noise_20220222) {
            assert_eq!([self.sample_indices.len(), 1024, 2], iq.shape());
        } else {
            assert_eq!([26 * self.sample_indices.len(), 1024, 2], iq.shape());
        }
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
