//! SMN / SINARAME (Argentina) RMA radar BUFR backend.
//!
//! The Argentine national network (RMA radars, SMN FTP `/L2/<RADAR>/…`)
//! distributes Level 2 data as WMO BUFR edition 4 with Argentina's local
//! tables: **one file per moment**, each holding every sweep of one scan,
//! named `<RADAR>_<task>_<subtask>_<MOMENT>_<YYYYMMDDTHHMMSS>Z.BUFR[.gz]`.
//! A scan task cycles through subtasks (e.g. RMA1 task 0315: 01 15-tilt
//! dual-pol volume, 02 short-range Doppler, 03 birdbath, 04 long-range
//! surveillance), each a separate "volume" in radish terms.
//!
//! Module layout:
//!
//! * `decode.rs` — bit reader, sections 0–4, product codes, zlib payloads
//!   (ported from swnesbitt/xradar `feat/argentina-bufr-ingest`).
//! * `adapter.rs` — merges per-moment files into one `VolumeData`.
//! * `sniff.rs` — detection (`BUFR` magic, also inside gzip).
//!
//! Inputs: a single file (that moment only), a directory or list of files
//! of **one** scan (merged), or in-memory buffers. [`group_files`] /
//! [`read_volumes`] split a directory holding several scans.

use std::path::{Path, PathBuf};

use crate::backends::common::par_map;
use crate::{backends::RadarBackend, RadishError, Result, SweepData, VolumeData, VolumeMetadata};

mod adapter;
mod decode;
mod sniff;

pub use adapter::{parse_file_name, FileKey};

/// Backend for SMN / SINARAME RMA radar BUFR files.
pub struct ArgentinaBufrBackend;

impl ArgentinaBufrBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ArgentinaBufrBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Expand directories to their BUFR files (sorted). When both `X.BUFR` and
/// `X.BUFR.gz` exist, the compressed copy is kept.
pub fn expand_paths(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(p)?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|q| {
                    q.is_file()
                        && q.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(sniff::has_bufr_name)
                })
                .collect();
            files.sort();
            out.extend(files);
        } else {
            out.push(p.clone());
        }
    }
    // drop plain copies shadowed by a .gz twin
    let gz: std::collections::HashSet<String> = out
        .iter()
        .filter_map(|p| p.to_str())
        .filter(|s| s.to_ascii_lowercase().ends_with(".gz"))
        .map(|s| s[..s.len() - 3].to_string())
        .collect();
    out.retain(|p| p.to_str().is_none_or(|s| !gz.contains(s)));
    out.dedup();
    Ok(out)
}

fn label(p: &Path) -> String {
    p.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("<file>")
        .to_string()
}

fn decode_paths(files: &[PathBuf], moments: Option<&[&str]>) -> Result<Vec<adapter::Source>> {
    par_map(files, |p: &PathBuf| -> Result<adapter::Source> {
        let raw = std::fs::read(p)?;
        let file = decode::decode_message(&raw, moments)
            .map_err(|e| RadishError::Decode(format!("{}: {e}", p.display())))?;
        Ok(adapter::Source {
            label: label(p),
            key: parse_file_name(&label(p)),
            file,
        })
    })
}

/// Read one scan from files and/or directories (all moments merged).
pub fn read_files(paths: &[PathBuf], moments: Option<&[&str]>) -> Result<VolumeData> {
    let files = expand_paths(paths)?;
    if files.is_empty() {
        return Err(RadishError::InvalidFormat(
            "argentina_bufr: no .BUFR / .BUFR.gz files found".into(),
        ));
    }
    adapter::merge(decode_paths(&files, moments)?)
}

/// Read one scan from in-memory messages (one per moment file). `names`
/// (optional, same length) are the original basenames, used for the
/// task / subtask attributes and multi-scan checks.
pub fn read_buffers(
    buffers: Vec<Vec<u8>>,
    names: Option<Vec<String>>,
    moments: Option<&[&str]>,
) -> Result<VolumeData> {
    if buffers.is_empty() {
        return Err(RadishError::InvalidFormat(
            "argentina_bufr: no input buffers".into(),
        ));
    }
    let names = names.unwrap_or_default();
    let items: Vec<(usize, Vec<u8>)> = buffers.into_iter().enumerate().collect();
    let sources = par_map(
        &items,
        |(i, b): &(usize, Vec<u8>)| -> Result<adapter::Source> {
            let file = decode::decode_message(b, moments)?;
            let name = names.get(*i).cloned();
            Ok(adapter::Source {
                label: name.clone().unwrap_or_else(|| format!("<bytes {i}>")),
                key: name.as_deref().and_then(parse_file_name),
                file,
            })
        },
    )?;
    adapter::merge(sources)
}

/// Split files (or directories) into scans by the SMN file-name key
/// (radar, task, subtask, time), ordered by time then subtask. Files whose
/// names do not follow the pattern each form their own group.
pub fn group_files(paths: &[PathBuf]) -> Result<Vec<Vec<PathBuf>>> {
    let files = expand_paths(paths)?;
    let mut keyed: std::collections::BTreeMap<
        (String, String, String, String, String),
        Vec<PathBuf>,
    > = Default::default();
    for p in files {
        let k = match parse_file_name(&label(&p)) {
            Some(k) => (
                k.time.clone(),
                k.radar.clone(),
                k.task.clone(),
                k.subtask.clone(),
                String::new(),
            ),
            None => (
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                p.display().to_string(),
            ),
        };
        keyed.entry(k).or_default().push(p);
    }
    Ok(keyed.into_values().collect())
}

/// Read every scan found in `paths` (see [`group_files`]).
pub fn read_volumes(paths: &[PathBuf], moments: Option<&[&str]>) -> Result<Vec<VolumeData>> {
    group_files(paths)?
        .iter()
        .map(|g| read_files(g, moments))
        .collect()
}

impl RadarBackend for ArgentinaBufrBackend {
    fn name(&self) -> &str {
        "argentina_bufr"
    }

    fn description(&self) -> &str {
        "SMN / SINARAME (Argentina) RMA radar BUFR edition 4"
    }

    fn supported_extensions(&self) -> &[&str] {
        sniff::EXTENSIONS
    }

    fn can_read(&self, path: &Path) -> bool {
        sniff::looks_like_bufr_path(path)
    }

    fn can_read_bytes(&self, head: &[u8]) -> bool {
        sniff::looks_like_bufr_bytes(head)
    }

    /// A file reads that moment only; a directory reads (and merges) the
    /// single scan it holds — it is an error if it holds several.
    fn read_volume(&self, path: &Path) -> Result<VolumeData> {
        read_files(&[path.to_path_buf()], None)
    }

    fn read_bytes_volume(&self, data: Vec<u8>) -> Result<VolumeData> {
        read_buffers(vec![data], None, None)
    }

    fn scan_file(&self, path: &Path) -> Result<VolumeMetadata> {
        // Payloads are a few MB per moment; a full decode is cheap enough
        // that a header-only path is not worth a second code path yet.
        Ok(self.read_volume(path)?.metadata)
    }

    fn read_sweep(&self, path: &Path, sweep_idx: usize) -> Result<SweepData> {
        let mut v = self.read_volume(path)?;
        if sweep_idx >= v.sweeps.len() {
            return Err(RadishError::InvalidSweepIndex(sweep_idx));
        }
        Ok(v.sweeps.swap_remove(sweep_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::decode::tests::{simple_sweep, synth_message, SynthSweep};
    use super::*;
    use radish_types::SweepMode;

    fn dbz(r: usize, b: usize) -> f64 {
        (r % 60) as f64 + b as f64 * 0.01
    }
    fn vel(r: usize, b: usize) -> f64 {
        ((r + b) % 17) as f64 - 8.0
    }
    fn rho(_r: usize, _b: usize) -> f64 {
        0.98
    }

    fn write(dir: &Path, name: &str, msg: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, msg).unwrap();
        p
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("radish_arbufr_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn merges_moment_files_by_elevation_and_dates_rays_by_rotation() {
        let d = tmpdir("merge");
        let t = "20261007T180000Z";
        // DBZH file: three sweeps; VRAD file: two (no 2.0°); RHOHV gzip
        let z = synth_message(
            "RMA1",
            -31.44,
            -64.19,
            484.0,
            &[
                simple_sweep(0.5, 40, 120, &[(0, dbz)]),
                simple_sweep(1.3, 40, 120, &[(0, dbz)]),
                simple_sweep(2.0, 40, 120, &[(0, dbz)]),
            ],
            false,
        );
        let v = synth_message(
            "RMA1",
            -31.44,
            -64.19,
            484.0,
            &[
                simple_sweep(0.52, 40, 120, &[(40, vel)]),
                simple_sweep(1.31, 40, 120, &[(40, vel)]),
            ],
            false,
        );
        let r = synth_message(
            "RMA1",
            -31.44,
            -64.19,
            484.0,
            &[
                simple_sweep(2.0, 40, 120, &[(241, rho)]),
                simple_sweep(0.5, 40, 120, &[(241, rho)]),
            ],
            true,
        );
        write(&d, &format!("RMA1_0315_01_DBZH_{t}.BUFR"), &z);
        write(&d, &format!("RMA1_0315_01_VRAD_{t}.BUFR"), &v);
        write(&d, &format!("RMA1_0315_01_RHOHV_{t}.BUFR.gz"), &r);

        let b = ArgentinaBufrBackend::new();
        assert!(b.can_read(&d));
        let vol = b.read_volume(&d).unwrap();
        assert_eq!(vol.sweeps.len(), 3);
        assert_eq!(vol.metadata.sweep_fixed_angles, vec![0.5, 1.3, 2.0]);
        assert_eq!(vol.metadata.instrument_name, "RMA1");
        assert_eq!(vol.metadata.attributes["scan_task"], "0315");
        assert_eq!(vol.metadata.attributes["scan_subtask"], "01");
        let s0 = &vol.sweeps[0];
        let mut names: Vec<_> = s0.moments.keys().cloned().collect();
        names.sort();
        assert_eq!(names, ["DBZH", "RHOHV", "VRADH"]);
        assert_eq!(vol.sweeps[2].moments.len(), 2); // no VRADH at 2.0°
                                                    // ranges: offset 2100 m, 120 m gates
        assert_eq!(s0.coordinates.range[0], 2100.0);
        assert_eq!(s0.coordinates.range[1], 2220.0);
        // azimuths fixed 0..359, start azimuth 90° gets the sweep start time
        assert_eq!(s0.coordinates.azimuth[90], 90.0);
        let t0 = s0.coordinates.time[90];
        assert!(s0.coordinates.time.iter().all(|&x| x >= t0));
        assert!((s0.coordinates.time[89] - t0 - 36.0 * 359.0 / 360.0).abs() < 1e-6);
        // data copied in storage order
        let zz = &s0.moments["DBZH"].data;
        assert!((zz[[61, 3]] - (1.0 + 0.03)).abs() < 1e-5);
        assert_eq!(s0.moments["DBZH"].units, "dBZ");
        // single file → that moment only
        let one = b
            .read_volume(&d.join(format!("RMA1_0315_01_VRAD_{t}.BUFR")))
            .unwrap();
        assert_eq!(one.sweeps.len(), 2);
        assert_eq!(one.sweeps[0].moments.len(), 1);
    }

    #[test]
    fn regrids_moments_with_different_gate_spacing() {
        // long-range DBZH (240 m) beside short-range VRAD (120 m)
        let z = synth_message(
            "RMA1",
            -31.0,
            -64.0,
            400.0,
            &[simple_sweep(0.5, 30, 240, &[(0, dbz)])],
            false,
        );
        let v = synth_message(
            "RMA1",
            -31.0,
            -64.0,
            400.0,
            &[simple_sweep(0.5, 20, 120, &[(40, vel)])],
            false,
        );
        let vol = read_buffers(vec![z, v], None, None).unwrap();
        let s = &vol.sweeps[0];
        // finest spacing, union extent: 2100 .. 2100 + 29*240
        assert_eq!(s.coordinates.range[1] - s.coordinates.range[0], 120.0);
        assert_eq!(*s.coordinates.range.last().unwrap(), 2100.0 + 29.0 * 240.0);
        let zz = &s.moments["DBZH"].data;
        // gate j=2 at 2340 m is DBZH bin 1 exactly
        assert!((zz[[5, 2]] - (5.0 + 0.01)).abs() < 1e-5);
        let vv = &s.moments["VRADH"].data;
        assert!(vv[[0, 19]].is_finite() && vv[[0, 20]].is_nan());
        assert!(vol.metadata.attributes["range_regridded"].contains("sweep_0"));
    }

    #[test]
    fn birdbath_is_vertical_pointing_and_multi_scan_dirs_are_grouped() {
        let d = tmpdir("group");
        let bird = SynthSweep {
            elevation: 90.0,
            ..simple_sweep(90.0, 20, 60, &[(0, dbz)])
        };
        write(
            &d,
            "RMA1_0315_03_DBZH_20261007T180300Z.BUFR",
            &synth_message("RMA1", -31.0, -64.0, 400.0, &[bird], false),
        );
        write(
            &d,
            "RMA1_0315_04_DBZH_20261007T180400Z.BUFR",
            &synth_message(
                "RMA1",
                -31.0,
                -64.0,
                400.0,
                &[simple_sweep(0.5, 3700, 120, &[(0, dbz)])],
                false,
            ),
        );
        write(
            &d,
            "RMA1_0315_01_DBZH_20261007T180000Z.BUFR",
            &synth_message(
                "RMA1",
                -31.0,
                -64.0,
                400.0,
                &[simple_sweep(0.5, 10, 120, &[(0, dbz)])],
                false,
            ),
        );
        // a directory with several scans must not silently merge
        assert!(ArgentinaBufrBackend::new().read_volume(&d).is_err());
        let groups = group_files(std::slice::from_ref(&d)).unwrap();
        assert_eq!(groups.len(), 3);
        let vols = read_volumes(std::slice::from_ref(&d), None).unwrap();
        let subs: Vec<_> = vols
            .iter()
            .map(|v| v.metadata.attributes["scan_subtask"].clone())
            .collect();
        assert_eq!(subs, ["01", "03", "04"]);
        assert_eq!(
            vols[1].sweeps[0].metadata.sweep_mode,
            SweepMode::VerticalPointing
        );
        // long-range surveillance: 3700 bins → 446 km
        let r = &vols[2].sweeps[0].coordinates.range;
        assert!((r.last().unwrap() - (2100.0 + 3699.0 * 120.0)).abs() < 1e-3);
    }

    #[test]
    fn bytes_input_and_moment_filter() {
        let z = synth_message(
            "RMA1",
            -31.0,
            -64.0,
            400.0,
            &[simple_sweep(0.5, 10, 120, &[(0, dbz), (40, vel)])],
            true,
        );
        let b = ArgentinaBufrBackend::new();
        assert!(b.can_read_bytes(&z[..4096.min(z.len())]));
        let v = b.read_bytes_volume(z.clone()).unwrap();
        assert_eq!(v.sweeps[0].moments.len(), 2);
        let only = read_buffers(vec![z], None, Some(&["VRADH"])).unwrap();
        assert_eq!(only.sweeps[0].moments.keys().collect::<Vec<_>>(), ["VRADH"]);
    }
}
