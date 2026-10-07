//! Merge per-moment RMA BUFR files into one radish `VolumeData`.
//!
//! The SMN writes one file per moment, each holding every sweep of the scan.
//! Sweeps are matched across files by elevation (±0.1°, with repeated
//! elevations kept apart by their order of occurrence) and sorted by
//! elevation, as in the xradar backend.
//!
//! Geometry conventions (shared with xradar's `argentina_bufr` backend):
//!
//! * rows are stored in fixed physical order, ray `i` at `i · 360 / n` deg;
//!   the start azimuth only says where the antenna began;
//! * range of gate `j` is `bin_offset + j · bin_size`.
//!
//! Differences from xradar, deliberate:
//!
//! * ray times follow the rotation: the ray at the start azimuth gets the
//!   sweep start time and time increases clockwise from it, instead of a
//!   ramp in storage order (which put the earliest time at 0° north);
//! * moments of one sweep that disagree on gate spacing or offset (e.g. a
//!   long-range DBZH beside short-range Doppler moments) are placed on the
//!   finest common range axis by nearest gate instead of being padded
//!   against the first moment's spacing; the sweep is flagged in the
//!   volume attributes when that happens;
//! * a sweep at ≥ 88° elevation (the RMA birdbath) is `vertical_pointing`.

use std::collections::{BTreeMap, HashMap};

use chrono::{DateTime, Utc};
use ndarray::Array2;

use super::decode::{DecodedFile, DecodedSweep};
use crate::backends::common::meta_for;
use crate::{
    Coordinates, MomentData, RadishError, Result, SweepData, SweepMetadata, VolumeData,
    VolumeMetadata,
};
use radish_types::{PlatformType, SweepMode};

/// Elevation tolerance for matching one sweep across moment files.
const ELEV_TOL_DEG: f64 = 0.1;
/// Elevation at or above which a sweep is treated as vertically pointing.
const BIRDBATH_MIN_ELEV_DEG: f64 = 88.0;

/// Parsed `<RADAR>_<task>_<subtask>_<MOMENT>_<YYYYMMDDTHHMMSS>Z.BUFR[.gz]`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FileKey {
    pub radar: String,
    pub task: String,
    pub subtask: String,
    pub moment: String,
    pub time: String,
}

impl FileKey {
    /// The volume a file belongs to: everything but the moment.
    pub fn volume(&self) -> (String, String, String, String) {
        (
            self.radar.clone(),
            self.task.clone(),
            self.subtask.clone(),
            self.time.clone(),
        )
    }
}

/// Parse an RMA BUFR basename; `None` if it does not follow the SMN pattern.
pub fn parse_file_name(name: &str) -> Option<FileKey> {
    let lower = name.to_ascii_lowercase();
    let stem_len = if lower.ends_with(".bufr.gz") {
        name.len() - 8
    } else if lower.ends_with(".bufr") {
        name.len() - 5
    } else {
        return None;
    };
    let parts: Vec<&str> = name[..stem_len].split('_').collect();
    if parts.len() != 5 {
        return None;
    }
    let t = parts[4];
    let ok_time = t.len() == 16
        && t.ends_with('Z')
        && t.as_bytes()[8] == b'T'
        && t[..8].bytes().all(|b| b.is_ascii_digit())
        && t[9..15].bytes().all(|b| b.is_ascii_digit());
    if !ok_time
        || !parts[1].bytes().all(|b| b.is_ascii_digit())
        || !parts[2].bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some(FileKey {
        radar: parts[0].to_string(),
        task: parts[1].to_string(),
        subtask: parts[2].to_string(),
        moment: parts[3].to_string(),
        time: t.to_string(),
    })
}

/// One decoded input file plus where it came from.
pub(crate) struct Source {
    pub label: String,
    pub key: Option<FileKey>,
    pub file: DecodedFile,
}

/// (source index, sweep index within that file, moment index within that sweep)
type MomentRef = (usize, usize, usize);

struct SweepGroup {
    elev: f64,
    occurrence: usize,
    moments: BTreeMap<String, MomentRef>,
}

fn to_secs(t: &DateTime<Utc>) -> f64 {
    t.timestamp() as f64 + t.timestamp_subsec_micros() as f64 / 1e6
}

/// CF metadata for moments the shared ODIM table lacks.
fn fallback_meta(name: &str) -> (&'static str, Option<&'static str>, &'static str) {
    match name {
        "DBZV" => (
            "dBZ",
            Some("radar_equivalent_reflectivity_factor_v"),
            "Equivalent reflectivity factor V",
        ),
        "DBTV" => (
            "dBZ",
            Some("radar_equivalent_reflectivity_factor_v"),
            "Total power V (uncorrected reflectivity)",
        ),
        "TDR" => ("dB", None, "Total (unfiltered) differential reflectivity"),
        "CM" => ("1", None, "Clutter map / clutter flag"),
        _ => ("", None, "Unknown RMA product"),
    }
}

fn make_moment(name: &str, data: Array2<f32>) -> MomentData {
    let mut m = match meta_for(name) {
        Some(meta) => {
            let mut m = MomentData::new(name.to_string(), meta.units.to_string(), data);
            m.standard_name = Some(meta.standard_name.to_string());
            m.long_name = Some(meta.long_name.to_string());
            m
        }
        None => {
            let (units, std, long) = fallback_meta(name);
            let mut m = MomentData::new(name.to_string(), units.to_string(), data);
            m.standard_name = std.map(str::to_string);
            m.long_name = Some(long.to_string());
            m
        }
    };
    m.fill_value = Some(f32::NAN);
    m.scale_factor = Some(1.0);
    m.add_offset = Some(0.0);
    m
}

/// Merge decoded files (one volume) into `VolumeData`.
pub(crate) fn merge(sources: Vec<Source>) -> Result<VolumeData> {
    if sources.is_empty() {
        return Err(RadishError::InvalidFormat(
            "argentina_bufr: no input files".into(),
        ));
    }
    let first = &sources[0].file;
    // all files must come from one radar and one scan
    for s in &sources[1..] {
        if s.file.instrument() != first.instrument() {
            return Err(RadishError::InvalidFormat(format!(
                "argentina_bufr: files from different radars ({} vs {} in {})",
                first.instrument(),
                s.file.instrument(),
                s.label
            )));
        }
    }
    let volumes: std::collections::BTreeSet<_> = sources
        .iter()
        .filter_map(|s| s.key.as_ref().map(FileKey::volume))
        .collect();
    if volumes.len() > 1 {
        return Err(RadishError::InvalidFormat(format!(
            "argentina_bufr: inputs span {} volumes ({}); group them first (radish.argentina_bufr_groups / read_volumes)",
            volumes.len(),
            volumes.iter().map(|v| format!("{}_{}_{}@{}", v.0, v.1, v.2, v.3)).collect::<Vec<_>>().join(", ")
        )));
    }

    // ---- group sweeps across files
    let mut groups: Vec<SweepGroup> = Vec::new();
    let mut duplicates: Vec<String> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for (fi, src) in sources.iter().enumerate() {
        let mut seen: Vec<(f64, usize)> = Vec::new(); // elevation → occurrences in this file
        for (si, sw) in src.file.sweeps.iter().enumerate() {
            let occ = match seen
                .iter_mut()
                .find(|(e, _)| (e - sw.elevation_deg).abs() <= ELEV_TOL_DEG)
            {
                Some((_, n)) => {
                    *n += 1;
                    *n - 1
                }
                None => {
                    seen.push((sw.elevation_deg, 1));
                    0
                }
            };
            for (name, why) in &sw.skipped {
                skipped.push(format!("{name}@{:.2}: {why}", sw.elevation_deg));
            }
            let gi = match groups.iter().position(|g| {
                g.occurrence == occ && (g.elev - sw.elevation_deg).abs() <= ELEV_TOL_DEG
            }) {
                Some(gi) => gi,
                None => {
                    groups.push(SweepGroup {
                        elev: sw.elevation_deg,
                        occurrence: occ,
                        moments: BTreeMap::new(),
                    });
                    groups.len() - 1
                }
            };
            for (mi, m) in sw.moments.iter().enumerate() {
                if groups[gi].moments.contains_key(&m.name) {
                    duplicates.push(format!("{}@{:.2}", m.name, sw.elevation_deg));
                } else {
                    groups[gi].moments.insert(m.name.clone(), (fi, si, mi));
                }
            }
        }
    }
    groups.retain(|g| !g.moments.is_empty());
    if groups.is_empty() {
        return Err(RadishError::InvalidFormat(
            "argentina_bufr: no decodable sweeps".into(),
        ));
    }
    let sweep_of = |r: &MomentRef| -> &DecodedSweep { &sources[r.0].file.sweeps[r.1] };
    groups.sort_by(|a, b| {
        a.elev
            .partial_cmp(&b.elev)
            .unwrap()
            .then(a.occurrence.cmp(&b.occurrence))
    });

    // ---- build sweeps
    let mut regridded: Vec<String> = Vec::new();
    let mut sweeps = Vec::with_capacity(groups.len());
    let (mut t_min, mut t_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut dt_min, mut dt_max): (Option<DateTime<Utc>>, Option<DateTime<Utc>>) = (None, None);
    for (idx, g) in groups.iter().enumerate() {
        // reference geometry: DBZH if present, else the first moment
        let ref_m = g
            .moments
            .get("DBZH")
            .or_else(|| g.moments.values().next())
            .unwrap();
        let rs = sweep_of(ref_m);
        let n_az = rs.n_azimuths;
        // range axis: shared spacing → pad to the longest; otherwise the
        // finest spacing over the union extent, nearest-gate placement
        let geoms: Vec<(f64, f64, usize)> = g
            .moments
            .values()
            .map(|r| {
                let s = sweep_of(r);
                (s.bin_offset_m, s.bin_size_m, s.n_bins)
            })
            .collect();
        let same = geoms
            .iter()
            .all(|&(o, d, _)| o == geoms[0].0 && d == geoms[0].1);
        let (off, dr, ng) = if same {
            (
                geoms[0].0,
                geoms[0].1,
                geoms.iter().map(|g| g.2).max().unwrap(),
            )
        } else {
            let dr = geoms.iter().map(|g| g.1).fold(f64::INFINITY, f64::min);
            let off = geoms.iter().map(|g| g.0).fold(f64::INFINITY, f64::min);
            let far = geoms
                .iter()
                .map(|g| g.0 + (g.2 as f64 - 1.0) * g.1)
                .fold(0.0, f64::max);
            regridded.push(format!("sweep_{idx}"));
            (off, dr, ((far - off) / dr).round() as usize + 1)
        };
        let range: Vec<f32> = (0..ng).map(|j| (off + j as f64 * dr) as f32).collect();
        let step = 360.0 / n_az as f64;
        let azimuth: Vec<f32> = (0..n_az).map(|i| (i as f64 * step) as f32).collect();
        let elevation = vec![g.elev as f32; n_az];
        let (t0, t1) = (to_secs(&rs.start_time), to_secs(&rs.end_time));
        let time: Vec<f64> = (0..n_az)
            .map(|i| {
                let frac = ((i as f64 * step - rs.start_azimuth_deg).rem_euclid(360.0)) / 360.0;
                t0 + frac * (t1 - t0)
            })
            .collect();
        t_min = t_min.min(t0);
        t_max = t_max.max(t1);
        dt_min = Some(dt_min.map_or(rs.start_time, |d| d.min(rs.start_time)));
        dt_max = Some(dt_max.map_or(rs.end_time, |d| d.max(rs.end_time)));

        let mut moments: HashMap<String, MomentData> = HashMap::with_capacity(g.moments.len());
        for (name, r) in &g.moments {
            let s = sweep_of(r);
            let src = &s.moments[r.2].data;
            let mut out = Array2::<f32>::from_elem((n_az, ng), f32::NAN);
            // azimuth rows: identical counts copy straight; otherwise nearest row
            let row_of = |i: usize| -> usize {
                if s.n_azimuths == n_az {
                    i
                } else {
                    ((i as f64 * s.n_azimuths as f64 / n_az as f64).round() as usize) % s.n_azimuths
                }
            };
            let direct = s.bin_offset_m == off && s.bin_size_m == dr;
            for (i, mut row) in out.rows_mut().into_iter().enumerate() {
                let srow = &src[row_of(i) * s.n_bins..(row_of(i) + 1) * s.n_bins];
                if direct {
                    let n = s.n_bins.min(ng);
                    row.as_slice_mut().unwrap()[..n].copy_from_slice(&srow[..n]);
                } else {
                    for (j, v) in row.iter_mut().enumerate() {
                        let k = ((range[j] as f64 - s.bin_offset_m) / s.bin_size_m).round();
                        if k >= 0.0 && (k as usize) < s.n_bins {
                            *v = srow[k as usize];
                        }
                    }
                }
            }
            if s.n_azimuths != n_az {
                regridded.push(format!("sweep_{idx}:{name}:azimuth"));
            }
            moments.insert(name.clone(), make_moment(name, out));
        }

        let mode = if g.elev >= BIRDBATH_MIN_ELEV_DEG {
            SweepMode::VerticalPointing
        } else {
            SweepMode::Azimuth
        };
        let mut meta = SweepMetadata::new(idx as u32, mode, g.elev);
        meta.rays_are_indexed = Some(true);
        meta.ray_angle_resolution = Some(step);
        sweeps.push(SweepData::new(
            meta,
            moments,
            Coordinates::new(time, range, azimuth, elevation),
        ));
    }

    // ---- volume metadata
    let start = dt_min.unwrap();
    let end = dt_max.unwrap();
    let mut md = VolumeMetadata::new(
        first.instrument(),
        first.latitude,
        first.longitude,
        first.height_m,
        start,
        end,
    );
    md.institution = "SMN (SINARAME)".to_string();
    md.platform_type = Some(PlatformType::Fixed);
    md.generate_sweep_names(sweeps.len());
    md.sweep_fixed_angles = sweeps.iter().map(|s| s.metadata.fixed_angle).collect();
    let a = &mut md.attributes;
    a.insert(
        "source_format".into(),
        "BUFR edition 4 (SMN / SINARAME RMA, centre 41 local tables)".into(),
    );
    a.insert("station_type".into(), first.station_type.clone());
    a.insert(
        "originating_centre".into(),
        first.section1.originating_centre.to_string(),
    );
    a.insert(
        "originating_subcentre".into(),
        first.section1.originating_subcentre.to_string(),
    );
    a.insert(
        "bufr_data_category".into(),
        first.section1.data_category.to_string(),
    );
    a.insert(
        "bufr_table_versions".into(),
        format!(
            "master {} local {}",
            first.section1.master_table_version, first.section1.local_table_version
        ),
    );
    a.insert("bufr_descriptors".into(), first.descriptors.join(" "));
    a.insert(
        "wmo_block_station".into(),
        format!("{:02}{:03}", first.wmo_block, first.wmo_station),
    );
    if let Some(t) = first.section1.time {
        a.insert(
            "nominal_time".into(),
            t.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        );
    }
    if let Some(k) = sources.iter().find_map(|s| s.key.as_ref()) {
        a.insert("scan_task".into(), k.task.clone());
        a.insert("scan_subtask".into(), k.subtask.clone());
        a.insert("file_time".into(), k.time.clone());
    }
    a.insert(
        "source_files".into(),
        sources
            .iter()
            .map(|s| s.label.as_str())
            .collect::<Vec<_>>()
            .join(","),
    );
    if !regridded.is_empty() {
        a.insert("range_regridded".into(), regridded.join(","));
    }
    if !duplicates.is_empty() {
        a.insert("duplicate_moments_ignored".into(), duplicates.join(","));
    }
    if !skipped.is_empty() {
        a.insert("skipped_moments".into(), skipped.join("; "));
    }
    let _ = (t_min, t_max);
    Ok(VolumeData::new(md, sweeps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_name_parsing() {
        let k = parse_file_name("RMA1_0315_01_DBZH_20261007T180457Z.BUFR").unwrap();
        assert_eq!(
            (
                k.radar.as_str(),
                k.task.as_str(),
                k.subtask.as_str(),
                k.moment.as_str()
            ),
            ("RMA1", "0315", "01", "DBZH")
        );
        assert_eq!(k.time, "20261007T180457Z");
        let g = parse_file_name("RMA1_0301_02_VRAD_20181111T000457Z.BUFR.gz").unwrap();
        assert_eq!(
            g.volume(),
            (
                "RMA1".into(),
                "0301".into(),
                "02".into(),
                "20181111T000457Z".into()
            )
        );
        assert!(parse_file_name("KLOT20230629_180000_V06").is_none());
        assert!(parse_file_name("RMA1_DBZH.BUFR").is_none());
    }
}
