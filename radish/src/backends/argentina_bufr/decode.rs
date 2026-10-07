//! Bit-level decoding of SMN / SINARAME RMA radar BUFR (edition 4) messages.
//!
//! Ported from the Rust parser on the `feat/argentina-bufr-ingest` branch of
//! swnesbitt/xradar (`rust/src/bufr/`). The messages use Argentina's local
//! tables (originating centre 41, local table version 2); rather than
//! interpreting tables at run time, the fixed descriptor expansion is
//! hard-coded:
//!
//! ```text
//! 3-21-204  station identifiers   n(8) × [type CCITT 3 chars, id CCITT 16 chars]
//! 3-01-031  location / time       WMO block(7) station(10) type(2)
//!                                  year(12) month(4) day(6) hour(5) minute(6)
//!                                  lat(25, 1e-5°, ref -9e6) lon(26, 1e-5°, ref -1.8e7)
//!                                  height(15, m, ref -400)
//! 3-21-203  polar volume          n_sweeps(8) × [
//!     3-21-205 start + end time   2 × (year12 month4 day6 hour5 minute6 second6)
//!     0-30-196 product type(8)
//!     0-02-135 elevation(15, 0.01°, ref -9000)
//!     0-30-194 n_bins(12)
//!     0-21-201 bin size(14, m)
//!     0-21-203 bin offset(14, 10 m)
//!     0-30-195 n_azimuths(11)
//!     0-02-134 start azimuth(16, 0.01°)
//!     n_params(8) × [
//!         0-30-196 product type(8)
//!         3-21-206 compression(8) n_outer(16) × [n_inner(16) × byte(8)]
//!     ]
//! ]
//! ```
//!
//! Each parameter's bytes form one zlib stream that inflates to
//! `n_azimuths × n_bins` little-endian `f64` values in physical units.
//! Values with `|v| > 1e30` are the BUFR missing indicator → `NaN`.

use std::io::Read;

use chrono::{DateTime, TimeZone, Utc};
use flate2::read::{GzDecoder, ZlibDecoder};

use crate::{RadishError, Result};

/// Gzip member magic (the `.BUFR.gz` outer wrapper).
pub(crate) const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
/// BUFR section 0 magic.
pub(crate) const BUFR_MAGIC: &[u8; 4] = b"BUFR";

/// `|v|` above this is the BUFR / RMA missing-value indicator.
const MISSING_ABS: f64 = 1e30;

fn err(msg: impl Into<String>) -> RadishError {
    RadishError::Decode(format!("argentina_bufr: {}", msg.into()))
}

// ---------------------------------------------------------------------------
// Bit reader
// ---------------------------------------------------------------------------

/// MSB-first bit reader over a byte slice (BUFR section 4 packing).
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Absolute bit position from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn ensure(&self, nbits: usize) -> Result<()> {
        if self.pos + nbits > self.data.len() * 8 {
            return Err(err(format!(
                "section 4 exhausted: need {nbits} bits at bit {} of {}",
                self.pos,
                self.data.len() * 8
            )));
        }
        Ok(())
    }

    /// Read `n` (0–64) bits as an unsigned integer.
    pub(crate) fn read(&mut self, n: u32) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 64 {
            return Err(err(format!("cannot read {n} bits at once")));
        }
        let n = n as usize;
        self.ensure(n)?;
        let end = self.pos + n;
        let mut v: u64 = 0;
        let mut p = self.pos;
        while p < end {
            let byte = self.data[p / 8];
            let bit_in = p % 8;
            let take = (8 - bit_in).min(end - p);
            let shift = 8 - bit_in - take;
            let bits = (byte >> shift) & (((1u16 << take) - 1) as u8);
            v = (v << take) | bits as u64;
            p += take;
        }
        self.pos = end;
        Ok(v)
    }

    /// Append `n` 8-bit values to `out` (fast path for the compressed
    /// payload, which is millions of bytes on long-range sweeps).
    pub(crate) fn extend_bytes(&mut self, n: usize, out: &mut Vec<u8>) -> Result<()> {
        self.ensure(n * 8)?;
        let idx = self.pos / 8;
        let s = self.pos % 8;
        if s == 0 {
            out.extend_from_slice(&self.data[idx..idx + n]);
        } else {
            out.reserve(n);
            // `ensure` guarantees idx + n < len when s > 0.
            for i in 0..n {
                let b0 = self.data[idx + i];
                let b1 = self.data[idx + i + 1];
                out.push((b0 << s) | (b1 >> (8 - s)));
            }
        }
        self.pos += n * 8;
        Ok(())
    }

    /// CCITT IA5 (ASCII) string of `nchars` characters, trailing spaces / NULs trimmed.
    pub(crate) fn read_ascii(&mut self, nchars: usize) -> Result<String> {
        let mut bytes = Vec::with_capacity(nchars);
        self.extend_bytes(nchars, &mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes)
            .trim_end_matches([' ', '\0'])
            .trim_start_matches([' ', '\0'])
            .to_string())
    }
}

// ---------------------------------------------------------------------------
// Sections 0–3
// ---------------------------------------------------------------------------

fn u24(b: &[u8]) -> usize {
    ((b[0] as usize) << 16) | ((b[1] as usize) << 8) | b[2] as usize
}

/// Section 1 fields we keep.
#[derive(Debug, Clone)]
pub(crate) struct Section1 {
    pub originating_centre: u16,
    pub originating_subcentre: u16,
    pub data_category: u8,
    pub master_table_version: u8,
    pub local_table_version: u8,
    /// Typical (nominal) time of the message.
    pub time: Option<DateTime<Utc>>,
}

/// Located sections of one message.
pub(crate) struct Sections<'a> {
    pub section1: Section1,
    /// Section 3 descriptors as `F-XX-YYY` strings.
    pub descriptors: Vec<String>,
    /// Section 4 payload (after its 4-byte header).
    pub section4: &'a [u8],
}

/// Validate section 0 and walk sections 1–4.
pub(crate) fn locate_sections(data: &[u8]) -> Result<Sections<'_>> {
    if data.len() < 8 || &data[..4] != BUFR_MAGIC {
        return Err(err("missing 'BUFR' magic"));
    }
    let total = u24(&data[4..7]);
    let edition = data[7];
    if edition != 4 {
        return Err(err(format!(
            "unsupported BUFR edition {edition} (expected 4)"
        )));
    }
    if total > data.len() {
        return Err(err(format!(
            "message length {total} exceeds buffer {}",
            data.len()
        )));
    }
    let need = |end: usize, what: &str| -> Result<()> {
        if end > total {
            Err(err(format!(
                "truncated message: {what} ends at {end}, message is {total} bytes"
            )))
        } else {
            Ok(())
        }
    };
    // Section 1
    need(8 + 22, "section 1")?;
    let s1 = &data[8..];
    let s1_len = u24(s1);
    if s1_len < 22 {
        return Err(err(format!("section 1 length {s1_len} < 22")));
    }
    let year = ((s1[15] as i32) << 8) | s1[16] as i32;
    let section1 = Section1 {
        originating_centre: ((s1[4] as u16) << 8) | s1[5] as u16,
        originating_subcentre: ((s1[6] as u16) << 8) | s1[7] as u16,
        data_category: s1[10],
        master_table_version: s1[13],
        local_table_version: s1[14],
        time: Utc
            .with_ymd_and_hms(
                year,
                s1[17] as u32,
                s1[18] as u32,
                s1[19] as u32,
                s1[20] as u32,
                s1[21] as u32,
            )
            .single(),
    };
    let s1_end = 8 + s1_len;
    let has_s2 = s1[9] & 0x80 != 0;
    // Section 2 (optional)
    let s2_end = if has_s2 {
        need(s1_end + 3, "section 2 header")?;
        s1_end + u24(&data[s1_end..])
    } else {
        s1_end
    };
    // Section 3
    need(s2_end + 7, "section 3 header")?;
    let s3_len = u24(&data[s2_end..]);
    let s3_end = s2_end + s3_len;
    need(s3_end, "section 3")?;
    let descriptors = data[s2_end + 7..s3_end]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|d| {
            let f = d[0] >> 6;
            let x = d[0] & 0x3f;
            let y = d[1];
            format!("{f}-{x:02}-{y:03}")
        })
        .collect();
    // Section 4
    need(s3_end + 4, "section 4 header")?;
    let s4_len = u24(&data[s3_end..]);
    if s4_len < 4 {
        return Err(err(format!("section 4 length {s4_len} < 4")));
    }
    need(s3_end + s4_len, "section 4")?;
    let section4 = &data[s3_end + 4..s3_end + s4_len];
    Ok(Sections {
        section1,
        descriptors,
        section4,
    })
}

// ---------------------------------------------------------------------------
// Product codes and element decoders (Argentine local table B / code tables)
// ---------------------------------------------------------------------------

/// RMA product-type code (0-30-196) → FM301 / ODIM moment name.
pub(crate) fn moment_name(code: u8) -> String {
    match code {
        0 => "DBZH",
        40 => "VRADH",
        60 | 92 => "WRADH",
        80 => "ZDR",
        91 | 230 => "DBTH",
        231 => "DBTV",
        232 | 233 => "DBZV",
        239 => "PHIDP",
        240 => "KDP",
        241 => "RHOHV",
        242 => "TDR",
        243 => "CM",
        other => return format!("UNKNOWN_{other}"),
    }
    .to_string()
}

fn elevation_deg(raw: u64) -> f64 {
    (raw as i64 - 9000) as f64 / 100.0
}
fn azimuth_deg(raw: u64) -> f64 {
    raw as f64 / 100.0
}
fn latitude_deg(raw: u64) -> f64 {
    (raw as i64 - 9_000_000) as f64 / 100_000.0
}
fn longitude_deg(raw: u64) -> f64 {
    (raw as i64 - 18_000_000) as f64 / 100_000.0
}
fn height_m(raw: u64) -> f64 {
    (raw as i64 - 400) as f64
}

// ---------------------------------------------------------------------------
// Section 4
// ---------------------------------------------------------------------------

/// One parameter (moment) of one sweep.
#[derive(Debug, Clone)]
pub(crate) struct DecodedMoment {
    #[allow(dead_code)] // kept for diagnostics / future product-specific handling
    pub product_code: u8,
    pub name: String,
    /// Row-major `n_azimuths × n_bins`, NaN where missing.
    pub data: Vec<f32>,
}

/// One sweep of one file.
#[derive(Debug, Clone)]
pub(crate) struct DecodedSweep {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub elevation_deg: f64,
    pub n_bins: usize,
    pub bin_size_m: f64,
    /// Range to the first bin (0-21-203), m.
    pub bin_offset_m: f64,
    pub n_azimuths: usize,
    /// Azimuth where the antenna began the sweep. Rows are stored in fixed
    /// 0°→360° order regardless; this only dates the rays.
    pub start_azimuth_deg: f64,
    pub moments: Vec<DecodedMoment>,
    /// Moments present in the file but undecodable (name, reason).
    pub skipped: Vec<(String, String)>,
}

/// A fully decoded message (file).
#[derive(Debug, Clone)]
pub(crate) struct DecodedFile {
    pub section1: Section1,
    pub descriptors: Vec<String>,
    pub station_type: String,
    pub station_id: String,
    pub wmo_block: u8,
    pub wmo_station: u16,
    pub latitude: f64,
    pub longitude: f64,
    pub height_m: f64,
    pub sweeps: Vec<DecodedSweep>,
}

impl DecodedFile {
    /// Radar name as the SMN uses it: station type + number ("RMA" + "1").
    pub(crate) fn instrument(&self) -> String {
        format!("{}{}", self.station_type, self.station_id)
    }
}

/// Inflate the outer gzip wrapper when present.
pub(crate) fn unwrap_gzip(raw: &[u8]) -> Result<std::borrow::Cow<'_, [u8]>> {
    if raw.len() >= 2 && raw[..2] == GZIP_MAGIC {
        let mut out = Vec::with_capacity(raw.len() * 4);
        GzDecoder::new(raw)
            .read_to_end(&mut out)
            .map_err(|e| err(format!("gzip: {e}")))?;
        Ok(std::borrow::Cow::Owned(out))
    } else {
        Ok(std::borrow::Cow::Borrowed(raw))
    }
}

fn read_time(r: &mut BitReader<'_>) -> Result<DateTime<Utc>> {
    let year = r.read(12)? as i32;
    let month = r.read(4)? as u32;
    let day = r.read(6)? as u32;
    let hour = r.read(5)? as u32;
    let minute = r.read(6)? as u32;
    let second = r.read(6)? as u32;
    Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
        .single()
        .ok_or_else(|| err(format!("invalid sweep time {year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}")))
}

fn inflate_moment(compressed: &[u8], expected: usize) -> std::result::Result<Vec<f32>, String> {
    let mut raw = Vec::with_capacity(expected * 8);
    ZlibDecoder::new(compressed)
        .read_to_end(&mut raw)
        .map_err(|e| format!("zlib: {e}"))?;
    if raw.len() < expected * 8 {
        return Err(format!(
            "inflated {} bytes, expected {}",
            raw.len(),
            expected * 8
        ));
    }
    Ok(raw[..expected * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            let v = f64::from_le_bytes(*c);
            if !v.is_finite() || v.abs() > MISSING_ABS {
                f32::NAN
            } else {
                v as f32
            }
        })
        .collect())
}

/// Station block (3-21-204) and location/time block (3-01-031).
struct Header {
    station_type: String,
    station_id: String,
    wmo_block: u8,
    wmo_station: u16,
    latitude: f64,
    longitude: f64,
    height_m: f64,
}

fn read_header(r: &mut BitReader<'_>) -> Result<Header> {
    let n_stations = r.read(8)? as usize;
    let mut station_type = String::new();
    let mut station_id = String::new();
    for _ in 0..n_stations {
        station_type = r.read_ascii(3)?;
        station_id = r.read_ascii(16)?;
    }
    let wmo_block = r.read(7)? as u8;
    let wmo_station = r.read(10)? as u16;
    r.read(2)?; // station type
    r.read(12 + 4 + 6 + 5 + 6)?; // year month day hour minute (section 1 has the full time)
    let latitude = latitude_deg(r.read(25)?);
    let longitude = longitude_deg(r.read(26)?);
    let height_m = height_m(r.read(15)?);
    Ok(Header {
        station_type,
        station_id,
        wmo_block,
        wmo_station,
        latitude,
        longitude,
        height_m,
    })
}

/// Decode one message. `want` filters moments by name before inflating
/// (`None` = all); filtered moments are still skipped over in the bitstream.
pub(crate) fn decode_message(raw: &[u8], want: Option<&[&str]>) -> Result<DecodedFile> {
    let data = unwrap_gzip(raw)?;
    let sections = locate_sections(&data)?;
    let mut r = BitReader::new(sections.section4);
    let h = read_header(&mut r)?;

    let n_sweeps = r.read(8)? as usize;
    let mut sweeps = Vec::with_capacity(n_sweeps);
    for si in 0..n_sweeps {
        let start_time = read_time(&mut r)?;
        let end_time = read_time(&mut r)?;
        r.read(8)?; // sweep-level product type
        let elevation_deg = elevation_deg(r.read(15)?);
        let n_bins = r.read(12)? as usize;
        let bin_size_m = r.read(14)? as f64;
        let bin_offset_m = r.read(14)? as f64 * 10.0;
        let n_azimuths = r.read(11)? as usize;
        let start_azimuth_deg = azimuth_deg(r.read(16)?);
        if n_bins == 0 || n_azimuths == 0 {
            return Err(err(format!(
                "sweep {si}: empty geometry ({n_azimuths} azimuths × {n_bins} bins)"
            )));
        }
        let n_params = r.read(8)? as usize;
        let mut moments = Vec::with_capacity(n_params);
        let mut skipped = Vec::new();
        let mut buf = Vec::new();
        for _ in 0..n_params {
            let product_code = r.read(8)? as u8;
            let name = moment_name(product_code);
            let compression = r.read(8)?;
            let n_outer = r.read(16)? as usize;
            buf.clear();
            for _ in 0..n_outer {
                let n_inner = r.read(16)? as usize;
                r.extend_bytes(n_inner, &mut buf)?;
            }
            if want.is_some_and(|w| !w.contains(&name.as_str())) {
                continue;
            }
            if compression != 0 {
                skipped.push((
                    name,
                    format!("unsupported compression method {compression}"),
                ));
                continue;
            }
            match inflate_moment(&buf, n_azimuths * n_bins) {
                Ok(data) => moments.push(DecodedMoment {
                    product_code,
                    name,
                    data,
                }),
                Err(e) => skipped.push((name, e)),
            }
        }
        sweeps.push(DecodedSweep {
            start_time,
            end_time,
            elevation_deg,
            n_bins,
            bin_size_m,
            bin_offset_m,
            n_azimuths,
            start_azimuth_deg,
            moments,
            skipped,
        });
    }

    Ok(DecodedFile {
        section1: sections.section1,
        descriptors: sections.descriptors,
        station_type: h.station_type,
        station_id: h.station_id,
        wmo_block: h.wmo_block,
        wmo_station: h.wmo_station,
        latitude: h.latitude,
        longitude: h.longitude,
        height_m: h.height_m,
        sweeps,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use flate2::write::{GzEncoder, ZlibEncoder};
    use flate2::Compression;
    use std::io::Write;

    /// MSB-first bit writer — test-only encoder used to build synthetic
    /// RMA messages with arbitrary geometry.
    #[derive(Default)]
    pub(crate) struct BitWriter {
        pub bytes: Vec<u8>,
        nbits: usize,
    }

    impl BitWriter {
        pub(crate) fn put(&mut self, v: u64, n: u32) {
            for i in (0..n).rev() {
                let bit = ((v >> i) & 1) as u8;
                if self.nbits.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let last = self.bytes.last_mut().unwrap();
                *last |= bit << (7 - self.nbits % 8);
                self.nbits += 1;
            }
        }
        fn put_ascii(&mut self, s: &str, n: usize) {
            let b = s.as_bytes();
            for i in 0..n {
                self.put(*b.get(i).unwrap_or(&b' ') as u64, 8);
            }
        }
        fn put_time(&mut self, t: (u64, u64, u64, u64, u64, u64)) {
            self.put(t.0, 12);
            self.put(t.1, 4);
            self.put(t.2, 6);
            self.put(t.3, 5);
            self.put(t.4, 6);
            self.put(t.5, 6);
        }
    }

    /// Product code and value generator `(ray, bin) -> value`.
    pub(crate) type Param = (u8, fn(usize, usize) -> f64);

    /// Geometry + data for one synthetic sweep.
    pub(crate) struct SynthSweep {
        pub elevation: f64,
        pub n_bins: usize,
        pub bin_size: u64,
        pub bin_offset_10m: u64,
        pub n_az: usize,
        pub start_az: f64,
        pub start: (u64, u64, u64, u64, u64, u64),
        pub end: (u64, u64, u64, u64, u64, u64),
        /// (product code, value function of (ray, bin))
        pub params: Vec<Param>,
    }

    /// Build a complete edition-4 message (optionally gzip-wrapped). The
    /// section 4 bitstream is deliberately not byte-aligned before the
    /// payload so the unaligned `extend_bytes` path is exercised.
    pub(crate) fn synth_message(
        station: &str,
        lat: f64,
        lon: f64,
        h: f64,
        sweeps: &[SynthSweep],
        gzip: bool,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.put(1, 8);
        // station type + number: "RMA1" → "RMA" / "1", "AR5" → "AR" / "5"
        let split = station
            .find(|c: char| c.is_ascii_digit())
            .unwrap_or(station.len());
        w.put_ascii(&station[..split], 3);
        w.put_ascii(&station[split..], 16);
        w.put(87, 7);
        w.put(344, 10);
        w.put(0, 2);
        w.put(2026, 12);
        w.put(10, 4);
        w.put(7, 6);
        w.put(18, 5);
        w.put(0, 6);
        w.put(((lat * 1e5).round() as i64 + 9_000_000) as u64, 25);
        w.put(((lon * 1e5).round() as i64 + 18_000_000) as u64, 26);
        w.put((h as i64 + 400) as u64, 15);
        w.put(sweeps.len() as u64, 8);
        for s in sweeps {
            w.put_time(s.start);
            w.put_time(s.end);
            w.put(s.params[0].0 as u64, 8);
            w.put(((s.elevation * 100.0).round() as i64 + 9000) as u64, 15);
            w.put(s.n_bins as u64, 12);
            w.put(s.bin_size, 14);
            w.put(s.bin_offset_10m, 14);
            w.put(s.n_az as u64, 11);
            w.put((s.start_az * 100.0).round() as u64, 16);
            w.put(s.params.len() as u64, 8);
            for (code, f) in &s.params {
                let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
                for ray in 0..s.n_az {
                    for bin in 0..s.n_bins {
                        z.write_all(&f(ray, bin).to_le_bytes()).unwrap();
                    }
                }
                let comp = z.finish().unwrap();
                w.put(*code as u64, 8);
                w.put(0, 8);
                // the inner byte count is 16 bits: blocks of ≤ 65535 bytes,
                // and at least two to exercise the nested replication
                let block = (comp.len() / 2 + 1).min(65_535);
                let chunks: Vec<&[u8]> = comp.chunks(block).collect();
                w.put(chunks.len() as u64, 16);
                for c in chunks {
                    w.put(c.len() as u64, 16);
                    for b in c {
                        w.put(*b as u64, 8);
                    }
                }
            }
        }
        let s4 = w.bytes;
        let mut msg = Vec::new();
        let s1: Vec<u8> = vec![
            0, 0, 22, 0, 0, 41, 0, 0, 0, 0, 6, 2, 0, 16, 2, 0x07, 0xEA, 10, 7, 18, 0, 0,
        ];
        // section 3: length 9, reserved, 1 subset, flags, one descriptor 3-21-203
        let s3: Vec<u8> = vec![0, 0, 9, 0, 0, 1, 0x80, 0xD5, 0xCB];
        let s4_len = s4.len() + 4;
        let total = 8 + s1.len() + s3.len() + s4_len + 4;
        msg.extend_from_slice(b"BUFR");
        msg.extend_from_slice(&[(total >> 16) as u8, (total >> 8) as u8, total as u8, 4]);
        msg.extend_from_slice(&s1);
        msg.extend_from_slice(&s3);
        msg.extend_from_slice(&[(s4_len >> 16) as u8, (s4_len >> 8) as u8, s4_len as u8, 0]);
        msg.extend_from_slice(&s4);
        msg.extend_from_slice(b"7777");
        if gzip {
            let mut g = GzEncoder::new(Vec::new(), Compression::fast());
            g.write_all(&msg).unwrap();
            g.finish().unwrap()
        } else {
            msg
        }
    }

    pub(crate) fn simple_sweep(
        elev: f64,
        n_bins: usize,
        bin_size: u64,
        codes: &[Param],
    ) -> SynthSweep {
        SynthSweep {
            elevation: elev,
            n_bins,
            bin_size,
            bin_offset_10m: 210,
            n_az: 360,
            start_az: 90.0,
            start: (2026, 10, 7, 18, 0, 0),
            end: (2026, 10, 7, 18, 0, 36),
            params: codes.to_vec(),
        }
    }

    #[test]
    fn bit_reader_reads_across_byte_boundaries() {
        let data = [0b1010_1010u8, 0b1111_0000];
        let mut r = BitReader::new(&data);
        assert_eq!(r.read(4).unwrap(), 0b1010);
        assert_eq!(r.read(8).unwrap(), 0b1010_1111);
        assert_eq!(r.read(4).unwrap(), 0);
        assert!(r.read(1).is_err());
    }

    #[test]
    fn extend_bytes_unaligned_matches_bitwise_reads() {
        let data: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37)).collect();
        for skip in 0..8u32 {
            let mut a = BitReader::new(&data);
            a.read(skip).unwrap();
            let mut fast = Vec::new();
            a.extend_bytes(20, &mut fast).unwrap();
            let mut b = BitReader::new(&data);
            b.read(skip).unwrap();
            let slow: Vec<u8> = (0..20).map(|_| b.read(8).unwrap() as u8).collect();
            assert_eq!(fast, slow, "skip {skip}");
        }
    }

    #[test]
    fn element_decoders_match_rma1_values() {
        assert!((elevation_deg(9051) - 0.51).abs() < 1e-9);
        assert!((azimuth_deg(1300) - 13.0).abs() < 1e-9);
        assert!((latitude_deg(5_855_870) + 31.4413).abs() < 1e-6);
        assert!((longitude_deg(11_580_808) + 64.19192).abs() < 1e-6);
        assert!((height_m(884) - 484.0).abs() < 1e-9);
        assert_eq!(moment_name(0), "DBZH");
        assert_eq!(moment_name(40), "VRADH");
        assert_eq!(moment_name(230), "DBTH");
        assert_eq!(moment_name(99), "UNKNOWN_99");
    }

    #[test]
    fn rejects_non_bufr_and_wrong_edition() {
        assert!(decode_message(b"GARBAGE!", None).is_err());
        assert!(decode_message(b"BUFR\x00\x00\x08\x03", None).is_err());
    }

    #[test]
    fn synthetic_message_round_trips_plain_and_gzip() {
        fn dbz(r: usize, b: usize) -> f64 {
            if b.is_multiple_of(7) {
                -1e38
            } else {
                (r as f64) * 0.1 + b as f64 * 0.01
            }
        }
        fn vel(r: usize, _b: usize) -> f64 {
            r as f64 / 36.0 - 5.0
        }
        for gz in [false, true] {
            let msg = synth_message(
                "RMA1",
                -31.44133,
                -64.19192,
                484.0,
                &[
                    simple_sweep(0.5, 50, 120, &[(0, dbz), (40, vel)]),
                    simple_sweep(1.3, 40, 450, &[(0, dbz)]),
                ],
                gz,
            );
            let f = decode_message(&msg, None).unwrap();
            assert_eq!(f.station_id, "1");
            assert_eq!(f.station_type, "RMA");
            assert_eq!(f.instrument(), "RMA1");
            assert_eq!(f.section1.originating_centre, 41);
            assert!((f.latitude + 31.44133).abs() < 1e-9);
            assert!((f.longitude + 64.19192).abs() < 1e-9);
            assert_eq!(f.height_m, 484.0);
            assert_eq!(f.sweeps.len(), 2);
            let s = &f.sweeps[0];
            assert_eq!(
                (s.n_azimuths, s.n_bins, s.bin_size_m, s.bin_offset_m),
                (360, 50, 120.0, 2100.0)
            );
            assert_eq!(s.moments.len(), 2);
            let z = &s.moments[0];
            assert_eq!(z.name, "DBZH");
            assert!(z.data[0].is_nan(), "missing indicator must become NaN");
            assert!((z.data[3 * 50 + 1] - 0.31).abs() < 1e-5);
            assert_eq!(s.moments[1].name, "VRADH");
            // moment filter skips inflating but keeps the bitstream aligned
            let f2 = decode_message(&msg, Some(&["VRADH"])).unwrap();
            assert_eq!(f2.sweeps[0].moments.len(), 1);
            assert_eq!(f2.sweeps[1].moments.len(), 0);
            assert_eq!(f2.sweeps[1].bin_size_m, 450.0);
        }
    }

    #[test]
    fn bad_payloads_are_errors_for_the_moment_only() {
        assert!(inflate_moment(b"not zlib data", 4).is_err());
        let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
        z.write_all(&1.0f64.to_le_bytes()).unwrap();
        let one = z.finish().unwrap();
        assert!(inflate_moment(&one, 2).unwrap_err().contains("expected 16"));
        assert_eq!(inflate_moment(&one, 1).unwrap(), vec![1.0f32]);
    }
}
