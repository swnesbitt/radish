//! Format detection for SMN / SINARAME RMA BUFR files.
//!
//! Files are `<RADAR>_<task>_<subtask>_<MOMENT>_<YYYYMMDDTHHMMSS>Z.BUFR`,
//! optionally gzip-wrapped (`.BUFR.gz`). A gzip prefix alone is *not*
//! enough — NEXRAD Archive II also ships gzip-wrapped — so for gzip input
//! the head is inflated and checked for the `BUFR` magic. This backend is
//! registered ahead of NEXRAD in `available_backends()` for that reason.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use flate2::read::GzDecoder;

use super::decode::{BUFR_MAGIC, GZIP_MAGIC};

/// Extensions claimed outright (case-insensitive).
pub(crate) const EXTENSIONS: &[&str] = &["bufr"];

/// Bytes read from disk when sniffing a gzip file (enough compressed input
/// for the inflater to emit the first 4 bytes of the message).
const GZIP_PEEK_BYTES: usize = 4096;

/// True if `name` (a basename) ends with `.bufr` or `.bufr.gz`.
pub(crate) fn has_bufr_name(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.ends_with(".bufr") || n.ends_with(".bufr.gz")
}

/// Head-bytes check: raw `BUFR`, or a gzip stream whose inflated head is `BUFR`.
pub(crate) fn looks_like_bufr_bytes(head: &[u8]) -> bool {
    if head.starts_with(BUFR_MAGIC) {
        return true;
    }
    if head.len() >= 2 && head[..2] == GZIP_MAGIC {
        let mut out = [0u8; 4];
        return GzDecoder::new(head).read_exact(&mut out).is_ok() && &out == BUFR_MAGIC;
    }
    false
}

/// Path check: a directory holding BUFR-named files, a BUFR-named file, or
/// any file whose (inflated) head is a BUFR message.
pub(crate) fn looks_like_bufr_path(path: &Path) -> bool {
    if path.is_dir() {
        return std::fs::read_dir(path)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .any(|e| e.file_name().to_str().is_some_and(has_bufr_name))
            })
            .unwrap_or(false);
    }
    if path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(has_bufr_name)
    {
        return true;
    }
    let Ok(mut f) = File::open(path) else {
        return false;
    };
    let mut buf = vec![0u8; GZIP_PEEK_BYTES];
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(_) => return false,
        }
    }
    looks_like_bufr_bytes(&buf[..n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn names() {
        assert!(has_bufr_name("RMA1_0315_01_DBZH_20261007T180000Z.BUFR"));
        assert!(has_bufr_name("RMA1_0301_02_DBZH_20181111T000457Z.BUFR.gz"));
        assert!(!has_bufr_name("KLOT20230629_180000_V06.gz"));
    }

    #[test]
    fn bytes_raw_and_gzip() {
        assert!(looks_like_bufr_bytes(b"BUFR\x00\x10\x00\x04"));
        let mut g = GzEncoder::new(Vec::new(), Compression::default());
        g.write_all(b"BUFR\x00\x10\x00\x04rest-of-message").unwrap();
        assert!(looks_like_bufr_bytes(&g.finish().unwrap()));
        // gzip-wrapped NEXRAD must not be claimed
        let mut g = GzEncoder::new(Vec::new(), Compression::default());
        g.write_all(b"AR2V0006.001xxxxxxxxxxxx").unwrap();
        assert!(!looks_like_bufr_bytes(&g.finish().unwrap()));
        // a bare gzip magic with no stream is not BUFR either
        assert!(!looks_like_bufr_bytes(&[0x1f, 0x8b, 0x08, 0x00]));
        assert!(!looks_like_bufr_bytes(b"AR2V0006"));
    }
}
