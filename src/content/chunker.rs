use std::cmp::min;
use std::fmt::{self, Debug, Display, Formatter};
use std::io::{Result as IoResult, Seek, SeekFrom, Write};
use std::ptr;

use fastcdc::v2020::{FastCDC, Normalization};
use serde::{Deserialize, Serialize};

// taken from pcompress implementation
// https://github.com/moinakg/pcompress
const PRIME: u64 = 153_191u64;
const MASK: u64 = 0x00ff_ffff_ffffu64;
const MIN_SIZE: usize = 16 * 1024; // minimal chunk size, 16k
const AVG_SIZE: usize = 32 * 1024; // average chunk size, 32k
const MAX_SIZE: usize = 64 * 1024; // maximum chunk size, 64k

const FASTCDC_MIN_SIZE: usize = 2 * 1024;
const FASTCDC_AVG_SIZE: usize = 2 * 1024;
const FASTCDC_MAX_SIZE: usize = 64 * 1024;

const FIXED_SIZE: usize = 4 * 1024; // maximum chunk size, 64k

// Irreducible polynomial for Rabin modulus, from pcompress
const FP_POLY: u64 = 0xbfe6_b8a5_bf37_8d83u64;

// since we will skip MIN_SIZE when sliding window, it only
// needs to target (AVG_SIZE - MIN_SIZE) cut length,
// note the (AVG_SIZE - MIN_SIZE) must be 2^n
const CUT_MASK: u64 = (AVG_SIZE - MIN_SIZE - 1) as u64;

// rolling hash window constants
const WIN_SIZE: usize = 16; // must be 2^n
const WIN_MASK: usize = WIN_SIZE - 1;
const WIN_SLIDE_OFFSET: usize = 64;
const WIN_SLIDE_POS: usize = MIN_SIZE - WIN_SLIDE_OFFSET;

// writer buffer length
const WTR_BUF_LEN: usize = 8 * MAX_SIZE;

const LVLS: [Normalization; 4] = [
    Normalization::Level0,
    Normalization::Level1,
    Normalization::Level2,
    Normalization::Level3,
];

/// Pre-calculated chunker parameters
#[derive(Clone, Deserialize, Serialize)]
pub struct ChunkerParams {
    poly_pow: u64,     // poly power
    out_map: Vec<u64>, // pre-computed out byte map, length is 256
    ir: Vec<u64>,      // irreducible polynomial, length is 256
}

impl ChunkerParams {
    pub fn new() -> Self {
        let mut cp = ChunkerParams::default();

        // calculate poly power, it is actually PRIME ^ WIN_SIZE
        for _ in 0..WIN_SIZE {
            cp.poly_pow = (cp.poly_pow * PRIME) & MASK;
        }

        // pre-calculate out map table and irreducible polynomial
        // for each possible byte, copy from PCompress implementation
        for i in 0..256 {
            cp.out_map[i] = (i as u64 * cp.poly_pow) & MASK;

            let (mut term, mut pow, mut val) = (1u64, 1u64, 1u64);
            for _ in 0..WIN_SIZE {
                if (term & FP_POLY) != 0 {
                    val += (pow * i as u64) & MASK;
                }
                pow = (pow * PRIME) & MASK;
                term *= 2;
            }
            cp.ir[i] = val;
        }

        cp
    }
}

impl Debug for ChunkerParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChunkerParams()")
    }
}

impl Default for ChunkerParams {
    fn default() -> Self {
        let mut ret = ChunkerParams {
            poly_pow: 1,
            out_map: vec![0u64; 256],
            ir: vec![0u64; 256],
        };
        ret.out_map.shrink_to_fit();
        ret.ir.shrink_to_fit();
        ret
    }
}

/// Chunker
pub struct Chunker<W: Write + Seek> {
    dst: W,                // destination writer
    params: ChunkerParams, // chunker parameters
    pos: usize,
    chunk_len: usize,
    buf_clen: usize,
    win_idx: usize,
    roll_hash: u64,
    win: [u8; WIN_SIZE], // rolling hash circle window
    buf: Vec<u8>,        // chunker buffer, fixed size: WTR_BUF_LEN
}

impl<W: Write + Seek> Chunker<W> {
    pub fn new(params: ChunkerParams, dst: W) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        Chunker {
            dst,
            params,
            pos: WIN_SLIDE_POS,
            chunk_len: WIN_SLIDE_POS,
            buf_clen: 0,
            win_idx: 0,
            roll_hash: 0,
            win: [0u8; WIN_SIZE],
            buf,
        }
    }

    pub fn into_inner(mut self) -> IoResult<W> {
        self.flush()?;
        Ok(self.dst)
    }
}

impl<W: Write + Seek> Write for Chunker<W> {
    // consume bytes stream, output chunks
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // copy source data into chunker buffer
        let in_len = min(WTR_BUF_LEN - self.buf_clen, buf.len());
        assert!(in_len > 0);
        self.buf[self.buf_clen..self.buf_clen + in_len]
            .copy_from_slice(&buf[..in_len]);
        self.buf_clen += in_len;

        while self.pos < self.buf_clen {
            // get current byte and pushed out byte
            let ch = self.buf[self.pos];
            let out = self.win[self.win_idx] as usize;
            let pushed_out = self.params.out_map[out];

            // calculate Rabin rolling hash
            self.roll_hash = (self.roll_hash * PRIME) & MASK;
            self.roll_hash += u64::from(ch);
            self.roll_hash = self.roll_hash.wrapping_sub(pushed_out) & MASK;

            // forward circle window
            self.win[self.win_idx] = ch;
            self.win_idx = (self.win_idx + 1) & WIN_MASK;

            self.chunk_len += 1;
            self.pos += 1;

            if self.chunk_len >= MIN_SIZE {
                let chksum = self.roll_hash ^ self.params.ir[out];

                // reached cut point, chunk can be produced now
                if (chksum & CUT_MASK) == 0 || self.chunk_len >= MAX_SIZE {
                    // write the chunk to destination writer,
                    // ensure it is consumed in whole
                    let p = self.pos - self.chunk_len;
                    let written = self.dst.write(&self.buf[p..self.pos])?;
                    assert_eq!(written, self.chunk_len);

                    // not enough space in buffer, copy remaining to
                    // the head of buffer and reset buf position
                    if self.pos + MAX_SIZE >= WTR_BUF_LEN {
                        let left_len = self.buf_clen - self.pos;
                        unsafe {
                            ptr::copy::<u8>(
                                self.buf[self.pos..].as_ptr(),
                                self.buf.as_mut_ptr(),
                                left_len,
                            );
                        }
                        self.buf_clen = left_len;
                        self.pos = 0;
                    }

                    // jump to next start sliding position
                    self.pos += WIN_SLIDE_POS;
                    self.chunk_len = WIN_SLIDE_POS;
                }
            }
        }

        Ok(in_len)
    }

    fn flush(&mut self) -> IoResult<()> {
        // flush remaining data to destination
        let p = self.pos - self.chunk_len;
        if p < self.buf_clen {
            self.chunk_len = self.buf_clen - p;
            let _ = self.dst.write(&self.buf[p..(p + self.chunk_len)])?;
        }

        // reset chunker
        self.pos = WIN_SLIDE_POS;
        self.chunk_len = WIN_SLIDE_POS;
        self.buf_clen = 0;
        self.win_idx = 0;
        self.roll_hash = 0;
        self.win = [0u8; WIN_SIZE];

        self.dst.flush()
    }
}

impl<W: Write + Seek> Debug for Chunker<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Chunker()")
    }
}

impl<W: Write + Seek> Seek for Chunker<W> {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        self.dst.seek(pos)
    }
}

/// Chunker
pub struct FastCdcChunker<W: Write + Seek> {
    dst: W, // destination writer
    min_size: usize,
    avg_size: usize,
    max_size: usize,
    lvl: Normalization,
    pos: usize,
    chunk_len: usize,
    buf_clen: usize, // current length
    roll_hash: u64,
    buf: Vec<u8>, // chunker buffer, fixed size: WTR_BUF_LEN
}

impl<W: Write + Seek> FastCdcChunker<W> {
    pub fn new(dst: W) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        FastCdcChunker {
            dst,
            min_size: FASTCDC_MIN_SIZE,
            avg_size: FASTCDC_AVG_SIZE,
            max_size: FASTCDC_MAX_SIZE,
            lvl: Normalization::Level0,
            pos: FASTCDC_MIN_SIZE,
            chunk_len: FASTCDC_MIN_SIZE,
            buf_clen: 0,
            roll_hash: 0,
            buf,
        }
    }

    pub fn fine_tuned(
        dst: W,
        min_size: usize,
        avg_size: usize,
        max_size: usize,
        lvl: Normalization,
    ) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        FastCdcChunker {
            dst,
            min_size,
            avg_size,
            max_size,
            lvl,
            pos: min_size,
            chunk_len: min_size,
            buf_clen: 0,
            roll_hash: 0,
            buf,
        }
    }

    pub fn into_inner(mut self) -> IoResult<W> {
        self.flush()?;
        Ok(self.dst)
    }
}

impl<W: Write + Seek> Write for FastCdcChunker<W> {
    // consume bytes stream, output chunks
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // copy source data into chunker buffer
        let in_len = min(WTR_BUF_LEN - self.buf_clen, buf.len());
        assert!(in_len > 0);
        self.buf[self.buf_clen..self.buf_clen + in_len]
            .copy_from_slice(&buf[..in_len]);
        self.buf_clen += in_len;

        while self.pos < self.buf_clen {
            self.pos -= self.min_size;

            let (hash, cut_point) = FastCDC::with_level(
                &*self.buf,
                self.min_size as u32,
                self.avg_size as u32,
                self.max_size as u32,
                match self.lvl {
                    Normalization::Level0 => Normalization::Level0,
                    Normalization::Level1 => Normalization::Level1,
                    Normalization::Level2 => Normalization::Level2,
                    Normalization::Level3 => Normalization::Level3,
                },
            )
            .cut(self.pos, self.buf_clen - self.pos);

            self.roll_hash = hash;
            self.chunk_len = cut_point - self.pos;
            self.pos = cut_point;

            // write the chunk to destination writer,
            // ensure it is consumed in whole
            let p = self.pos - self.chunk_len;
            let written = self.dst.write(&self.buf[p..self.pos])?;
            assert_eq!(written, self.chunk_len);

            // not enough space in buffer, copy remaining to
            // the head of buffer and reset buf position
            if self.pos + self.max_size >= WTR_BUF_LEN {
                let left_len = self.buf_clen - self.pos;
                unsafe {
                    ptr::copy::<u8>(
                        self.buf[self.pos..].as_ptr(),
                        self.buf.as_mut_ptr(),
                        left_len,
                    );
                }
                self.buf_clen = left_len;
                self.pos = 0;
            }

            // jump to next start sliding position
            self.pos += self.min_size;
            self.chunk_len = self.min_size;
        }

        Ok(in_len)
    }

    fn flush(&mut self) -> IoResult<()> {
        // flush remaining data to destination
        let p = self.pos - self.chunk_len;
        if p < self.buf_clen {
            self.chunk_len = self.buf_clen - p;
            let _ = self.dst.write(&self.buf[p..(p + self.chunk_len)])?;
        }

        // reset chunker
        self.pos = self.min_size;
        self.chunk_len = self.min_size;
        self.buf_clen = 0;
        self.roll_hash = 0;

        self.dst.flush()
    }
}

impl<W: Write + Seek> Debug for FastCdcChunker<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Chunker()")
    }
}

impl<W: Write + Seek> Seek for FastCdcChunker<W> {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        self.dst.seek(pos)
    }
}

/// Chunker
pub struct FixedSizeChunker<W: Write + Seek> {
    dst: W, // destination writer
    chunk_size: usize,
    pos: usize,
    chunk_len: usize,
    buf_clen: usize, // current length
    buf: Vec<u8>,    // chunker buffer, fixed size: WTR_BUF_LEN
}

impl<W: Write + Seek> FixedSizeChunker<W> {
    pub fn new(dst: W) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        FixedSizeChunker {
            dst,
            chunk_size: FIXED_SIZE,
            pos: FIXED_SIZE,
            chunk_len: FIXED_SIZE,
            buf_clen: 0,
            buf,
        }
    }

    pub fn fine_tuned(dst: W, chunk_size: usize) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        FixedSizeChunker {
            dst,
            chunk_size,
            pos: chunk_size,
            chunk_len: chunk_size,
            buf_clen: 0,
            buf,
        }
    }

    pub fn into_inner(mut self) -> IoResult<W> {
        self.flush()?;
        Ok(self.dst)
    }
}

impl<W: Write + Seek> Write for FixedSizeChunker<W> {
    // consume bytes stream, output chunks
    fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        // copy source data into chunker buffer
        let in_len = min(WTR_BUF_LEN - self.buf_clen, buf.len());
        assert!(in_len > 0);
        self.buf[self.buf_clen..self.buf_clen + in_len]
            .copy_from_slice(&buf[..in_len]);
        self.buf_clen += in_len;

        while self.pos < self.buf_clen {
            // write the chunk to destination writer,
            // ensure it is consumed in whole
            let p = self.pos - self.chunk_len;
            let written = self.dst.write(&self.buf[p..self.pos])?;
            assert_eq!(written, self.chunk_len);

            // not enough space in buffer, copy remaining to
            // the head of buffer and reset buf position
            if self.pos + self.chunk_size >= WTR_BUF_LEN {
                let left_len = self.buf_clen - self.pos;
                unsafe {
                    ptr::copy::<u8>(
                        self.buf[self.pos..].as_ptr(),
                        self.buf.as_mut_ptr(),
                        left_len,
                    );
                }
                self.buf_clen = left_len;
                self.pos = 0;
            }

            // jump to next start sliding position
            self.pos += self.chunk_size;
            self.chunk_len = self.chunk_size;
        }

        Ok(in_len)
    }

    fn flush(&mut self) -> IoResult<()> {
        // flush remaining data to destination
        let p = self.pos - self.chunk_len;
        if p < self.buf_clen {
            self.chunk_len = self.buf_clen - p;
            let _ = self.dst.write(&self.buf[p..(p + self.chunk_len)])?;
        }

        // reset chunker
        self.pos = self.chunk_size;
        self.chunk_len = self.chunk_size;
        self.buf_clen = 0;

        self.dst.flush()
    }
}

impl<W: Write + Seek> Debug for FixedSizeChunker<W> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Chunker()")
    }
}

impl<W: Write + Seek> Seek for FixedSizeChunker<W> {
    fn seek(&mut self, pos: SeekFrom) -> IoResult<u64> {
        self.dst.seek(pos)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{copy, Cursor, Result as IoResult, Seek, SeekFrom, Write};
    use std::iter::FromIterator;
    use test::Bencher;

    use crate::base::crypto::{Crypto, Hash, RandomSeed, RANDOM_SEED_SIZE};
    use crate::base::init_env;
    use crate::content::chunk::Chunk;

    use super::*;

    #[derive(Debug)]
    struct Sinker {
        len: usize,
        chks: Vec<Chunk>,
    }

    impl Write for Sinker {
        fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
            self.chks.push(Chunk::new(self.len, buf.len()));
            self.len += buf.len();
            Ok(buf.len())
        }

        fn flush(&mut self) -> IoResult<()> {
            // verify
            let sum = self.chks.iter().fold(0, |sum, ref t| sum + t.len);
            assert_eq!(sum, self.len);
            for i in 0..(self.chks.len() - 2) {
                assert_eq!(
                    self.chks[i].pos + self.chks[i].len,
                    self.chks[i + 1].pos
                );
            }

            Ok(())
        }
    }

    impl Seek for Sinker {
        fn seek(&mut self, _: SeekFrom) -> IoResult<u64> {
            Ok(0)
        }
    }

    #[derive(Debug)]
    struct VoidSinker {}

    impl Write for VoidSinker {
        fn write(&mut self, buf: &[u8]) -> IoResult<usize> {
            Ok(buf.len())
        }

        fn flush(&mut self) -> IoResult<()> {
            Ok(())
        }
    }

    impl Seek for VoidSinker {
        fn seek(&mut self, _: SeekFrom) -> IoResult<u64> {
            Ok(0)
        }
    }

    #[test]
    fn chunker() {
        init_env();

        // perpare test data
        const DATA_LEN: usize = 765 * 1024;
        let params = ChunkerParams::new();
        let mut data = vec![0u8; DATA_LEN];
        Crypto::random_buf(&mut data);
        let mut cur = Cursor::new(data);
        let sinker = Sinker {
            len: 0,
            chks: Vec::new(),
        };

        // test chunker
        let mut ckr = Chunker::new(params, sinker);
        let result = copy(&mut cur, &mut ckr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), DATA_LEN as u64);
        ckr.flush().unwrap();
    }

    #[bench]
    fn fsc_chunker_perf(b: &mut Bencher) {
        let vec = std::fs::read(LNX).unwrap();

        b.iter(|| {
            init_env();

            let mut cur = Cursor::new(&vec);
            let sinker = VoidSinker {};

            let mut ckr = FixedSizeChunker::fine_tuned(sinker, KB_2);
            copy(&mut cur, &mut ckr).unwrap();
            ckr.flush().unwrap();
        });
    }

    #[bench]
    fn chunker_perf(b: &mut Bencher) {
        let vec = std::fs::read(LNX).unwrap();

        b.iter(|| {
            init_env();

            let mut cur = Cursor::new(&vec);
            let sinker = VoidSinker {};

            let mut ckr = Chunker::new(ChunkerParams::new(), sinker);
            copy(&mut cur, &mut ckr).unwrap();
            ckr.flush().unwrap();
        });
    }

    #[bench]
    fn fastcdc_chunker_perf(b: &mut Bencher) {
        let vec = std::fs::read(LNX).unwrap();

        b.iter(|| {
            init_env();

            let mut cur = Cursor::new(&vec);
            let sinker = VoidSinker {};

            let mut ckr = FastCdcChunker::fine_tuned(
                sinker,
                KB_2,
                KB_2,
                KB_4,
                Normalization::Level3,
            );
            copy(&mut cur, &mut ckr).unwrap();
            ckr.flush().unwrap();
        });
    }

    #[test]
    fn fsc_chunker_compression_all() {
        for chunk_size in KB_ALL {
            for ds in ALL_DS {
                fsc_chunker_compression(ds, chunk_size);
            }
        }
    }

    #[test]
    fn chunker_compression_all() {
        for ds in ALL_DS {
            chunker_compression(ds);
        }
    }

    #[test]
    fn fastcdc_chunker_compression_all() {
        for min_size in KB_SMALL {
            for avg_size in KB_SMALL.iter().filter(|kb| **kb >= min_size) {
                for max_size in KB_ALL.iter().filter(|kb| **kb > *avg_size) {
                    for ds in ALL_DS {
                        for lvl in LVLS {
                            fastcdc_chunker_compression(
                                ds, min_size, *avg_size, *max_size, lvl,
                            );
                        }
                    }
                }
            }
        }
    }

    const LNX: &str = "/home/graywolf/dev/learn/university/summer-school-2023/project/data-sets/linux-6.4.5-dups.tar";
    const UBN: &str = "/home/graywolf/dev/learn/university/summer-school-2023/project/data-sets/ubuntu-22.04.2-desktop-amd64.iso";

    const ALL_DS: [&str; 2] = [LNX, UBN];

    const KB_2: usize = 2 * 1024;
    const KB_4: usize = 4 * 1024;
    const KB_8: usize = 8 * 1024;
    const KB_16: usize = 16 * 1024;
    const KB_32: usize = 32 * 1024;
    const KB_64: usize = 64 * 1024;

    const KB_SMALL: [usize; 3] = [KB_2, KB_4, KB_8];
    const KB_ALL: [usize; 6] = [KB_2, KB_4, KB_8, KB_16, KB_32, KB_64];

    fn fsc_chunker_compression(path: &str, chunk_size: usize) {
        let vec = std::fs::read(path).unwrap();

        init_env();

        let mut sinker = Sinker {
            len: 0,
            chks: Vec::new(),
        };

        let mut cur = Cursor::new(vec.clone());
        let mut ckr = FixedSizeChunker::fine_tuned(&mut sinker, chunk_size);
        copy(&mut cur, &mut ckr).unwrap();
        ckr.flush().unwrap();

        let chks_cnt = sinker.chks.len();
        let chks_map: HashMap<Hash, usize> = HashMap::from_iter(
            sinker.chks.into_iter().map(|Chunk { pos, len, .. }| {
                (Crypto::hash(&vec[pos..(pos + len)]), len)
            }),
        );

        println!("FSC chunks: {} / {}", chks_map.len(), chks_cnt);
        println!(
            "FSC bytes: {} / {}",
            chks_map.iter().map(|(a, b)| b).sum::<usize>(),
            vec.len()
        );
    }

    fn chunker_compression(path: &str) {
        let vec = std::fs::read(path).unwrap();

        init_env();

        let params = ChunkerParams::new();

        let mut sinker = Sinker {
            len: 0,
            chks: Vec::new(),
        };

        let mut cur = Cursor::new(vec.clone());
        let mut ckr = Chunker::new(params, &mut sinker);
        copy(&mut cur, &mut ckr).unwrap();
        ckr.flush().unwrap();

        let chks_cnt = sinker.chks.len();
        let chks_map: HashMap<Hash, usize> = HashMap::from_iter(
            sinker.chks.into_iter().map(|Chunk { pos, len, .. }| {
                (Crypto::hash(&vec[pos..(pos + len)]), len)
            }),
        );

        println!("Rabin-based CDC chunks: {} / {}", chks_map.len(), chks_cnt);
        println!(
            "Rabin-based CDC bytes: {} / {}",
            chks_map.iter().map(|(a, b)| b).sum::<usize>(),
            vec.len()
        );
    }

    fn fastcdc_chunker_compression(
        path: &str,
        min_size: usize,
        avg_size: usize,
        max_size: usize,
        normalization_lvl: Normalization,
    ) {
        let vec = std::fs::read(path).unwrap();

        init_env();

        let mut sinker = Sinker {
            len: 0,
            chks: Vec::new(),
        };

        let mut cur = Cursor::new(vec.clone());
        let mut ckr = FastCdcChunker::fine_tuned(
            &mut sinker,
            min_size,
            avg_size,
            max_size,
            match normalization_lvl {
                Normalization::Level0 => Normalization::Level0,
                Normalization::Level1 => Normalization::Level1,
                Normalization::Level2 => Normalization::Level2,
                Normalization::Level3 => Normalization::Level3,
            },
        );
        copy(&mut cur, &mut ckr).unwrap();
        ckr.flush().unwrap();

        let chks_cnt = sinker.chks.len();
        let chks_map: HashMap<Hash, usize> = HashMap::from_iter(
            sinker.chks.into_iter().map(|Chunk { pos, len, .. }| {
                (Crypto::hash(&vec[pos..(pos + len)]), len)
            }),
        );

        println!(
            "FastCDC ({}|{}|{} - {}) chunks: {} / {}",
            min_size,
            avg_size,
            max_size,
            match normalization_lvl {
                Normalization::Level0 => 0,
                Normalization::Level1 => 1,
                Normalization::Level2 => 2,
                Normalization::Level3 => 3,
            },
            chks_map.len(),
            chks_cnt
        );
        println!(
            "FastCDC ({}|{}|{} - {}) bytes: {} / {}",
            min_size,
            avg_size,
            max_size,
            match normalization_lvl {
                Normalization::Level0 => 0,
                Normalization::Level1 => 1,
                Normalization::Level2 => 2,
                Normalization::Level3 => 3,
            },
            chks_map.iter().map(|(a, b)| b).sum::<usize>(),
            vec.len()
        );
    }
}
