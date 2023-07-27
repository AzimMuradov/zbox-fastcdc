use fastcdc::v2020::{FastCDC, Normalization};
use std::cmp::min;
use std::fmt::{self, Debug};
use std::io::{Result as IoResult, Seek, SeekFrom, Write};
use std::ptr;

const MIN_SIZE: usize = 2 * 1024; // minimal chunk size, 2k
const AVG_SIZE: usize = 2 * 1024; // average chunk size, 2k
const MAX_SIZE: usize = 32 * 1024; // maximum chunk size, 32k

const NORMALIZATION_LEVEL: Normalization = Normalization::Level2;

// writer buffer length
const WTR_BUF_LEN: usize = 8 * MAX_SIZE;

/// Chunker
pub struct Chunker<W: Write + Seek> {
    dst: W, // destination writer
    pos: usize,
    chunk_len: usize,
    buf_clen: usize, // current length
    roll_hash: u64,
    buf: Vec<u8>, // chunker buffer, fixed size: WTR_BUF_LEN
}

impl<W: Write + Seek> Chunker<W> {
    pub fn new(dst: W) -> Self {
        let mut buf = vec![0u8; WTR_BUF_LEN];
        buf.shrink_to_fit();

        Chunker {
            dst,
            pos: MIN_SIZE,
            chunk_len: MIN_SIZE,
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
            self.pos -= MIN_SIZE;

            let (hash, cut_point) = FastCDC::with_level(
                &*self.buf,
                MIN_SIZE as u32,
                AVG_SIZE as u32,
                MAX_SIZE as u32,
                NORMALIZATION_LEVEL,
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
            self.pos += MIN_SIZE;
            self.chunk_len = MIN_SIZE;
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
        self.pos = MIN_SIZE;
        self.chunk_len = MIN_SIZE;
        self.buf_clen = 0;
        self.roll_hash = 0;

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

#[cfg(test)]
mod tests {
    use std::io::{copy, Cursor, Result as IoResult, Seek, SeekFrom, Write};
    use std::time::Instant;

    use super::*;
    use crate::base::crypto::{Crypto, RandomSeed, RANDOM_SEED_SIZE};
    use crate::base::init_env;
    use crate::base::utils::speed_str;
    use crate::content::chunk::Chunk;

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
        let mut data = vec![0u8; DATA_LEN];
        Crypto::random_buf(&mut data);
        let mut cur = Cursor::new(data);
        let sinker = Sinker {
            len: 0,
            chks: Vec::new(),
        };

        // test chunker
        let mut ckr = Chunker::new(sinker);
        let result = copy(&mut cur, &mut ckr);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), DATA_LEN as u64);
        ckr.flush().unwrap();
    }

    #[test]
    fn chunker_perf() {
        init_env();

        // perpare test data
        const DATA_LEN: usize = 10 * 1024 * 1024;
        let mut data = vec![0u8; DATA_LEN];
        let seed = RandomSeed::from(&[0u8; RANDOM_SEED_SIZE]);
        Crypto::random_buf_deterministic(&mut data, &seed);
        let mut cur = Cursor::new(data);
        let sinker = VoidSinker {};

        // test chunker performance
        let mut ckr = Chunker::new(sinker);
        let now = Instant::now();
        copy(&mut cur, &mut ckr).unwrap();
        ckr.flush().unwrap();
        let time = now.elapsed();

        println!("Chunker perf: {}", speed_str(&time, DATA_LEN));
    }
}
