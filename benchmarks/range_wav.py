"""Extract an exact PCM time window from a remote ZIP using bounded reads."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
import wave
from zipfile import ZipFile


class RangeFile(io.RawIOBase):
    def __init__(self, url, max_bytes):
        self.url = url
        self.max_bytes = max_bytes
        self.fetched = 0
        self.position = 0
        self.buffer = b""
        self.buffer_start = 0
        self.size = None
        self.block(0, 1)

    def block(self, start, stop):
        length = stop - start
        if self.fetched + length > self.max_bytes:
            raise ValueError(f"Remote read would exceed the {self.max_bytes:,}-byte cap")
        request = Request(self.url, headers={
            "Range": f"bytes={start}-{stop - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "Aircraft-Doppler-Benchmark/1.0",
        })
        with urlopen(request, timeout=30) as response:
            interval = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                    response.headers.get("Content-Range", ""))
            if response.status != 206 or not interval:
                raise ValueError("Archive server did not honor the bounded byte range")
            first, last, size = map(int, interval.groups())
            if first != start or last != stop - 1 or self.size not in (None, size):
                raise ValueError("Archive server returned a different range or file size")
            data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("Archive byte range was truncated")
        self.size = size
        self.fetched += length
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}
        position = origin[whence] + offset
        if position < 0:
            raise OSError("Cannot seek before the archive")
        self.position = position
        return self.position

    def read(self, size=-1):
        stop = self.size if size < 0 else min(self.size, self.position + size)
        if self.position >= stop:
            return b""
        parts = []
        while self.position < stop:
            buffer_stop = self.buffer_start + len(self.buffer)
            if not self.buffer_start <= self.position < buffer_stop:
                self.buffer_start = self.position
                buffer_stop = min(self.size, max(stop, self.position + 4 * 2**20))
                self.buffer = self.block(self.position, buffer_stop)
            end = min(stop, buffer_stop)
            parts.append(self.buffer[self.position - self.buffer_start:end - self.buffer_start])
            self.position = end
        return b"".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("member")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--max-mib", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.offset < 0 or args.duration < 1 or args.max_mib < 1:
        parser.error("Use nonnegative offsets and positive duration and transfer cap")
    remote = RangeFile(args.url, args.max_mib * 2**20)
    with ZipFile(remote) as archive:
        info = archive.getinfo(args.member)
        with archive.open(info) as member, wave.open(member, "rb") as source:
            rate = source.getframerate()
            metadata = {
                "url": args.url, "member": info.filename, "member_crc32": f"{info.CRC:08x}",
                "member_compressed_bytes": info.compress_size, "member_bytes": info.file_size,
                "compression_method": info.compress_type, "sample_rate_hz": rate,
                "member_duration_s": source.getnframes() / rate,
                "offset_s": args.offset, "duration_s": args.duration,
            }
            if args.output is not None:
                count = args.duration * rate
                if (args.offset + args.duration) * rate > source.getnframes():
                    raise ValueError("Requested window extends beyond the WAV")
                source.setpos(args.offset * rate)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(".partial")
                try:
                    with wave.open(str(temporary), "wb") as out:
                        out.setparams(source.getparams())
                        out.setnframes(count)
                        for start in range(0, count, rate):
                            samples = source.readframes(min(rate, count - start))
                            if len(samples) != min(rate, count - start) * source.getsampwidth() * source.getnchannels():
                                raise ValueError("Source WAV ended before the fixed window")
                            out.writeframesraw(samples)
                    with temporary.open("rb") as saved:
                        metadata["excerpt_sha256"] = hashlib.file_digest(saved, "sha256").hexdigest()
                    temporary.replace(args.output)
                finally:
                    temporary.unlink(missing_ok=True)
    metadata["archive_bytes_fetched"] = remote.fetched
    if args.output is not None:
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
