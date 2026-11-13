"""Read selected members of a remote COCO ZIP using HTTP byte ranges."""
from http.client import IncompleteRead
import io
import re
import struct
import time
from urllib.request import Request, urlopen
import zipfile
import zlib


class RangeZip(io.RawIOBase):
    def __init__(self, url, retries):
        super().__init__()
        self.url = url
        self.retries = retries
        self.size = None
        self.position = 0
        self.read_range(0, 1)
        with zipfile.ZipFile(self) as archive:
            entries = sorted(archive.infolist(), key=lambda info: info.header_offset)
            offsets = [info.header_offset for info in entries] + [archive.start_dir]
        self.members = {info.filename: (info, end) for info, end in zip(entries, offsets[1:])}

    def read_range(self, start, length):
        end = start+length-1
        request = Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        for attempt in range(self.retries+1):
            try:
                with urlopen(request, timeout=60) as response:
                    if response.status == 200:
                        raise ValueError(f'Mirror ignored HTTP Range for {self.url}')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match:
                        raise OSError('Invalid HTTP Range response')
                    actual_start, actual_end, size = map(int, match.groups())
                    if (actual_start, actual_end) != (start, end) or end >= size or self.size not in (None, size):
                        raise OSError('Unexpected HTTP Content-Range')
                    data = response.read(length+1)
                    if len(data) != length:
                        raise OSError(f'Incomplete HTTP Range: expected {length} bytes, got {len(data)}')
                if self.size is None:
                    self.size = size
                return data
            except (OSError, IncompleteRead) as exc:
                if attempt == self.retries:
                    raise OSError(f'{self.url}, bytes {start}-{end}: {exc}') from exc
                time.sleep(min(2**attempt, 15))

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.size
        elif whence != io.SEEK_SET:
            raise ValueError(f'Invalid seek mode: {whence}')
        if offset < 0:
            raise OSError('Negative ZIP offset')
        self.position = offset
        return offset

    def read(self, size=-1):
        remaining = self.size-self.position
        length = remaining if size < 0 else min(size, remaining)
        if length <= 0:
            return b''
        data = self.read_range(self.position, length)
        self.position += length
        return data

    def image(self, split, filename):
        member = f'{split}2017/{filename}'
        if member not in self.members:
            raise ValueError(f'Image missing from mirror ZIP: {member}')
        info, end = self.members[member]
        payload = self.read_range(info.header_offset, end-info.header_offset)
        if len(payload) < 30:
            raise ValueError(f'Truncated local ZIP entry: {member}')
        signature, *_, name_length, extra_length = struct.unpack_from('<4s5H3I2H', payload)
        if signature != b'PK\x03\x04' or payload[30:30+name_length] != member.encode('ascii'):
            raise ValueError(f'Unexpected local ZIP entry: {member}')
        start = 30+name_length+extra_length
        compressed = payload[start:start+info.compress_size]
        if info.compress_type == zipfile.ZIP_STORED:
            data = compressed
        elif info.compress_type == zipfile.ZIP_DEFLATED:
            try:
                data = zlib.decompress(compressed, -zlib.MAX_WBITS)
            except zlib.error as exc:
                raise ValueError(f'Invalid ZIP member: {member}') from exc
        else:
            raise ValueError(f'Unsupported ZIP compression for {member}: {info.compress_type}')
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise ValueError(f'ZIP CRC/size mismatch: {member}')
        return data
