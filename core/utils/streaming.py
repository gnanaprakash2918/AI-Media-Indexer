"""Utilities for streaming file content with Range support."""

from typing import BinaryIO, Generator


def range_generator(
    file_obj: BinaryIO, start: int, end: int, chunk_size: int = 64 * 1024
) -> Generator[bytes, None, None]:
    """Yield file chunks for range request.

    Args:
        file_obj: Opened file object (binary mode).
        start: Start byte position.
        end: End byte position (inclusive).
        chunk_size: Chunk size in bytes.

    Yields:
        Bytes chunks.
    """
    file_obj.seek(start)
    bytes_to_read = end - start + 1

    while bytes_to_read > 0:
        read_size = min(chunk_size, bytes_to_read)
        data = file_obj.read(read_size)

        if not data:
            break

        yield data
        bytes_to_read -= len(data)
