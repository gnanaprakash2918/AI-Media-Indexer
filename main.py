"""Simple example script to demonstrate scanning and probing media files."""

from core.ingestion.scanner import LibraryScanner
from core.processing.prober import MediaProbeError, MediaProber


def main():
    """Main entry point for the script."""
    scanner = LibraryScanner()
    prober = MediaProber()

    target_folder = "D:\\Hibernate"  # noqa: E501

    print(f"Scanning: {target_folder}")

    for asset in scanner.scan(target_folder):
        print(f"Found {asset.media_type.value}: {asset.file_path.name}")

        if asset.media_type == "video":
            try:
                meta = prober.probe(asset.file_path)
                duration = meta.get("format", {}).get("duration", "N/A")
                print(f"   --> Duration: {duration}s")
            except MediaProbeError as e:
                print(f"   --> Error probing: {e}")


if __name__ == "__main__":
    main()
