"""Lightweight file dialog helper using multiprocessing.

Tkinter requires the main thread. Since FastAPI/uvicorn owns the main
thread, we spawn a separate process where tkinter gets its own main thread.

This module is intentionally kept free of heavy imports so that
multiprocessing's 'spawn' start method (Windows) can import it quickly.
"""

import multiprocessing


def _open_dialog(queue: multiprocessing.Queue, directory: bool) -> None:
    """Target for the child process — runs on the child's main thread."""
    import tkinter as tk
    from tkinter import filedialog

    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if directory:
            path = filedialog.askdirectory(title="Select Folder")
        else:
            path = filedialog.askopenfilename(
                title="Select Media File",
                filetypes=[
                    ("All files", "*.*"),
                    (
                        "Video files",
                        "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.ts *.wmv *.flv",
                    ),
                    (
                        "Audio files",
                        "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.wma",
                    ),
                ],
            )

        root.destroy()
        queue.put(path if path else None)
    except Exception:
        queue.put(None)


async def open_file_dialog(directory: bool = False) -> str | None:
    """Open a native file/folder dialog in a separate process.

    Spawns a child process so tkinter runs on that process's main thread,
    avoiding the 'main thread is not in main loop' error with FastAPI.
    """
    import asyncio

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_open_dialog, args=(queue, directory))
    proc.start()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, proc.join, 60)

    # Clean up if the dialog timed out
    if proc.is_alive():
        proc.terminate()
        return None

    if not queue.empty():
        return queue.get_nowait()
    return None
