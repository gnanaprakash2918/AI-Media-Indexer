import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import { Dialog, DialogContent, IconButton, Box, Typography, Button, Stack } from '@mui/material';
import { Close, Loop, AllInclusive } from '@mui/icons-material';

interface VideoPlayerProps {
    videoPath: string;
    startTime?: number;
    endTime?: number;
    open: boolean;
    onClose: () => void;
}

export function VideoPlayer({ videoPath, startTime, endTime, open, onClose }: VideoPlayerProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const [error, setError] = useState(false);
    const [isFullVideo, setIsFullVideo] = useState(false);

    // Calculate segment bounds
    const segmentStart = startTime ?? 0;
    const segmentEnd = endTime ?? (startTime ? startTime + 10 : 0);
    const hasSegment = startTime !== undefined && startTime > 0;

    // Detect video format for correct MIME type
    const videoFormat = useMemo(() => {
        const ext = videoPath.split('.').pop()?.toLowerCase() || 'mp4';
        const formatMap: Record<string, string> = {
            'webm': 'video/webm',
            'mp4': 'video/mp4',
            'mkv': 'video/x-matroska',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime',
            'm4v': 'video/mp4',
        };
        return formatMap[ext] || 'video/mp4';
    }, [videoPath]);

    // Use segment endpoint for instant playback, or full media endpoint for full video
    const videoUrl = useMemo(() => {
        if (hasSegment && !isFullVideo) {
            // Use FFmpeg-based segment endpoint for INSTANT playback
            return `http://localhost:8000/media/segment?path=${encodeURIComponent(videoPath)}&start=${segmentStart}&end=${segmentEnd}`;
        }
        // Full video - use regular streaming endpoint
        return `http://localhost:8000/media?path=${encodeURIComponent(videoPath)}`;
    }, [videoPath, hasSegment, isFullVideo, segmentStart, segmentEnd]);

    // Reset state when dialog opens/closes
    useEffect(() => {
        if (open) {
            setError(false);
        }
    }, [open]);

    // Handle video loaded - for full video mode, seek to start position
    const handleLoadedMetadata = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;

        // Only seek if in full video mode and we have a segment start
        if (isFullVideo && hasSegment && segmentStart > 0) {
            video.currentTime = segmentStart;
        }
    }, [isFullVideo, hasSegment, segmentStart]);

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="lg"
            fullWidth
            PaperProps={{
                sx: { borderRadius: 2, overflow: 'hidden' }
            }}
        >
            {/* Header */}
            <Box sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                p: 1.5,
                borderBottom: 1,
                borderColor: 'divider',
            }}>
                <Typography variant="subtitle2" noWrap sx={{ maxWidth: '70%', fontWeight: 600 }}>
                    {videoPath.split(/[/\\]/).pop()}
                </Typography>
                <IconButton onClick={onClose} size="small">
                    <Close />
                </IconButton>
            </Box>

            <DialogContent sx={{ p: 0 }}>
                {error ? (
                    <Box sx={{ p: 4, textAlign: 'center' }}>
                        <Typography color="error">
                            Unable to play video. File may not be accessible.
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                            Path: {videoPath}
                        </Typography>
                    </Box>
                ) : (
                    <video
                        ref={videoRef}
                        key={videoUrl}  // Force reload when URL changes
                        controls
                        autoPlay
                        playsInline
                        preload="auto"
                        style={{
                            width: '100%',
                            maxHeight: '70vh',
                            display: 'block',
                            backgroundColor: '#000',
                        }}
                        onError={() => setError(true)}
                        onLoadedMetadata={handleLoadedMetadata}
                    >
                        <source src={videoUrl} type={videoFormat} />
                        Your browser does not support video playback.
                    </video>
                )}
            </DialogContent>

            {/* Segment controls */}
            {hasSegment && (
                <Box sx={{
                    p: 1.5,
                    borderTop: 1,
                    borderColor: 'divider',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                }}>
                    <Typography variant="body2" color="text.secondary">
                        {isFullVideo ? 'Full video' : `Segment: ${formatTime(segmentStart)} - ${formatTime(segmentEnd)}`}
                    </Typography>
                    <Stack direction="row" spacing={1}>
                        <Button
                            size="small"
                            variant={isFullVideo ? "outlined" : "contained"}
                            startIcon={<Loop />}
                            onClick={() => setIsFullVideo(false)}
                        >
                            Play Segment
                        </Button>
                        <Button
                            size="small"
                            variant={isFullVideo ? "contained" : "outlined"}
                            startIcon={<AllInclusive />}
                            onClick={() => setIsFullVideo(true)}
                        >
                            Full Video
                        </Button>
                    </Stack>
                </Box>
            )}
        </Dialog>
    );
}
