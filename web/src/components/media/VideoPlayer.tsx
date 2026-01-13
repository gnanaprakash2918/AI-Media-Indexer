import { useState, useRef, useCallback, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  IconButton,
  Box,
  Typography,
  Button,
  Stack,
  LinearProgress,
} from '@mui/material';
import { Close, Loop, AllInclusive } from '@mui/icons-material';

interface VideoPlayerProps {
  videoPath: string;
  startTime?: number;
  endTime?: number;
  open: boolean;
  onClose: () => void;
}

function InnerVideoPlayer({
  videoPath,
  startTime,
  endTime,
}: {
  videoPath: string;
  startTime?: number;
  endTime?: number;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [error, setError] = useState(false);
  const [isFullVideo, setIsFullVideo] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  // Calculate segment bounds
  const segmentStart = startTime ?? 0;
  const segmentEnd = endTime ?? (startTime ? startTime + 10 : 0);
  const segmentDuration = segmentEnd - segmentStart;
  const hasSegment = startTime !== undefined && startTime > 0;

  // Video URL - always use the direct media endpoint (no encoding)
  const videoUrl = useMemo(() => {
    return `http://localhost:8000/stream?path=${encodeURIComponent(videoPath)}`;
  }, [videoPath]);

  // Handle video loaded - seek to segment start
  const handleLoadedMetadata = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    setIsLoading(false);

    // Seek to segment start (or 0 for full video)
    if (hasSegment && !isFullVideo) {
      video.currentTime = segmentStart;
    }
  }, [hasSegment, isFullVideo, segmentStart]);

  // Restrict seeking and loop within segment
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || isFullVideo || !hasSegment) return;

    const currentTime = video.currentTime;

    // Update progress bar
    const segmentProgress =
      ((currentTime - segmentStart) / segmentDuration) * 100;
    setProgress(Math.max(0, Math.min(100, segmentProgress)));

    // Restrict to segment bounds
    if (currentTime < segmentStart) {
      video.currentTime = segmentStart;
    } else if (currentTime >= segmentEnd) {
      // Loop back to start of segment
      video.currentTime = segmentStart;
      video.play();
    }
  }, [isFullVideo, hasSegment, segmentStart, segmentEnd, segmentDuration]);

  // Prevent seeking outside segment
  const handleSeeking = useCallback(() => {
    const video = videoRef.current;
    if (!video || isFullVideo || !hasSegment) return;

    const currentTime = video.currentTime;

    // Clamp to segment bounds
    if (currentTime < segmentStart) {
      video.currentTime = segmentStart;
    } else if (currentTime > segmentEnd) {
      video.currentTime = segmentEnd - 0.1;
    }
  }, [isFullVideo, hasSegment, segmentStart, segmentEnd]);

  // Handle mode switch (context seek)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (isFullVideo) {
      // Full video mode - seek to start of segment for context
      if (hasSegment) {
        video.currentTime = segmentStart;
      }
    } else if (hasSegment) {
      // Segment mode - ensure we're in bounds
      video.currentTime = segmentStart;
    }
  }, [isFullVideo, hasSegment, segmentStart]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <>
      <DialogContent sx={{ p: 0, position: 'relative' }}>
        {error ? (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography color="error">
              Unable to play video. File may not be accessible.
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 1, display: 'block' }}
            >
              Path: {videoPath}
            </Typography>
          </Box>
        ) : (
          <>
            {isLoading && (
              <Box
                sx={{
                  position: 'absolute',
                  top: '50%',
                  left: '50%',
                  transform: 'translate(-50%, -50%)',
                  textAlign: 'center',
                  zIndex: 1,
                }}
              >
                <Typography color="text.secondary" sx={{ mb: 1 }}>
                  Loading...
                </Typography>
              </Box>
            )}
            <video
              ref={videoRef}
              controls // Always show controls - seeking is bounded by JS
              autoPlay
              playsInline
              style={{
                width: '100%',
                maxHeight: '70vh',
                display: 'block',
                backgroundColor: '#000',
              }}
              onError={() => setError(true)}
              onLoadedMetadata={handleLoadedMetadata}
              onTimeUpdate={handleTimeUpdate}
              onSeeking={handleSeeking}
            >
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support video playback.
            </video>
          </>
        )}
      </DialogContent>

      {/* Segment controls and progress */}
      {hasSegment && (
        <Box
          sx={{
            p: 1.5,
            borderTop: 1,
            borderColor: 'divider',
          }}
        >
          {/* Custom progress bar for segment mode */}
          {!isFullVideo && (
            <Box sx={{ mb: 1.5 }}>
              <LinearProgress
                variant="determinate"
                value={progress}
                sx={{
                  height: 6,
                  borderRadius: 3,
                  '& .MuiLinearProgress-bar': {
                    borderRadius: 3,
                  },
                }}
              />
            </Box>
          )}

          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <Typography variant="body2" color="text.secondary">
              {isFullVideo
                ? 'Full video mode'
                : `Segment: ${formatTime(segmentStart)} - ${formatTime(segmentEnd)} (${segmentDuration.toFixed(1)}s)`}
            </Typography>
            <Stack direction="row" spacing={1}>
              <Button
                size="small"
                variant={isFullVideo ? 'outlined' : 'contained'}
                startIcon={<Loop />}
                onClick={() => setIsFullVideo(false)}
              >
                Segment Only
              </Button>
              <Button
                size="small"
                variant={isFullVideo ? 'contained' : 'outlined'}
                startIcon={<AllInclusive />}
                onClick={() => setIsFullVideo(true)}
              >
                Full Video
              </Button>
            </Stack>
          </Box>
        </Box>
      )}
    </>
  );
}

export function VideoPlayer({
  videoPath,
  startTime,
  endTime,
  open,
  onClose,
}: VideoPlayerProps) {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { borderRadius: 2, overflow: 'hidden' },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          p: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Typography
          variant="subtitle2"
          noWrap
          sx={{ maxWidth: '70%', fontWeight: 600 }}
        >
          {videoPath.split(/[/\\]/).pop()}
        </Typography>
        <IconButton onClick={onClose} size="small">
          <Close />
        </IconButton>
      </Box>

      <InnerVideoPlayer
        key={`${videoPath}-${open}`} // Force reset when path or open state changes
        videoPath={videoPath}
        startTime={startTime}
        endTime={endTime}
      />
    </Dialog>
  );
}
