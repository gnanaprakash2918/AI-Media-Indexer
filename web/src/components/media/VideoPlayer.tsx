import { useState } from 'react';
import { Dialog, DialogContent, IconButton, Box, Typography } from '@mui/material';
import { Close } from '@mui/icons-material';

interface VideoPlayerProps {
    videoPath: string;
    startTime?: number;
    endTime?: number;
    open: boolean;
    onClose: () => void;
}

export function VideoPlayer({ videoPath, startTime, endTime, open, onClose }: VideoPlayerProps) {
    const [error, setError] = useState(false);
    const [isFullVideo, setIsFullVideo] = useState(false);

    return (
        <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1 }}>
                <Typography variant="subtitle2" noWrap sx={{ maxWidth: '80%' }}>
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
                        controls
                        autoPlay
                        muted={false}
                        style={{ width: '100%', maxHeight: '70vh' }}
                        onError={() => setError(true)}
                        onLoadedMetadata={(e) => {
                            const video = e.target as HTMLVideoElement;
                            if (startTime) video.currentTime = startTime;
                        }}
                        onTimeUpdate={(e) => {
                            const video = e.target as HTMLVideoElement;
                            // Auto-loop segment if in segment mode
                            if (startTime && !isFullVideo) {
                                // Default segment is 5s if no end provided
                                const end = endTime || (startTime + 5);
                                if (video.currentTime >= end) {
                                    video.pause();
                                    video.currentTime = startTime;
                                    video.play();
                                }
                            }
                        }}
                    >
                        <source src={`http://localhost:8000/media?path=${encodeURIComponent(videoPath)}`} type="video/mp4" />
                        Your browser does not support video playback.
                    </video>
                )}
            </DialogContent>
            {startTime && (
                <Box sx={{ p: 2, bgcolor: 'background.paper', borderTop: 1, borderColor: 'divider', display: 'flex', gap: 2, alignItems: 'center' }}>
                    <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold' }}>
                        {isFullVideo ? "Viewing Full Video" : "Looping Search Result Segment"}
                    </Typography>
                    <IconButton
                        size="small"
                        onClick={() => setIsFullVideo(!isFullVideo)}
                        sx={{ border: 1, borderColor: 'divider', borderRadius: 1 }}
                    >
                        {isFullVideo ? "Limit to Segment" : "Watch Full Video"}
                    </IconButton>
                </Box>
            )}
        </Dialog>
    );
}
