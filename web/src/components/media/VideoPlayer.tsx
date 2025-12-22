import { useState } from 'react';
import { Dialog, DialogContent, IconButton, Box, Typography } from '@mui/material';
import { Close } from '@mui/icons-material';

interface VideoPlayerProps {
    videoPath: string;
    startTime?: number;
    open: boolean;
    onClose: () => void;
}

export function VideoPlayer({ videoPath, startTime, open, onClose }: VideoPlayerProps) {
    const [error, setError] = useState(false);

    // For local files, we need a file:// URL or API endpoint
    // Backend should serve video files via /media endpoint
    const videoUrl = `http://localhost:8000/media?path=${encodeURIComponent(videoPath)}${startTime ? `#t=${startTime}` : ''}`;

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
                        style={{ width: '100%', maxHeight: '70vh' }}
                        onError={() => setError(true)}
                    >
                        <source src={videoUrl} />
                        Your browser does not support video playback.
                    </video>
                )}
            </DialogContent>
        </Dialog>
    );
}
