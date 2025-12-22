import { useState } from 'react';
import { PlayArrow, AccessTime, Videocam, GraphicEq, TextFields } from '@mui/icons-material';
import { Card, CardMedia, CardContent, Typography, Box, Chip, IconButton, alpha, useTheme } from '@mui/material';
import { VideoPlayer } from './VideoPlayer';

interface MediaResult {
    score: number;
    video_path: string;
    start?: number;
    end?: number;
    text?: string;
    type?: string;
    action?: string;
}

export function MediaCard({ item }: { item: MediaResult }) {
    const theme = useTheme();
    const [playerOpen, setPlayerOpen] = useState(false);

    // Extract filename for thumbnail
    const filename = item.video_path.split(/[/\\]/).pop();
    const baseName = filename?.replace(/\\.[^/.]+$/, "") || "";
    const thumbnailUrl = `http://localhost:8000/thumbnails/${filename}.jpg`;

    // Format timestamp
    const formatTime = (s?: number) => {
        if (s === undefined) return "";
        const mins = Math.floor(s / 60);
        const secs = Math.floor(s % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const getIcon = () => {
        if (item.type === 'dialogue') return <GraphicEq fontSize="small" />;
        if (item.type === 'visual') return <Videocam fontSize="small" />;
        return <TextFields fontSize="small" />;
    };

    const handlePlay = () => setPlayerOpen(true);

    return (
        <>
            <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}>
                <Box sx={{ position: 'relative', paddingTop: '56.25%' /* 16:9 Aspect Ratio */ }}>
                    <CardMedia
                        component="img"
                        image={thumbnailUrl}
                        alt={filename}
                        sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            objectFit: 'cover',
                        }}
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://placehold.co/600x400/18181b/52525b?text=No+Preview';
                        }}
                    />

                    {/* Overlay Gradient */}
                    <Box
                        sx={{
                            position: 'absolute',
                            bottom: 0,
                            left: 0,
                            width: '100%',
                            height: '50%',
                            background: 'linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)',
                        }}
                    />

                    {/* Timestamp Badge */}
                    {item.start !== undefined && (
                        <Box
                            sx={{
                                position: 'absolute',
                                bottom: 8,
                                right: 8,
                                bgcolor: 'rgba(0, 0, 0, 0.75)',
                                color: 'white',
                                px: 1,
                                py: 0.5,
                                borderRadius: 1,
                                display: 'flex',
                                alignItems: 'center',
                                gap: 0.5,
                                fontSize: '0.75rem',
                                fontWeight: 600,
                                backdropFilter: 'blur(4px)',
                            }}
                        >
                            <AccessTime sx={{ fontSize: 14 }} />
                            {formatTime(item.start)}
                        </Box>
                    )}

                    {/* Play Button Overlay (Hover) */}
                    <Box
                        className="play-overlay"
                        sx={{
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            opacity: 0,
                            transition: 'opacity 0.2s',
                            bgcolor: 'rgba(0,0,0,0.3)',
                            '&:hover': { opacity: 1 },
                        }}
                    >
                        <IconButton
                            onClick={handlePlay}
                            sx={{
                                bgcolor: 'rgba(255,255,255,0.2)',
                                backdropFilter: 'blur(8px)',
                                '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' },
                                width: 56,
                                height: 56,
                            }}
                        >
                            <PlayArrow sx={{ color: 'white', fontSize: 32 }} />
                        </IconButton>
                    </Box>
                </Box>

                <CardContent sx={{ flexGrow: 1, p: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Chip
                            icon={getIcon()}
                            label={item.type || "Match"}
                            size="small"
                            sx={{
                                height: 24,
                                fontSize: '0.75rem',
                                fontWeight: 600,
                                bgcolor: alpha(theme.palette.primary.main, 0.1),
                                color: theme.palette.primary.main
                            }}
                        />
                        <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 700, fontFamily: 'monospace' }}>
                            {(item.score * 100).toFixed(0)}%
                        </Typography>
                    </Box>

                    {/* Subtitle with time range */}
                    {item.text && (
                        <Box sx={{ bgcolor: 'action.hover', p: 1, borderRadius: 1, mb: 1 }}>
                            <Typography variant="body2" sx={{ fontStyle: 'italic', fontWeight: 500 }}>
                                "{item.text}"
                            </Typography>
                            {item.start !== undefined && item.end !== undefined && (
                                <Typography variant="caption" color="text.secondary">
                                    {formatTime(item.start)} - {formatTime(item.end)}
                                </Typography>
                            )}
                        </Box>
                    )}

                    {!item.text && (
                        <Typography variant="body2" color="text.secondary" sx={{
                            display: '-webkit-box',
                            WebkitLineClamp: 2,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                            lineHeight: 1.6,
                            fontWeight: 500
                        }}>
                            {item.action || "Visual match detected in scene."}
                        </Typography>
                    )}

                    <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block', fontFamily: 'monospace' }} noWrap>
                        {baseName}
                    </Typography>
                </CardContent>
            </Card>
            <VideoPlayer
                videoPath={item.video_path}
                startTime={item.start}
                open={playerOpen}
                onClose={() => setPlayerOpen(false)}
            />
        </>
    );
}
