import { useState, memo } from 'react';
import { PlayArrow, AccessTime, Videocam, GraphicEq, TextFields, Face, Mic, LocalOffer, Room, Visibility } from '@mui/icons-material';
import { Card, CardMedia, CardContent, Typography, Box, Chip, IconButton, alpha, useTheme, Tooltip, Collapse } from '@mui/material';
import { VideoPlayer } from './VideoPlayer';

interface MediaResult {
    score: number;
    base_score?: number;
    keyword_boost?: number;
    video_path: string;
    start?: number;
    end?: number;
    timestamp?: number;
    text?: string;
    type?: string;
    action?: string;
    description?: string;
    thumbnail_url?: string;
    // HITL Identity Fields
    face_names?: string[];
    speaker_names?: string[];
    face_cluster_ids?: number[];
    // Structured Analysis Fields
    entities?: string[];
    visible_text?: string[];
    scene_location?: string;
    identity_text?: string;
}

export const MediaCard = memo(function MediaCard({ item }: { item: MediaResult }) {
    const theme = useTheme();
    const [playerOpen, setPlayerOpen] = useState(false);
    const [expanded, setExpanded] = useState(false);

    // Extract filename for thumbnail
    const filename = item.video_path.split(/[/\\]/).pop();
    const baseName = filename?.replace(/\\.[^/.]+$/, "") || "";

    // Use dynamic thumbnail if available, else fallback to static or placeholder
    const thumbnailUrl = item.thumbnail_url
        ? `http://localhost:8000${item.thumbnail_url}`
        : `http://localhost:8000/thumbnails/${filename}.jpg`;

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

    // Check if we have HITL data
    const hasFaces = item.face_names && item.face_names.length > 0;
    const hasSpeakers = item.speaker_names && item.speaker_names.length > 0;
    const hasEntities = item.entities && item.entities.length > 0;
    const hasVisibleText = item.visible_text && item.visible_text.length > 0;
    const hasScene = item.scene_location && item.scene_location.length > 0;
    const hasDetails = hasFaces || hasSpeakers || hasEntities || hasVisibleText || hasScene;

    // Use timestamp or start
    const timeValue = item.timestamp ?? item.start;

    return (
        <>
            <Card
                sx={{ height: '100%', display: 'flex', flexDirection: 'column', position: 'relative' }}
                onClick={() => hasDetails && setExpanded(!expanded)}
            >
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
                    {timeValue !== undefined && (
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
                            {formatTime(timeValue)}
                        </Box>
                    )}

                    {/* Face Names Badge (top-left) */}
                    {hasFaces && (
                        <Box
                            sx={{
                                position: 'absolute',
                                top: 8,
                                left: 8,
                                display: 'flex',
                                gap: 0.5,
                                flexWrap: 'wrap',
                            }}
                        >
                            {item.face_names!.slice(0, 2).map((name, i) => (
                                <Chip
                                    key={i}
                                    icon={<Face sx={{ fontSize: 14 }} />}
                                    label={name}
                                    size="small"
                                    sx={{
                                        bgcolor: alpha(theme.palette.info.main, 0.9),
                                        color: 'white',
                                        fontWeight: 600,
                                        height: 24,
                                    }}
                                />
                            ))}
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
                            onClick={(e) => { e.stopPropagation(); handlePlay(); }}
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
                    {/* Score and Type Row */}
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
                        <Tooltip title={
                            item.base_score !== undefined ?
                                `Base: ${(item.base_score * 100).toFixed(0)}% + Keyword: ${((item.keyword_boost || 0) * 100).toFixed(0)}%` :
                                "Confidence Score"
                        }>
                            <Typography variant="caption" sx={{ color: 'success.main', fontWeight: 700, fontFamily: 'monospace' }}>
                                {(item.score * 100).toFixed(0)}%
                            </Typography>
                        </Tooltip>
                    </Box>

                    {/* Speaker Names */}
                    {hasSpeakers && (
                        <Box sx={{ display: 'flex', gap: 0.5, mb: 1, flexWrap: 'wrap' }}>
                            {item.speaker_names!.map((name, i) => (
                                <Chip
                                    key={i}
                                    icon={<Mic sx={{ fontSize: 12 }} />}
                                    label={name}
                                    size="small"
                                    variant="outlined"
                                    sx={{ height: 20, fontSize: '0.7rem' }}
                                />
                            ))}
                        </Box>
                    )}

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
                            {item.action || item.description || "Visual match detected in scene."}
                        </Typography>
                    )}

                    {/* Expandable Details */}
                    <Collapse in={expanded}>
                        <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
                            {/* Entities */}
                            {hasEntities && (
                                <Box sx={{ mb: 1 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                                        <LocalOffer sx={{ fontSize: 12, color: 'text.secondary' }} />
                                        <Typography variant="caption" color="text.secondary">Entities</Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                                        {item.entities!.slice(0, 5).map((e, i) => (
                                            <Chip key={i} label={e} size="small" sx={{ height: 18, fontSize: '0.65rem' }} />
                                        ))}
                                    </Box>
                                </Box>
                            )}

                            {/* Scene Location */}
                            {hasScene && (
                                <Box sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <Room sx={{ fontSize: 12, color: 'text.secondary' }} />
                                    <Typography variant="caption">{item.scene_location}</Typography>
                                </Box>
                            )}

                            {/* Visible Text (OCR) */}
                            {hasVisibleText && (
                                <Box sx={{ mb: 1 }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
                                        <Visibility sx={{ fontSize: 12, color: 'text.secondary' }} />
                                        <Typography variant="caption" color="text.secondary">Visible Text</Typography>
                                    </Box>
                                    <Typography variant="caption" sx={{ fontFamily: 'monospace', bgcolor: 'action.hover', px: 0.5, borderRadius: 0.5 }}>
                                        {item.visible_text!.join(' | ')}
                                    </Typography>
                                </Box>
                            )}

                            {/* Score Breakdown */}
                            {item.base_score !== undefined && (
                                <Box sx={{ display: 'flex', gap: 1, fontSize: '0.65rem', color: 'text.secondary' }}>
                                    <span>Base: {(item.base_score * 100).toFixed(0)}%</span>
                                    <span>Boost: +{((item.keyword_boost || 0) * 100).toFixed(0)}%</span>
                                </Box>
                            )}
                        </Box>
                    </Collapse>

                    <Typography variant="caption" color="text.disabled" sx={{ mt: 1, display: 'block', fontFamily: 'monospace' }} noWrap>
                        {baseName}
                    </Typography>

                    {/* Expand indicator */}
                    {hasDetails && (
                        <Typography variant="caption" color="primary" sx={{ cursor: 'pointer', display: 'block', textAlign: 'center', mt: 0.5 }}>
                            {expanded ? '▲ Less' : '▼ More details'}
                        </Typography>
                    )}
                </CardContent>
            </Card>
            <VideoPlayer
                videoPath={item.video_path}
                startTime={item.start ?? item.timestamp}
                endTime={item.end}
                open={playerOpen}
                onClose={() => setPlayerOpen(false)}
            />
        </>
    );
});

