import { useState, useRef, useEffect } from 'react';
import {
    Box,
    Paper,
    Typography,
    Button,
    Slider,
    Stack,
    IconButton,
    Tooltip,
    CircularProgress,
    Alert,
    Snackbar,
    Card,
    CardContent,
    Divider,
    TextField,
} from '@mui/material';
import {
    PlayArrow,
    Pause,
    AutoFixHigh,
    BlurOn,
    ContentCut,
    Undo,
    Redo,
    Save,
    VideoFile,
    ZoomIn,
    ZoomOut,
    RestartAlt,
} from '@mui/icons-material';
import { useMutation, useQuery } from '@tanstack/react-query';

interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

interface Job {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    result_path?: string;
    error?: string;
}

import {
    triggerInpaint,
    triggerRedact,
    getManipulationJob as getJobStatus,
} from '../api/client';

export default function VideoEditor() {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [videoPath, setVideoPath] = useState('');
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [startTime, setStartTime] = useState(0);
    const [endTime, setEndTime] = useState(10);

    // Bounding box state
    const [bbox, setBbox] = useState<BoundingBox | null>(null);
    const [isDrawing, setIsDrawing] = useState(false);
    const [drawStart, setDrawStart] = useState({ x: 0, y: 0 });

    // Job state
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const [snackMessage, setSnackMessage] = useState<string | null>(null);

    // Video dimensions
    const [videoWidth, setVideoWidth] = useState(640);
    const [videoHeight, setVideoHeight] = useState(360);

    // Poll for job status
    const { data: jobData } = useQuery({
        queryKey: ['job', activeJobId],
        queryFn: () => getJobStatus(activeJobId!),
        enabled: !!activeJobId,
        refetchInterval: activeJobId ? 1000 : false,
    });

    useEffect(() => {
        if (jobData?.status === 'completed') {
            setSnackMessage(`✅ Job completed! Output: ${jobData.result_path}`);
            setActiveJobId(null);
        } else if (jobData?.status === 'failed') {
            setSnackMessage(`❌ Job failed: ${jobData.error}`);
            setActiveJobId(null);
        }
    }, [jobData]);

    // Inpaint mutation
    const inpaintMutation = useMutation({
        mutationFn: triggerInpaint,
        onSuccess: (data) => {
            setActiveJobId(data.job_id);
            setSnackMessage('🎬 Inpainting job started...');
        },
        onError: (err: Error) => setSnackMessage(`Error: ${err.message}`),
    });

    // Redact mutation
    const redactMutation = useMutation({
        mutationFn: triggerRedact,
        onSuccess: (data) => {
            setActiveJobId(data.job_id);
            setSnackMessage('🔒 Redaction job started...');
        },
        onError: (err: Error) => setSnackMessage(`Error: ${err.message}`),
    });

    const handleLoadVideo = () => {
        if (videoRef.current && videoPath) {
            videoRef.current.src = `/api/media/${encodeURIComponent(videoPath)}`;
        }
    };

    const handleVideoLoaded = () => {
        if (videoRef.current) {
            setDuration(videoRef.current.duration);
            setVideoWidth(videoRef.current.videoWidth);
            setVideoHeight(videoRef.current.videoHeight);
            setEndTime(Math.min(10, videoRef.current.duration));
        }
    };

    const handleTimeUpdate = () => {
        if (videoRef.current) {
            setCurrentTime(videoRef.current.currentTime);
        }
    };

    const togglePlay = () => {
        if (videoRef.current) {
            if (isPlaying) {
                videoRef.current.pause();
            } else {
                videoRef.current.play();
            }
            setIsPlaying(!isPlaying);
        }
    };

    const handleSeek = (_: Event, value: number | number[]) => {
        const time = value as number;
        if (videoRef.current) {
            videoRef.current.currentTime = time;
            setCurrentTime(time);
        }
    };

    // Drawing handlers
    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        setIsDrawing(true);
        setDrawStart({ x, y });
        setBbox(null);
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDrawing || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear and redraw
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw rectangle
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(
            drawStart.x,
            drawStart.y,
            x - drawStart.x,
            y - drawStart.y
        );
    };

    const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
        if (!isDrawing || !canvasRef.current) return;

        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        setIsDrawing(false);

        // Calculate bbox in video coordinates
        const scaleX = videoWidth / canvas.width;
        const scaleY = videoHeight / canvas.height;

        const newBbox: BoundingBox = {
            x: Math.round(Math.min(drawStart.x, x) * scaleX),
            y: Math.round(Math.min(drawStart.y, y) * scaleY),
            width: Math.round(Math.abs(x - drawStart.x) * scaleX),
            height: Math.round(Math.abs(y - drawStart.y) * scaleY),
        };

        setBbox(newBbox);
    };

    const handleInpaint = () => {
        if (!bbox || !videoPath) return;

        inpaintMutation.mutate({
            video_path: videoPath,
            start_time: startTime,
            end_time: endTime,
            bbox: [bbox.x, bbox.y, bbox.width, bbox.height],
        });
    };

    const handleRedact = () => {
        if (!bbox || !videoPath) return;

        redactMutation.mutate({
            video_path: videoPath,
            start_time: startTime,
            end_time: endTime,
            bbox: [bbox.x, bbox.y, bbox.width, bbox.height],
        });
    };

    const clearSelection = () => {
        setBbox(null);
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
    };

    const formatTime = (t: number) => {
        const mins = Math.floor(t / 60);
        const secs = Math.floor(t % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <Box sx={{ p: 3, maxWidth: 1200, mx: 'auto' }}>
            <Typography variant="h4" fontWeight="bold" gutterBottom>
                🎬 Video Editor
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Select a region and time range, then apply inpainting or privacy redaction (blur).
            </Typography>

            {/* Video Path Input */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <VideoFile color="primary" />
                        <TextField
                            fullWidth
                            size="small"
                            label="Video Path"
                            placeholder="e.g., D:\Videos\sample.mp4"
                            value={videoPath}
                            onChange={(e) => setVideoPath(e.target.value)}
                        />
                        <Button
                            variant="contained"
                            onClick={handleLoadVideo}
                            disabled={!videoPath}
                        >
                            Load
                        </Button>
                    </Stack>
                </CardContent>
            </Card>

            {/* Video Player with Canvas Overlay */}
            <Paper
                sx={{
                    position: 'relative',
                    mb: 3,
                    overflow: 'hidden',
                    bgcolor: '#000',
                    borderRadius: 2,
                }}
            >
                <video
                    ref={videoRef}
                    style={{ width: '100%', display: 'block' }}
                    onLoadedMetadata={handleVideoLoaded}
                    onTimeUpdate={handleTimeUpdate}
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                />
                <canvas
                    ref={canvasRef}
                    width={640}
                    height={360}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        cursor: 'crosshair',
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                />

                {/* Controls Overlay */}
                <Box
                    sx={{
                        position: 'absolute',
                        bottom: 0,
                        left: 0,
                        right: 0,
                        p: 2,
                        background: 'linear-gradient(transparent, rgba(0,0,0,0.8))',
                    }}
                >
                    <Stack direction="row" spacing={2} alignItems="center">
                        <IconButton onClick={togglePlay} sx={{ color: 'white' }}>
                            {isPlaying ? <Pause /> : <PlayArrow />}
                        </IconButton>
                        <Typography variant="caption" color="white" sx={{ minWidth: 50 }}>
                            {formatTime(currentTime)}
                        </Typography>
                        <Slider
                            size="small"
                            value={currentTime}
                            max={duration}
                            onChange={handleSeek}
                            sx={{ color: 'white' }}
                        />
                        <Typography variant="caption" color="white" sx={{ minWidth: 50 }}>
                            {formatTime(duration)}
                        </Typography>
                    </Stack>
                </Box>
            </Paper>

            {/* Time Range Selector */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                        ⏱️ Time Range for Effect
                    </Typography>
                    <Stack direction="row" spacing={3} alignItems="center">
                        <Box sx={{ flex: 1 }}>
                            <Typography variant="caption">Start: {formatTime(startTime)}</Typography>
                            <Slider
                                size="small"
                                value={startTime}
                                max={duration}
                                onChange={(_, v) => setStartTime(v as number)}
                            />
                        </Box>
                        <Box sx={{ flex: 1 }}>
                            <Typography variant="caption">End: {formatTime(endTime)}</Typography>
                            <Slider
                                size="small"
                                value={endTime}
                                max={duration}
                                onChange={(_, v) => setEndTime(v as number)}
                            />
                        </Box>
                    </Stack>
                </CardContent>
            </Card>

            {/* Selection Info & Actions */}
            <Card sx={{ mb: 3 }}>
                <CardContent>
                    <Stack direction="row" spacing={2} alignItems="center" justifyContent="space-between">
                        <Box>
                            <Typography variant="subtitle2" fontWeight="bold">
                                🎯 Selected Region
                            </Typography>
                            {bbox ? (
                                <Typography variant="caption" color="text.secondary">
                                    x: {bbox.x}, y: {bbox.y}, width: {bbox.width}, height: {bbox.height}
                                </Typography>
                            ) : (
                                <Typography variant="caption" color="text.secondary">
                                    Draw a rectangle on the video to select a region
                                </Typography>
                            )}
                        </Box>

                        <Stack direction="row" spacing={1}>
                            <Tooltip title="Clear Selection">
                                <IconButton onClick={clearSelection}>
                                    <RestartAlt />
                                </IconButton>
                            </Tooltip>
                            <Divider orientation="vertical" flexItem />

                            <Tooltip title="Remove Object (AI Inpaint)">
                                <span>
                                    <Button
                                        variant="contained"
                                        color="secondary"
                                        startIcon={
                                            inpaintMutation.isPending ? (
                                                <CircularProgress size={16} color="inherit" />
                                            ) : (
                                                <AutoFixHigh />
                                            )
                                        }
                                        onClick={handleInpaint}
                                        disabled={!bbox || inpaintMutation.isPending}
                                    >
                                        Inpaint
                                    </Button>
                                </span>
                            </Tooltip>

                            <Tooltip title="Blur Region (Privacy)">
                                <span>
                                    <Button
                                        variant="contained"
                                        color="error"
                                        startIcon={
                                            redactMutation.isPending ? (
                                                <CircularProgress size={16} color="inherit" />
                                            ) : (
                                                <BlurOn />
                                            )
                                        }
                                        onClick={handleRedact}
                                        disabled={!bbox || redactMutation.isPending}
                                    >
                                        Redact
                                    </Button>
                                </span>
                            </Tooltip>
                        </Stack>
                    </Stack>
                </CardContent>
            </Card>

            {/* Job Progress */}
            {activeJobId && jobData && (
                <Alert severity="info" sx={{ mb: 2 }}>
                    <Stack direction="row" spacing={2} alignItems="center">
                        <CircularProgress size={20} />
                        <Box>
                            <Typography variant="body2" fontWeight="bold">
                                Job: {activeJobId}
                            </Typography>
                            <Typography variant="caption">
                                Status: {jobData.status} | Progress: {(jobData.progress * 100).toFixed(0)}%
                            </Typography>
                        </Box>
                    </Stack>
                </Alert>
            )}

            <Snackbar
                open={!!snackMessage}
                autoHideDuration={5000}
                onClose={() => setSnackMessage(null)}
                message={snackMessage}
            />
        </Box>
    );
}
