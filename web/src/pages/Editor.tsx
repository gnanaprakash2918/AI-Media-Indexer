import { useState, useRef, useEffect, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
    Box,
    Typography,
    Paper,
    Button,
    Grid,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Slider,
    CircularProgress,
    Alert,
    IconButton,
    Tooltip,
} from '@mui/material';
import {
    Brush,
    HideImage,
    Undo,
    Save,
    PlayArrow,
    Pause,
    ContentCut,
} from '@mui/icons-material';

import {
    getLibrary,
    getMasklets,
    triggerInpaint,
    triggerRedact,
    getManipulationJob,
    type RegionRequest,
    type ManipulationJob,
} from '../api/client';

export default function EditorPage() {
    const [selectedVideo, setSelectedVideo] = useState('');
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);

    // Selection state
    const [selection, setSelection] = useState<number[] | null>(null); // [x, y, w, h]
    const [startTime, setStartTime] = useState<number | null>(null);
    const [endTime, setEndTime] = useState<number | null>(null);

    // Job tracking
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Queries
    const { data: libraryData } = useQuery({
        queryKey: ['library'],
        queryFn: getLibrary
    });

    const videos: string[] = (libraryData?.media || []).map((m: any) => m.video_path);

    // Job Polling
    const jobQuery = useQuery<ManipulationJob>({
        queryKey: ['job', activeJobId],
        queryFn: () => getManipulationJob(activeJobId!),
        enabled: !!activeJobId,
        refetchInterval: (query) => {
            const data = query.state.data;
            if (!data || ['completed', 'failed'].includes(data.status)) {
                return false;
            }
            return 1000;
        },
    });

    // Mutations
    const inpaintMutation = useMutation({
        mutationFn: triggerInpaint,
        onSuccess: (data) => setActiveJobId(data.job_id),
    });

    const redactMutation = useMutation({
        mutationFn: triggerRedact,
        onSuccess: (data) => setActiveJobId(data.job_id),
    });

    // Selection Logic
    const handleMouseDown = (e: React.MouseEvent) => {
        if (!canvasRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Simple drag to draw rect logic would go here
        // For MVP, we'll just set a start point
        setSelection([x, y, 0, 0]);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (e.buttons !== 1 || !selection) return;
        const rect = canvasRef.current!.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const startX = selection[0];
        const startY = selection[1];

        setSelection([
            Math.min(startX, x),
            Math.min(startY, y),
            Math.abs(x - startX),
            Math.abs(y - startY)
        ]);
    };

    const handleApply = (mode: 'inpaint' | 'redact') => {
        if (!selectedVideo || !selection || startTime === null || endTime === null) return;

        // Scale selection to video resolution
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        const scaleX = video.videoWidth / canvas.width;
        const scaleY = video.videoHeight / canvas.height;

        const bbox = [
            Math.round(selection[0] * scaleX),
            Math.round(selection[1] * scaleY),
            Math.round(selection[2] * scaleX),
            Math.round(selection[3] * scaleY),
        ];

        const req: RegionRequest = {
            video_path: selectedVideo,
            start_time: startTime,
            end_time: endTime,
            bbox: bbox,
        };

        if (mode === 'inpaint') {
            inpaintMutation.mutate(req);
        } else {
            redactMutation.mutate(req);
        }
    };

    return (
        <Box>
            <Box sx={{ mb: 3 }}>
                <Typography variant="h5" fontWeight={700}>Video Editor</Typography>
                <Typography variant="body2" color="text.secondary">
                    Remove objects or redact sensitive information.
                </Typography>
            </Box>

            <Grid container spacing={3}>
                <Grid size={{ xs: 12, md: 8 }}>
                    <Paper sx={{ p: 0, overflow: 'hidden', position: 'relative', bgcolor: 'black' }}>
                        {selectedVideo ? (
                            <Box sx={{ position: 'relative' }}>
                                <video
                                    ref={videoRef}
                                    src={`http://localhost:8000/media?path=${encodeURIComponent(selectedVideo)}`}
                                    style={{ width: '100%', display: 'block' }}
                                    onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                                    onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                                />

                                {/* Overlay Canvas for Selection */}
                                <canvas
                                    ref={canvasRef}
                                    style={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: '100%',
                                        cursor: 'crosshair',
                                    }}
                                    width={640} // Scaled internally
                                    height={360}
                                    onMouseDown={handleMouseDown}
                                    onMouseMove={handleMouseMove}
                                />

                                {/* Selection Box Render */}
                                {selection && (
                                    <Box
                                        sx={{
                                            position: 'absolute',
                                            left: selection[0], // These coordinates need to match canvas scaling
                                            top: selection[1],
                                            width: selection[2],
                                            height: selection[3],
                                            border: '2px dashed red',
                                            bgcolor: 'rgba(255, 0, 0, 0.2)',
                                            pointerEvents: 'none',
                                        }}
                                    />
                                )}
                            </Box>
                        ) : (
                            <Box sx={{ height: 400, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                <Typography color="text.secondary">Select a video to start editing</Typography>
                            </Box>
                        )}

                        {/* Controls */}
                        <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                <IconButton onClick={() => videoRef.current?.paused ? videoRef.current.play() : videoRef.current?.pause()}>
                                    {isPlaying ? <Pause /> : <PlayArrow />}
                                </IconButton>
                                <Slider
                                    value={currentTime}
                                    max={duration}
                                    onChange={(_, v) => { videoRef.current!.currentTime = v as number; }}
                                    sx={{ flex: 1 }}
                                />
                                <Typography variant="caption">{currentTime.toFixed(1)} / {duration.toFixed(1)}s</Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12, md: 4 }}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <FormControl fullWidth size="small" sx={{ mb: 3 }}>
                            <InputLabel>Source Video</InputLabel>
                            <Select
                                value={selectedVideo}
                                label="Source Video"
                                onChange={(e) => setSelectedVideo(e.target.value)}
                            >
                                {videos.map(v => (
                                    <MenuItem key={v} value={v}>{v.split(/[/\\]/).pop()}</MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Box sx={{ mb: 3 }}>
                            <Typography variant="subtitle2" gutterBottom>1. Select Time Range</Typography>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={() => setStartTime(currentTime)}
                                >
                                    Set Start ({startTime?.toFixed(1) || '-'})
                                </Button>
                                <Button
                                    variant="outlined"
                                    size="small"
                                    onClick={() => setEndTime(currentTime)}
                                >
                                    Set End ({endTime?.toFixed(1) || '-'})
                                </Button>
                            </Box>
                        </Box>

                        <Box sx={{ mb: 3 }}>
                            <Typography variant="subtitle2" gutterBottom>2. Select Region</Typography>
                            <Typography variant="caption" color="text.secondary">
                                Click and drag on the video to select the area to modify.
                            </Typography>
                        </Box>

                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                            <Button
                                variant="contained"
                                color="error" // Inpaint is destructive/removal
                                startIcon={<Brush />}
                                onClick={() => handleApply('inpaint')}
                                disabled={!selection || startTime === null}
                            >
                                Inpaint (Remove Object)
                            </Button>
                            <Button
                                variant="contained"
                                color="inherit"
                                startIcon={<HideImage />}
                                onClick={() => handleApply('redact')}
                                disabled={!selection || startTime === null}
                            >
                                Redact (Blur Face/Text)
                            </Button>
                        </Box>

                        {/* Job Status Card */}
                        {activeJobId && (
                            <Alert
                                severity={jobQuery.data?.status === 'completed' ? 'success' : 'info'}
                                sx={{ mt: 3 }}
                            >
                                <Typography variant="subtitle2">Job Status: {jobQuery.data?.status}</Typography>
                                {jobQuery.data?.progress !== undefined && (
                                    <Typography variant="caption">{jobQuery.data.progress.toFixed(1)}%</Typography>
                                )}
                                {jobQuery.data?.result_path && (
                                    <Typography variant="caption" display="block" sx={{ mt: 1, wordBreak: 'break-all' }}>
                                        Result: {jobQuery.data.result_path}
                                    </Typography>
                                )}
                            </Alert>
                        )}

                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
}
