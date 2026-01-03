import { useState, useEffect, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
    Box,
    Typography,
    TextField,
    Button,
    Paper,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    LinearProgress,
    Chip,
    Alert,
    List,
    ListItem,
    ListItemText,
    IconButton,
    Tooltip,
} from '@mui/material';
import { CloudUpload, Stop, FolderOpen, Add, Delete, Pause, PlayArrow } from '@mui/icons-material';

import { ingestMedia, getJobs, cancelJob, pauseJob, resumeJob, deleteLibraryItem } from '../api/client';

type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';

interface Job {
    job_id: string;
    status: JobStatus;
    progress: number;
    file_path: string;
    media_type: string;
    current_stage: string;
    pipeline_stage: string;
    message: string;
    started_at: number;
    completed_at: number | null;
    error: string | null;
    current_item_index?: number;
    total_items?: number;
    processed_frames?: number;
    total_frames?: number;
    timestamp?: number;
    duration?: number;
}

function JobCard({ job, onCancel, onPause, onResume }: { job: Job; onCancel: (id: string) => void, onPause: (id: string) => void, onResume: (id: string) => void }) {
    const isRunning = job.status === 'running';
    const isPaused = job.status === 'paused';
    const fileName = job.file_path.split(/[/\\]/).pop() || job.file_path;

    // Granular Stats
    const framesText = (job.processed_frames !== undefined && job.total_frames)
        ? `Frames: ${job.processed_frames} / ${job.total_frames}`
        : '';

    const timeText = (job.timestamp !== undefined && job.duration)
        ? `Time: ${new Date(job.timestamp * 1000).toISOString().substr(11, 8)} / ${new Date(job.duration * 1000).toISOString().substr(11, 8)}`
        : '';

    // Calculate accurate percentage if available
    let progress = job.progress;
    if (job.duration && job.timestamp) {
        progress = (job.timestamp / job.duration) * 100;
        // Adjust for stage coverage if needed, but timestamp is best indicator for user
    }

    return (
        <Paper sx={{ p: 2, mb: 1.5, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 1 }}>
                <Box>
                    <Typography variant="body2" fontWeight={600} noWrap sx={{ maxWidth: 300 }} title={job.file_path}>
                        {fileName}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        {job.message || job.current_stage || 'Initializing...'}
                    </Typography>
                </Box>
                <Chip
                    size="small"
                    label={job.status}
                    color={
                        job.status === 'completed' ? 'success' :
                            job.status === 'failed' ? 'error' :
                                isRunning ? 'primary' :
                                    isPaused ? 'warning' : 'default'
                    }
                />
            </Box>

            {(isRunning || isPaused) && (
                <>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{ flex: 1, height: 6, borderRadius: 3 }}
                        />
                        <Typography variant="caption" fontWeight={600} sx={{ minWidth: 35 }}>
                            {Math.round(progress)}%
                        </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            {framesText && <Typography variant="caption" color="text.secondary">{framesText}</Typography>}
                            {timeText && <Typography variant="caption" color="text.secondary">{timeText}</Typography>}
                        </Box>
                        <Box>
                            {isRunning && (
                                <Tooltip title="Pause">
                                    <IconButton size="small" onClick={() => onPause(job.job_id)} color="primary">
                                        <Pause fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                            {isPaused && (
                                <Tooltip title="Resume">
                                    <IconButton size="small" onClick={() => onResume(job.job_id)} color="success">
                                        <PlayArrow fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                            <Tooltip title="Stop">
                                <IconButton size="small" onClick={() => onCancel(job.job_id)} color="error">
                                    <Stop fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>
                </>
            )}
            {job.status === 'failed' && job.error && (
                <Typography variant="caption" color="error">{job.error}</Typography>
            )}
            {job.status === 'paused' && !job.message && (
                <Typography variant="caption" color="warning.main">Job is paused. Click resume to continue.</Typography>
            )}
        </Paper>
    );
}

export default function IngestPage() {
    const [paths, setPaths] = useState<string[]>(['']);
    const [mediaType, setMediaType] = useState('unknown');
    const [startTime, setStartTime] = useState('');
    const [endTime, setEndTime] = useState('');
    const [pendingJobs, setPendingJobs] = useState<string[]>([]);
    const queryClient = useQueryClient();
    const eventSourceRef = useRef<EventSource | null>(null);

    const jobs = useQuery({
        queryKey: ['jobs'],
        queryFn: getJobs,
        refetchInterval: 2000,
    });

    // Parse time string (supports s, m:s, h:m:s formats), returns undefined if invalid
    const parseTime = (t: string): number | undefined => {
        if (!t.trim()) return undefined;
        const parts = t.split(':').map(Number);
        if (parts.some(isNaN)) return undefined;  // Invalid input = use full video
        if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];  // h:m:s
        if (parts.length === 2) return parts[0] * 60 + parts[1];  // m:s
        if (parts.length === 1) return parts[0];  // seconds only
        return undefined;
    };

    const ingestMutation = useMutation({
        mutationFn: (data: { path: string; hint: string; start?: number; end?: number }) =>
            ingestMedia(data.path, data.hint, data.start, data.end),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
        },
    });

    const cancelMutation = useMutation({
        mutationFn: cancelJob,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
    });

    const pauseMutation = useMutation({
        mutationFn: pauseJob,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
    });

    const resumeMutation = useMutation({
        mutationFn: resumeJob,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
    });

    const deleteMutation = useMutation({
        mutationFn: deleteLibraryItem,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
    });

    // SSE connection
    useEffect(() => {
        const es = new EventSource('http://localhost:8000/events');
        eventSourceRef.current = es;
        es.onmessage = () => queryClient.invalidateQueries({ queryKey: ['jobs'] });
        es.onerror = () => {
            es.close();
            setTimeout(() => {
                const newEs = new EventSource('http://localhost:8000/events');
                eventSourceRef.current = newEs;
                newEs.onmessage = () => queryClient.invalidateQueries({ queryKey: ['jobs'] });
            }, 5000);
        };
        return () => es.close();
    }, [queryClient]);

    const handleSubmit = () => {
        const validPaths = paths.filter(p => p.trim());
        if (validPaths.length === 0) return;

        const start = parseTime(startTime);
        const end = parseTime(endTime);
        validPaths.forEach(path => {
            setPendingJobs(prev => [...prev, path.trim()]);
            ingestMutation.mutate({ path: path.trim(), hint: mediaType, start, end });
        });
        setPaths(['']);
        setStartTime('');
        setEndTime('');
    };

    // Helper to normalize path for comparison (handles Windows/Linux differences)
    const normalizePath = (p: string) => p.replace(/\\/g, '/').toLowerCase();
    const getBasename = (p: string) => p.split(/[/\\]/).pop()?.toLowerCase() || '';

    // Clear pending jobs when they appear in real jobs
    useEffect(() => {
        if (!jobs.data?.jobs?.length) return;

        const realJobPaths = jobs.data.jobs.map((j: Job) => normalizePath(j.file_path));
        const realJobBasenames = jobs.data.jobs.map((j: Job) => getBasename(j.file_path));

        setPendingJobs(prev => prev.filter(p => {
            const normalizedPending = normalizePath(p);
            const pendingBasename = getBasename(p);
            // Clear if exact path matches OR if basename matches (fallback)
            return !realJobPaths.includes(normalizedPending) && !realJobBasenames.includes(pendingBasename);
        }));
    }, [jobs.data]);

    const addPath = () => setPaths([...paths, '']);
    const removePath = (idx: number) => setPaths(paths.filter((_, i) => i !== idx));
    const updatePath = (idx: number, value: string) => {
        const newPaths = [...paths];
        newPaths[idx] = value;
        setPaths(newPaths);
    };

    const activeJobs = jobs.data?.jobs?.filter((j: Job) =>
        ['running', 'paused', 'pending'].includes(j.status)
    ) || [];

    const historyJobs = jobs.data?.jobs?.filter((j: Job) =>
        !['running', 'paused', 'pending'].includes(j.status)
    ) || [];

    return (
        <Box>
            <Typography variant="h5" fontWeight={700} gutterBottom>Ingest Media</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                Add files or directories. Supports Windows and Linux paths.
            </Typography>

            {/* Input Form */}
            <Paper sx={{ p: 2, borderRadius: 2, mb: 3 }}>
                {paths.map((path, idx) => (
                    <Box key={idx} sx={{ display: 'flex', gap: 1, mb: 1 }}>
                        <TextField
                            fullWidth
                            size="small"
                            placeholder="/path/to/video.mp4 or D:\Videos\movie.mkv"
                            value={path}
                            onChange={(e) => updatePath(idx, e.target.value)}
                            slotProps={{
                                input: { startAdornment: <FolderOpen sx={{ mr: 1, color: 'action.active' }} fontSize="small" /> }
                            }}
                        />
                        {paths.length > 1 && (
                            <IconButton size="small" onClick={() => removePath(idx)} color="error">
                                <Delete fontSize="small" />
                            </IconButton>
                        )}
                    </Box>
                ))}

                <Box sx={{ display: 'flex', gap: 1, mt: 1.5, flexWrap: 'wrap' }}>
                    <Button size="small" startIcon={<Add />} onClick={addPath} variant="outlined">
                        Add Path
                    </Button>
                    <Button
                        size="small"
                        startIcon={<FolderOpen />}
                        onClick={async () => {
                            try {
                                const { browseFileSystem } = await import('../api/client');
                                const path = await browseFileSystem();
                                if (path) {
                                    const newPaths = [...paths];
                                    if (newPaths[newPaths.length - 1] === '') {
                                        newPaths[newPaths.length - 1] = path;
                                    } else {
                                        newPaths.push(path);
                                    }
                                    setPaths(newPaths);
                                }
                            } catch (e) {
                                console.error(e);
                            }
                        }}
                        variant="outlined"
                        color="secondary"
                    >
                        Browse System
                    </Button>
                    <TextField
                        size="small"
                        sx={{ width: 100 }}
                        placeholder="0:00:00"
                        label="Start"
                        value={startTime}
                        onChange={(e) => setStartTime(e.target.value)}
                    />
                    <TextField
                        size="small"
                        sx={{ width: 100 }}
                        placeholder="(full)"
                        label="End"
                        value={endTime}
                        onChange={(e) => setEndTime(e.target.value)}
                    />
                    <FormControl size="small" sx={{ minWidth: 120 }}>
                        <InputLabel>Type</InputLabel>
                        <Select value={mediaType} label="Type" onChange={(e) => setMediaType(e.target.value)}>
                            <MenuItem value="unknown">Auto</MenuItem>
                            <MenuItem value="movie">Movie</MenuItem>
                            <MenuItem value="tv">TV</MenuItem>
                            <MenuItem value="personal">Personal</MenuItem>
                        </Select>
                    </FormControl>
                    <Button
                        variant="contained"
                        startIcon={<CloudUpload />}
                        onClick={handleSubmit}
                        disabled={!paths.some(p => p.trim()) || ingestMutation.isPending}
                    >
                        Start
                    </Button>
                </Box>

                {ingestMutation.isError && (
                    <Alert severity="error" sx={{ mt: 1.5 }}>
                        {(ingestMutation.error as Error)?.message || 'Failed to start'}
                    </Alert>
                )}
            </Paper>

            {/* Pending Jobs - Shows immediately after clicking ingest */}
            {pendingJobs.length > 0 && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                        Submitting... ({pendingJobs.length})
                    </Typography>
                    {pendingJobs.map((path, idx) => (
                        <Paper key={idx} sx={{ p: 2, mb: 1.5, borderRadius: 2 }}>
                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="body2" fontWeight={600} noWrap sx={{ maxWidth: '60%' }}>
                                    {path.split(/[/\\]/).pop() || path}
                                </Typography>
                                <Chip size="small" label="submitting" color="warning" />
                            </Box>
                            <LinearProgress sx={{ height: 6, borderRadius: 3 }} />
                        </Paper>
                    ))}
                </Box>
            )}

            {/* Active Jobs */}
            {activeJobs.length > 0 && (
                <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                        Active Jobs ({activeJobs.length})
                    </Typography>
                    {activeJobs.map((job: Job) => (
                        <JobCard
                            key={job.job_id}
                            job={job}
                            onCancel={(id) => cancelMutation.mutate(id)}
                            onPause={(id) => pauseMutation.mutate(id)}
                            onResume={(id) => resumeMutation.mutate(id)}
                        />
                    ))}
                </Box>
            )}

            {/* History */}
            {historyJobs.length > 0 && (
                <Box>
                    <Typography variant="subtitle2" fontWeight={600} gutterBottom>History</Typography>
                    <Paper sx={{ borderRadius: 2 }}>
                        <List dense disablePadding>
                            {historyJobs.map((job: Job, idx: number) => (
                                <ListItem
                                    key={job.job_id}
                                    divider={idx < historyJobs.length - 1}
                                    secondaryAction={
                                        <Tooltip title="Delete from Library">
                                            <IconButton edge="end" aria-label="delete" onClick={() => {
                                                if (window.confirm('Delete this media from the library? This will remove all index data and thumbnails.')) {
                                                    deleteMutation.mutate(job.file_path);
                                                }
                                            }}>
                                                <Delete fontSize="small" />
                                            </IconButton>
                                        </Tooltip>
                                    }
                                >
                                    <ListItemText
                                        primary={job.file_path.split(/[/\\]/).pop()}
                                        secondary={
                                            <>
                                                {job.completed_at ? new Date(job.completed_at * 1000).toLocaleString() : job.status}
                                                {job.status === 'failed' && ` - ${job.error}`}
                                            </>
                                        }
                                        sx={{ mr: 4 }}
                                    />
                                    <Chip
                                        size="small"
                                        label={job.status}
                                        color={job.status === 'completed' ? 'success' : job.status === 'failed' ? 'error' : 'default'}
                                        sx={{ mr: 2 }}
                                    />
                                </ListItem>
                            ))}
                        </List>
                    </Paper>
                </Box>
            )}

            {/* Empty State */}
            {!jobs.isLoading && jobs.data?.jobs?.length === 0 && pendingJobs.length === 0 && (
                <Paper sx={{ p: 4, textAlign: 'center', borderRadius: 2, bgcolor: 'action.hover' }}>
                    <CloudUpload sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
                    <Typography color="text.secondary">No jobs yet. Add a file path above.</Typography>
                </Paper>
            )}
        </Box>
    );
}
