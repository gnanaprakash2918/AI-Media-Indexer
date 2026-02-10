import { useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Button,
    Chip,
    LinearProgress,
    IconButton,
    Tooltip,
    Alert,
} from '@mui/material';
import {
    Stop,
    Pause,
    PlayArrow,
    GpsFixed,
    Description as SummaryIcon,
} from '@mui/icons-material';

import type { Job, StageStats } from '../api/client';

// Pipeline stage labels for human-readable display
export const STAGE_LABELS: Record<string, { label: string; icon: string; order: number }> = {
    init: { label: 'Initializing', icon: '⚙️', order: 0 },
    extract: { label: 'Extracting Audio', icon: '🎵', order: 1 },
    transcribe: { label: 'Transcribing', icon: '📝', order: 2 },
    diarize: { label: 'Speaker Diarization', icon: '🗣️', order: 3 },
    voice_embed: { label: 'Voice Embeddings', icon: '🎤', order: 4 },
    face_detect: { label: 'Face Detection', icon: '👤', order: 5 },
    face_track: { label: 'Face Tracking', icon: '🔗', order: 6 },
    vlm_caption: { label: 'Frame Analysis (VLM)', icon: '🖼️', order: 7 },
    index: { label: 'Indexing', icon: '📊', order: 8 },
    complete: { label: 'Complete', icon: '✅', order: 9 },
};

// Get all stages in order
export const ORDERED_STAGES = Object.entries(STAGE_LABELS)
    .sort(([, a], [, b]) => a.order - b.order)
    .map(([key]) => key);

// Format seconds to MM:SS or HH:MM:SS
export function formatTime(seconds: number): string {
    if (!seconds || seconds < 0) return '00:00';
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h > 0) {
        return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// Format duration in human-readable format (e.g., "2m 30s", "1h 15m")
export function formatDuration(seconds: number | null | undefined): string {
    if (!seconds || seconds <= 0) return '';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) {
        const m = Math.floor(seconds / 60);
        const s = Math.round(seconds % 60);
        return s > 0 ? `${m}m ${s}s` : `${m}m`;
    }
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

// Format ETA - prefer backend calculation when available
export function formatETA(etaSeconds: number | null | undefined): string | null {
    if (!etaSeconds || etaSeconds <= 0) return null;
    if (etaSeconds < 60) return `~${Math.ceil(etaSeconds)}s left`;
    if (etaSeconds < 3600) return `~${Math.ceil(etaSeconds / 60)}m left`;
    return `~${Math.round(etaSeconds / 3600 * 10) / 10}h left`;
}

export default function JobCard({
    job,
    onCancel,
    onPause,
    onResume,
    onGround,
    onShowSummary,
}: {
    job: Job;
    onCancel: (id: string) => void;
    onPause: (id: string) => void;
    onResume: (id: string) => void;
    onGround: (path: string) => void;
    onShowSummary: (path: string) => void;
}) {
    const isRunning = job.status === 'running';
    const isPaused = job.status === 'paused';
    const fileName = job.file_path.split(/[/\\]/).pop() || job.file_path;

    // Get current stage info
    const currentStage = job.pipeline_stage || 'init';
    const stageInfo = STAGE_LABELS[currentStage] || STAGE_LABELS.init;
    const currentStageIndex = ORDERED_STAGES.indexOf(currentStage);

    // Granular Stats
    const hasTimeData = job.timestamp !== undefined && job.duration;

    // Use weighted progress from backend for more accurate display
    const progress = job.weighted_progress ?? job.progress;

    // Use backend ETA (more accurate as it tracks actual processing speed)
    const eta = formatETA(job.eta_seconds);

    // Calculate elapsed time for running jobs
    const elapsed = isRunning || isPaused
        ? Math.floor(Date.now() / 1000 - job.started_at)
        : null;

    // Calculate total job duration for completed jobs
    const totalJobDuration = job.completed_at && job.started_at
        ? job.completed_at - job.started_at
        : null;

    return (
        <Paper sx={{ p: 2, mb: 1.5, borderRadius: 2 }}>
            {/* Header: Filename + Status */}
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'flex-start',
                    mb: 1,
                }}
            >
                <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography
                        variant="body2"
                        fontWeight={600}
                        noWrap
                        sx={{ maxWidth: '100%' }}
                        title={job.file_path}
                    >
                        {fileName}
                    </Typography>
                    {/* Current Stage - Prominent Display */}
                    {(isRunning || isPaused) && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
                            <Typography variant="body2" sx={{ fontSize: '1.1em' }}>
                                {stageInfo.icon}
                            </Typography>
                            <Typography variant="body2" fontWeight={500} color="primary">
                                {stageInfo.label}
                            </Typography>
                            {job.message && job.message !== stageInfo.label && (
                                <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                                    — {job.message}
                                </Typography>
                            )}
                        </Box>
                    )}
                </Box>
                <Chip
                    size="small"
                    label={job.status}
                    color={
                        job.status === 'completed'
                            ? 'success'
                            : job.status === 'failed'
                                ? 'error'
                                : isRunning
                                    ? 'primary'
                                    : isPaused
                                        ? 'warning'
                                        : 'default'
                    }
                />
            </Box>

            {/* Progress Section - Only for active jobs */}
            {(isRunning || isPaused) && (
                <>
                    {/* Main Progress Bar */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <LinearProgress
                            variant="determinate"
                            value={Math.min(100, Math.max(0, progress))}
                            sx={{ flex: 1, height: 8, borderRadius: 4 }}
                        />
                        <Typography variant="body2" fontWeight={700} sx={{ minWidth: 45 }}>
                            {Math.round(progress)}%
                        </Typography>
                    </Box>

                    {/* Detailed Stats Row */}
                    <Box
                        sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            mb: 1.5,
                            flexWrap: 'wrap',
                            gap: 1,
                        }}
                    >
                        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                            {/* Frame Progress */}
                            {job.processed_frames !== undefined && (
                                <Typography variant="caption" sx={{
                                    bgcolor: 'action.hover',
                                    px: 1,
                                    py: 0.25,
                                    borderRadius: 1,
                                    fontFamily: 'monospace'
                                }}>
                                    🎞️ {job.processed_frames} {job.total_frames ? `/ ${job.total_frames}` : ''} frames
                                </Typography>
                            )}
                            {/* Time Progress */}
                            {hasTimeData && (
                                <Typography variant="caption" sx={{
                                    bgcolor: 'action.hover',
                                    px: 1,
                                    py: 0.25,
                                    borderRadius: 1,
                                    fontFamily: 'monospace'
                                }}>
                                    ⏱️ {formatTime(job.timestamp!)} / {formatTime(job.duration!)}
                                </Typography>
                            )}
                            {/* Speed Metrics */}
                            {job.speed && job.speed.speed_ratio > 0 && (
                                <Typography variant="caption" sx={{
                                    bgcolor: 'info.main',
                                    color: 'info.contrastText',
                                    px: 1,
                                    py: 0.25,
                                    borderRadius: 1,
                                    fontWeight: 600
                                }}>
                                    ⚡ {job.speed.speed_ratio.toFixed(1)}x {job.speed.fps > 0 ? `(${job.speed.fps.toFixed(1)} fps)` : ''}
                                </Typography>
                            )}
                            {/* Elapsed Time */}
                            {elapsed && elapsed > 0 && (
                                <Typography variant="caption" sx={{
                                    bgcolor: 'action.hover',
                                    px: 1,
                                    py: 0.25,
                                    borderRadius: 1,
                                }}>
                                    ⏳ {formatDuration(elapsed)}
                                </Typography>
                            )}
                            {/* ETA */}
                            {eta && (
                                <Typography variant="caption" color="success.main" fontWeight={600}>
                                    {eta}
                                </Typography>
                            )}
                        </Box>

                        {/* Action Buttons */}
                        <Box>
                            {isRunning && (
                                <Tooltip title="Pause">
                                    <IconButton
                                        size="small"
                                        onClick={() => onPause(job.job_id)}
                                        color="primary"
                                    >
                                        <Pause fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                            {isPaused && (
                                <Tooltip title="Resume">
                                    <IconButton
                                        size="small"
                                        onClick={() => onResume(job.job_id)}
                                        color="success"
                                    >
                                        <PlayArrow fontSize="small" />
                                    </IconButton>
                                </Tooltip>
                            )}
                            <Tooltip title="Stop">
                                <IconButton
                                    size="small"
                                    onClick={() => onCancel(job.job_id)}
                                    color="error"
                                >
                                    <Stop fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    </Box>

                    {/* Pipeline Stage Checklist */}
                    <Box sx={{
                        display: 'flex',
                        flexWrap: 'wrap',
                        gap: 0.5,
                        mt: 1,
                        pt: 1,
                        borderTop: 1,
                        borderColor: 'divider'
                    }}>
                        {ORDERED_STAGES.filter(s => s !== 'complete').map((stage, idx) => {
                            const info = STAGE_LABELS[stage];
                            const stats = job.stage_stats?.[stage];

                            let isDone = idx < currentStageIndex;
                            let isCurrent = stage === currentStage;
                            let isSkipped = false;
                            let duration: number | null = null;

                            if (stats) {
                                if (stats.status === 'completed') isDone = true;
                                if (stats.status === 'running') isCurrent = true;
                                if (stats.status === 'skipped') isSkipped = true;
                                if (stats.duration) duration = stats.duration;
                            } else if (job.pipeline_stage === 'complete') {
                                isDone = true;
                                isCurrent = false;
                            }

                            const durationStr = duration ? (duration < 60 ? `${duration.toFixed(1)}s` : `${(duration / 60).toFixed(1)}m`) : '';

                            return (
                                <Tooltip
                                    key={stage}
                                    title={
                                        <Box sx={{ textAlign: 'center' }}>
                                            <Typography variant="caption" fontWeight="bold">{info.label}</Typography>
                                            {duration && <Typography variant="body2">{duration.toFixed(2)}s</Typography>}
                                            {isSkipped && <Typography variant="caption" color="warning.light">Skipped</Typography>}
                                            {stats?.retries ? <Typography variant="caption" color="warning.light">{stats.retries} retries</Typography> : null}
                                            {stats?.error && <Typography variant="caption" color="error.light">{stats.error}</Typography>}
                                        </Box>
                                    }
                                >
                                    <Chip
                                        size="small"
                                        label={`${info.icon} ${info.label.split(' ')[0]}${durationStr ? ` (${durationStr})` : ''}`}
                                        variant={isDone || isSkipped ? 'filled' : 'outlined'}
                                        color={
                                            isCurrent ? 'primary'
                                                : isSkipped ? 'default'
                                                    : isDone ? 'success'
                                                        : 'default'
                                        }
                                        sx={{
                                            opacity: isDone || isCurrent || isSkipped ? 1 : 0.5,
                                            fontWeight: isCurrent ? 700 : 400,
                                            textDecoration: isSkipped ? 'line-through' : 'none',
                                            animation: isCurrent ? 'pulse 1.5s infinite' : 'none',
                                            '@keyframes pulse': {
                                                '0%, 100%': { opacity: 1 },
                                                '50%': { opacity: 0.7 },
                                            },
                                        }}
                                    />
                                </Tooltip>
                            );
                        })}
                    </Box>
                </>
            )}

            {/* Completed State Actions */}
            {job.status === 'completed' && (
                <Box sx={{ mt: 1.5 }}>
                    <Box sx={{ display: 'flex', gap: 2, mb: 1.5, flexWrap: 'wrap', alignItems: 'center' }}>
                        {totalJobDuration && (
                            <Typography variant="caption" sx={{
                                bgcolor: 'success.main',
                                color: 'success.contrastText',
                                px: 1,
                                py: 0.25,
                                borderRadius: 1,
                                fontWeight: 600
                            }}>
                                ✅ Completed in {formatDuration(totalJobDuration)}
                            </Typography>
                        )}
                        {job.completed_at && (
                            <Typography variant="caption" color="text.secondary">
                                {new Date(job.completed_at * 1000).toLocaleString()}
                            </Typography>
                        )}
                        {job.total_frames && job.total_frames > 0 && (
                            <Typography variant="caption" color="text.secondary">
                                {job.total_frames.toLocaleString()} frames indexed
                            </Typography>
                        )}
                    </Box>
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'flex-end' }}>
                        <Button
                            size="small"
                            startIcon={<GpsFixed />}
                            onClick={() => onGround(job.file_path)}
                            variant="outlined"
                            color="primary"
                        >
                            Ground
                        </Button>
                        <Button
                            size="small"
                            startIcon={<SummaryIcon />}
                            onClick={() => onShowSummary(job.file_path)}
                            variant="outlined"
                            color="secondary"
                        >
                            Recap
                        </Button>
                    </Box>
                </Box>
            )}

            {/* Failed State - Show Error */}
            {job.status === 'failed' && job.error && (
                <Alert severity="error" sx={{ mt: 1 }}>
                    <Typography variant="body2" sx={{ wordBreak: 'break-word' }}>
                        {job.error}
                    </Typography>
                </Alert>
            )}
        </Paper>
    );
}
