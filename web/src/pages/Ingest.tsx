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
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Checkbox,
  FormControlLabel,
} from '@mui/material';
import {
  CloudUpload,
  Stop,
  FolderOpen,
  Add,
  Delete,
  Pause,
  PlayArrow,
  GpsFixed,
  Description as SummaryIcon,
} from '@mui/icons-material';

import {
  ingestMedia,
  getJobs,
  cancelJob,
  pauseJob,
  resumeJob,
  deleteJob,
  triggerGrounding,
  getVideoSummary,
} from '../api/client';
import { API_BASE } from '../api/client';
import type { Job, StageStats } from '../api/client';

type JobStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'paused';

// Pipeline stage labels for human-readable display
const STAGE_LABELS: Record<string, { label: string; icon: string; order: number }> = {
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
const ORDERED_STAGES = Object.entries(STAGE_LABELS)
  .sort(([, a], [, b]) => a.order - b.order)
  .map(([key]) => key);

// Format seconds to MM:SS or HH:MM:SS
function formatTime(seconds: number): string {
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
function formatDuration(seconds: number | null | undefined): string {
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
function formatETA(etaSeconds: number | null | undefined): string | null {
  if (!etaSeconds || etaSeconds <= 0) return null;
  if (etaSeconds < 60) return `~${Math.ceil(etaSeconds)}s left`;
  if (etaSeconds < 3600) return `~${Math.ceil(etaSeconds / 60)}m left`;
  return `~${Math.round(etaSeconds / 3600 * 10) / 10}h left`;
}

function JobCard({
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
  const hasFrameData = job.processed_frames !== undefined && job.total_frames;
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
              // Use stage_stats if available, otherwise fallback to simple order logic
              const stats = job.stage_stats?.[stage];

              let isDone = idx < currentStageIndex;
              let isCurrent = stage === currentStage;
              let isSkipped = false;
              let duration = null;

              if (stats) {
                if (stats.status === 'completed') isDone = true;
                if (stats.status === 'running') isCurrent = true;
                if (stats.status === 'skipped') isSkipped = true;
                if (stats.duration) duration = stats.duration;
              } else if (job.pipeline_stage === 'complete') {
                isDone = true;
                isCurrent = false;
              }

              // Format duration if available
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
          {/* Completion Stats */}
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
          {/* Action Buttons */}
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

export default function IngestPage() {
  const [paths, setPaths] = useState<string[]>(['']);
  const [mediaType, setMediaType] = useState('unknown');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [enableChunking, setEnableChunking] = useState(true); // Default ON
  const [pendingJobs, setPendingJobs] = useState<string[]>([]);

  interface SummaryData {
    summary?: string;
    entities?: string[];
    storyline?: string;
  }

  const [summaryDialog, setSummaryDialog] = useState<{ open: boolean; path: string; data: SummaryData | null }>({
    open: false,
    path: '',
    data: null,
  });
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
    if (parts.some(isNaN)) return undefined; // Invalid input = use full video
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2]; // h:m:s
    if (parts.length === 2) return parts[0] * 60 + parts[1]; // m:s
    if (parts.length === 1) return parts[0]; // seconds only
    return undefined;
  };

  const ingestMutation = useMutation({
    mutationFn: (data: {
      path: string;
      hint: string;
      start?: number;
      end?: number;
      enableChunking?: boolean;
    }) => ingestMedia(data.path, data.hint, data.start, data.end, data.enableChunking),
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
    mutationFn: deleteJob,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['jobs'] }),
  });

  const groundMutation = useMutation({
    mutationFn: (path: string) => triggerGrounding(path),
    onSuccess: () => {
      alert("Grounding pipeline triggered in background.");
    }
  });

  const handleShowSummary = async (path: string) => {
    try {
      const data = await getVideoSummary(path);
      setSummaryDialog({ open: true, path, data });
    } catch {
      alert("No summary found for this video yet.");
    }
  };

  // SSE connection
  useEffect(() => {
    const connect = () => {
      const es = new EventSource(`${API_BASE}/events`);
      eventSourceRef.current = es;
      es.onmessage = () => queryClient.invalidateQueries({ queryKey: ['jobs'] });
      es.onerror = () => {
        es.close();
        setTimeout(() => {
          connect();
        }, 5000);
      };
    };
    connect();
    return () => {
      eventSourceRef.current?.close();
      eventSourceRef.current = null;
    };
  }, [queryClient]);

  const handleSubmit = () => {
    const validPaths = paths.filter(p => p.trim());
    if (validPaths.length === 0) return;

    const start = parseTime(startTime);
    const end = parseTime(endTime);
    validPaths.forEach(path => {
      setPendingJobs(prev => [...prev, path.trim()]);
      ingestMutation.mutate({ path: path.trim(), hint: mediaType, start, end, enableChunking });
    });
    setPaths(['']);
    setStartTime('');
    setEndTime('');
  };

  // Helper to normalize path for comparison (handles Windows/Linux differences)
  const normalizePath = (p: string) => p.replace(/\\/g, '/').toLowerCase();
  const getBasename = (p: string) =>
    p.split(/[/\\]/).pop()?.toLowerCase() || '';

  // Clear pending jobs when they appear in real jobs
  useEffect(() => {
    if (!jobs.data?.jobs?.length) return;

    const realJobPaths = jobs.data.jobs.map((j: Job) =>
      normalizePath(j.file_path),
    );
    const realJobBasenames = jobs.data.jobs.map((j: Job) =>
      getBasename(j.file_path),
    );

    setPendingJobs(prev =>
      prev.filter(p => {
        const normalizedPending = normalizePath(p);
        const pendingBasename = getBasename(p);
        // Clear if exact path matches OR if basename matches (fallback)
        return (
          !realJobPaths.includes(normalizedPending) &&
          !realJobBasenames.includes(pendingBasename)
        );
      }),
    );
  }, [jobs.data]);

  const addPath = () => setPaths([...paths, '']);
  const removePath = (idx: number) =>
    setPaths(paths.filter((_, i) => i !== idx));
  const updatePath = (idx: number, value: string) => {
    const newPaths = [...paths];
    newPaths[idx] = value;
    setPaths(newPaths);
  };

  const activeJobs =
    jobs.data?.jobs?.filter((j: Job) =>
      ['running', 'paused', 'pending'].includes(j.status),
    ) || [];

  const historyJobs =
    jobs.data?.jobs?.filter(
      (j: Job) => !['running', 'paused', 'pending'].includes(j.status),
    ) || [];

  return (
    <Box>
      <Typography variant="h5" fontWeight={700} gutterBottom>
        Ingest Media
      </Typography>
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
              onChange={e => updatePath(idx, e.target.value)}
              slotProps={{
                input: {
                  startAdornment: (
                    <FolderOpen
                      sx={{ mr: 1, color: 'action.active' }}
                      fontSize="small"
                    />
                  ),
                },
              }}
            />
            {paths.length > 1 && (
              <IconButton
                size="small"
                onClick={() => removePath(idx)}
                color="error"
              >
                <Delete fontSize="small" />
              </IconButton>
            )}
          </Box>
        ))}

        <Box sx={{ display: 'flex', gap: 1, mt: 1.5, flexWrap: 'wrap' }}>
          <Button
            size="small"
            startIcon={<Add />}
            onClick={addPath}
            variant="outlined"
          >
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
            onChange={e => setStartTime(e.target.value)}
          />
          <TextField
            size="small"
            sx={{ width: 100 }}
            placeholder="(full)"
            label="End"
            value={endTime}
            onChange={e => setEndTime(e.target.value)}
          />
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Type</InputLabel>
            <Select
              value={mediaType}
              label="Type"
              onChange={e => setMediaType(e.target.value)}
            >
              <MenuItem value="unknown">Auto</MenuItem>
              <MenuItem value="movie">Movie</MenuItem>
              <MenuItem value="tv">TV</MenuItem>
              <MenuItem value="personal">Personal</MenuItem>
            </Select>
          </FormControl>
          <Tooltip title="Split long media into smaller segments to prevent Out-Of-Memory errors">
            <FormControlLabel
              control={
                <Checkbox
                  checked={enableChunking}
                  onChange={e => setEnableChunking(e.target.checked)}
                  size="small"
                />
              }
              label={<Typography variant="body2">Memory Chunking</Typography>}
              sx={{ mr: 2 }}
            />
          </Tooltip>
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
        {deleteMutation.isError && (
          <Alert severity="error" sx={{ mt: 1.5 }}>
            Failed to delete job: {(deleteMutation.error as Error)?.message}
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
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 1,
                }}
              >
                <Typography
                  variant="body2"
                  fontWeight={600}
                  noWrap
                  sx={{ maxWidth: '60%' }}
                >
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
              onCancel={id => cancelMutation.mutate(id)}
              onPause={id => pauseMutation.mutate(id)}
              onResume={id => resumeMutation.mutate(id)}
              onGround={path => groundMutation.mutate(path)}
              onShowSummary={handleShowSummary}
            />
          ))}
        </Box>
      )}

      {/* History */}
      {historyJobs.length > 0 && (
        <Box>
          <Typography variant="subtitle2" fontWeight={600} gutterBottom>
            History
          </Typography>
          <Paper sx={{ borderRadius: 2 }}>
            <List dense disablePadding>
              {historyJobs.map((job: Job, idx: number) => (
                <ListItem
                  key={job.job_id}
                  divider={idx < historyJobs.length - 1}
                  secondaryAction={
                    <Tooltip title="Delete Job">
                      <IconButton
                        edge="end"
                        aria-label="delete"
                        onClick={() => {
                          if (window.confirm('Delete this job from history?')) {
                            deleteMutation.mutate(job.job_id);
                          }
                        }}
                      >
                        <Delete fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  }
                >
                  <ListItemText
                    primary={job.file_path.split(/[/\\]/).pop()}
                    secondary={
                      <>
                        {job.completed_at
                          ? new Date(job.completed_at * 1000).toLocaleString()
                          : job.status}
                        {job.status === 'failed' && ` - ${job.error}`}
                      </>
                    }
                    sx={{ mr: 4 }}
                  />
                  <Chip
                    size="small"
                    label={job.status}
                    color={
                      job.status === 'completed'
                        ? 'success'
                        : job.status === 'failed'
                          ? 'error'
                          : 'default'
                    }
                    sx={{ mr: 2 }}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
        </Box>
      )}

      {/* Empty State */}
      {!jobs.isLoading &&
        jobs.data?.jobs?.length === 0 &&
        pendingJobs.length === 0 && (
          <Paper
            sx={{
              p: 4,
              textAlign: 'center',
              borderRadius: 2,
              bgcolor: 'action.hover',
            }}
          >
            <CloudUpload sx={{ fontSize: 40, color: 'text.secondary', mb: 1 }} />
            <Typography color="text.secondary">
              No jobs yet. Add a file path above.
            </Typography>
          </Paper>
        )}
      <Dialog
        open={summaryDialog.open}
        onClose={() => setSummaryDialog({ ...summaryDialog, open: false })}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ fontWeight: 700, borderBottom: 1, borderColor: 'divider' }}>
          AI Recap: {summaryDialog.path.split(/[/\\]/).pop()}
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          {summaryDialog.data && (
            <Box>
              <Typography variant="h6" gutterBottom color="primary" sx={{ fontWeight: 600 }}>
                High-Level Summary
              </Typography>
              <Typography variant="body1" sx={{ mb: 3, fontStyle: 'italic', lineHeight: 1.6 }}>
                "{summaryDialog.data.summary}"
              </Typography>

              {summaryDialog.data.entities && summaryDialog.data.entities.length > 0 && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom fontWeight={700}>
                    Key Entities Detected
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {summaryDialog.data.entities.map((e: string, i: number) => (
                      <Chip key={i} label={e} size="small" variant="outlined" />
                    ))}
                  </Box>
                </Box>
              )}

              {summaryDialog.data.storyline && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom fontWeight={700}>
                    Full Analysis
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                    {summaryDialog.data.storyline}
                  </Typography>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
          <Button onClick={() => setSummaryDialog({ ...summaryDialog, open: false })} variant="contained">
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
