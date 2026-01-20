import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Button,
  IconButton,
} from '@mui/material';
import { useQueryClient } from '@tanstack/react-query';
import {
  VideoLibrary,
  Face,
  RecordVoiceOver,
  TextSnippet,
  TrendingUp,
  PlayArrow,
  CheckCircle,
  Error as ErrorIcon,
  Schedule,
  Pause,
  Stop,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import {
  getStats,
  getJobs,
  healthCheck,
  pauseJob,
  resumeJob,
  cancelJob,
  type Job,
} from '../api/client';

interface Stats {
  frames: number;
  faces: number;
  voice_segments: number;
  videos: number;
  // Legacy nested format (fallback)
  collections?: {
    media_segments?: { points_count: number };
    media_frames?: { points_count: number };
    faces?: { points_count: number };
    voice_segments?: { points_count: number };
  };
  jobs?: {
    active: number;
    completed: number;
    failed: number;
    total: number;
  };
}

function StatCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <Paper
      sx={{
        p: 3,
        borderRadius: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 2,
      }}
    >
      <Box
        sx={{
          width: 56,
          height: 56,
          borderRadius: 2,
          bgcolor: `${color}20`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: color,
        }}
      >
        {icon}
      </Box>
      <Box>
        <Typography variant="h4" fontWeight={700}>
          {value}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {title}
        </Typography>
      </Box>
    </Paper>
  );
}

export default function DashboardPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  const stats = useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
    refetchInterval: 10000,
  });

  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: 2000, // Faster updates for progress
  });

  const health = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
  });

  const handlePause = async (jobId: string) => {
    await pauseJob(jobId);
    queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };

  const handleResume = async (jobId: string) => {
    await resumeJob(jobId);
    queryClient.invalidateQueries({ queryKey: ['jobs'] });
  };

  const handleCancel = async (jobId: string) => {
    if (confirm('Stop this job? This cannot be undone.')) {
      await cancelJob(jobId);
      queryClient.invalidateQueries({ queryKey: ['jobs'] });
    }
  };

  const data = stats.data as Stats | undefined;
  const activeJobs =
    jobs.data?.jobs?.filter((j: Job) =>
      ['running', 'paused'].includes(j.status),
    ) || [];
  const recentJobs = jobs.data?.jobs?.slice(0, 5) || [];

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 4,
        }}
      >
        <Box>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Dashboard
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Overview of your media library and processing status.
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<PlayArrow />}
          onClick={() => navigate('/ingest')}
        >
          Ingest Media
        </Button>
      </Box>

      {/* Stats Grid */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid size={{ xs: 6, md: 3 }}>
          <StatCard
            title="Indexed Videos"
            value={data?.videos ?? 0}
            icon={<VideoLibrary />}
            color="#3b82f6"
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <StatCard
            title="Visual Frames"
            value={data?.frames ?? 0}
            icon={<TextSnippet />}
            color="#8b5cf6"
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <StatCard
            title="Faces Detected"
            value={data?.faces ?? 0}
            icon={<Face />}
            color="#ec4899"
          />
        </Grid>
        <Grid size={{ xs: 6, md: 3 }}>
          <StatCard
            title="Voice Segments"
            value={data?.voice_segments ?? 0}
            icon={<RecordVoiceOver />}
            color="#10b981"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Active Jobs */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <TrendingUp color="primary" />
              <Typography variant="h6" fontWeight={600}>
                Active Processing
              </Typography>
              {activeJobs.length > 0 && (
                <Chip label={activeJobs.length} size="small" color="primary" />
              )}
            </Box>
            <Divider sx={{ mb: 2 }} />

            {activeJobs.length > 0 ? (
              <List disablePadding>
                {activeJobs.map((job: Job) => (
                  <ListItem key={job.job_id} sx={{ px: 0 }}>
                    <ListItemIcon>
                      {job.status === 'paused' ? (
                        <Pause color="warning" />
                      ) : (
                        <Schedule color="primary" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={job.file_path.split(/[/\\]/).pop()}
                      secondary={
                        <Box>
                          <Typography
                            variant="body2"
                            component="span"
                            display="block"
                          >
                            {job.status === 'paused' ? 'PAUSED: ' : ''}
                            {job.message || job.current_stage}
                          </Typography>
                          {job.total_frames && job.total_frames > 0 && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              Frame: {job.processed_frames}/{job.total_frames} |
                              Time: {job.timestamp?.toFixed(1)}s /{' '}
                              {job.duration?.toFixed(1)}s
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {job.status === 'paused' ? (
                        <IconButton
                          size="small"
                          onClick={() => handleResume(job.job_id)}
                          title="Resume"
                        >
                          <PlayArrow fontSize="small" color="primary" />
                        </IconButton>
                      ) : (
                        <IconButton
                          size="small"
                          onClick={() => handlePause(job.job_id)}
                          title="Pause"
                        >
                          <Pause fontSize="small" color="warning" />
                        </IconButton>
                      )}
                      <IconButton
                        size="small"
                        onClick={() => handleCancel(job.job_id)}
                        title="Stop"
                      >
                        <Stop fontSize="small" color="error" />
                      </IconButton>
                    </Box>
                    <Box sx={{ width: 100, ml: 2 }}>
                      <LinearProgress
                        variant="determinate"
                        value={job.progress}
                        color={job.status === 'paused' ? 'warning' : 'primary'}
                        sx={{ height: 6, borderRadius: 3 }}
                      />
                    </Box>
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography
                color="text.secondary"
                sx={{ py: 2, textAlign: 'center' }}
              >
                No active jobs
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Recent Jobs */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              Recent Jobs
            </Typography>
            <Divider sx={{ mb: 2 }} />

            {recentJobs.length > 0 ? (
              <List disablePadding>
                {recentJobs.map((job: Job) => (
                  <ListItem key={job.job_id} sx={{ px: 0 }}>
                    <ListItemIcon>
                      {job.status === 'completed' ? (
                        <CheckCircle color="success" />
                      ) : job.status === 'failed' ? (
                        <ErrorIcon color="error" />
                      ) : (
                        <Schedule color="action" />
                      )}
                    </ListItemIcon>
                    <ListItemText
                      primary={job.file_path.split(/[/\\]/).pop()}
                      secondary={
                        job.completed_at
                          ? new Date(job.completed_at * 1000).toLocaleString()
                          : job.status
                      }
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography
                color="text.secondary"
                sx={{ py: 2, textAlign: 'center' }}
              >
                No jobs yet
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* System Status */}
        <Grid size={{ xs: 12 }}>
          <Paper sx={{ p: 3, borderRadius: 3 }}>
            <Typography variant="h6" fontWeight={600} gutterBottom>
              System Status
            </Typography>
            <Divider sx={{ mb: 2 }} />
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip
                icon={<CheckCircle />}
                label={`Device: ${health.data?.device?.toUpperCase() || 'Unknown'}`}
                color="primary"
                variant="outlined"
              />
              <Chip
                icon={
                  health.data?.qdrant === 'connected' ? (
                    <CheckCircle />
                  ) : (
                    <ErrorIcon />
                  )
                }
                label={`Qdrant: ${health.data?.qdrant || 'Unknown'}`}
                color={
                  health.data?.qdrant === 'connected' ? 'success' : 'error'
                }
                variant="outlined"
              />
              <Chip
                icon={
                  health.data?.pipeline === 'ready' ? (
                    <CheckCircle />
                  ) : (
                    <ErrorIcon />
                  )
                }
                label={`Pipeline: ${health.data?.pipeline || 'Unknown'}`}
                color={health.data?.pipeline === 'ready' ? 'success' : 'error'}
                variant="outlined"
              />
              <Chip
                label={`Jobs: ${data?.jobs?.completed ?? 0} completed, ${data?.jobs?.failed ?? 0} failed`}
                variant="outlined"
              />
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
