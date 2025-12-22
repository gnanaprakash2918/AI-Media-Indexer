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
} from '@mui/material';
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
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

import { getStats, getJobs, healthCheck } from '../api/client';

interface Stats {
    collections: {
        media_segments: { points_count: number };
        media_frames: { points_count: number };
        faces: { points_count: number };
        voice_segments: { points_count: number };
    };
    jobs: {
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

    const stats = useQuery({
        queryKey: ['stats'],
        queryFn: getStats,
        refetchInterval: 10000,
    });

    const jobs = useQuery({
        queryKey: ['jobs'],
        queryFn: getJobs,
        refetchInterval: 5000,
    });

    const health = useQuery({
        queryKey: ['health'],
        queryFn: healthCheck,
    });

    const data = stats.data as Stats | undefined;
    const activeJobs = jobs.data?.jobs?.filter((j: any) => j.status === 'running') || [];
    const recentJobs = jobs.data?.jobs?.slice(0, 5) || [];

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
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
                        title="Dialogue Segments"
                        value={data?.collections?.media_segments?.points_count ?? 0}
                        icon={<TextSnippet />}
                        color="#3b82f6"
                    />
                </Grid>
                <Grid size={{ xs: 6, md: 3 }}>
                    <StatCard
                        title="Visual Frames"
                        value={data?.collections?.media_frames?.points_count ?? 0}
                        icon={<VideoLibrary />}
                        color="#8b5cf6"
                    />
                </Grid>
                <Grid size={{ xs: 6, md: 3 }}>
                    <StatCard
                        title="Faces Detected"
                        value={data?.collections?.faces?.points_count ?? 0}
                        icon={<Face />}
                        color="#ec4899"
                    />
                </Grid>
                <Grid size={{ xs: 6, md: 3 }}>
                    <StatCard
                        title="Voice Segments"
                        value={data?.collections?.voice_segments?.points_count ?? 0}
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
                                {activeJobs.map((job: any) => (
                                    <ListItem key={job.job_id} sx={{ px: 0 }}>
                                        <ListItemIcon>
                                            <Schedule color="primary" />
                                        </ListItemIcon>
                                        <ListItemText
                                            primary={job.file_path.split(/[/\\]/).pop()}
                                            secondary={`${job.current_stage} - ${Math.round(job.progress)}%`}
                                        />
                                        <Box sx={{ width: 100 }}>
                                            <LinearProgress
                                                variant="determinate"
                                                value={job.progress}
                                                sx={{ height: 6, borderRadius: 3 }}
                                            />
                                        </Box>
                                    </ListItem>
                                ))}
                            </List>
                        ) : (
                            <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
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
                                {recentJobs.map((job: any) => (
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
                                            secondary={job.completed_at
                                                ? new Date(job.completed_at * 1000).toLocaleString()
                                                : job.status}
                                        />
                                    </ListItem>
                                ))}
                            </List>
                        ) : (
                            <Typography color="text.secondary" sx={{ py: 2, textAlign: 'center' }}>
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
                                icon={health.data?.qdrant === 'connected' ? <CheckCircle /> : <ErrorIcon />}
                                label={`Qdrant: ${health.data?.qdrant || 'Unknown'}`}
                                color={health.data?.qdrant === 'connected' ? 'success' : 'error'}
                                variant="outlined"
                            />
                            <Chip
                                icon={health.data?.pipeline === 'ready' ? <CheckCircle /> : <ErrorIcon />}
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
