import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Box,
    Typography,
    Paper,
    Grid,
    TextField,
    Button,
    Chip,
    Avatar,
    Card,
    CardContent,
    CircularProgress,
    List,
    ListItem,
    ListItemAvatar,
    ListItemText,
    Divider,
} from '@mui/material';
import {
    Hub,
    Person,
    Timeline,
    Search,
    ArrowForward,
} from '@mui/icons-material';

import {
    getSocialGraph,
    getGraphStats,
    type CoOccurrence,
} from '../api/client';

// Simple visualization component using force-directed layout conceptualization
// For MVP, we use lists and cards. Full D3 implementation would be too heavy for this step.

function SocialNode({ data }: { data: CoOccurrence }) {
    return (
        <Card variant="outlined" sx={{ mb: 1, display: 'flex', alignItems: 'center', p: 1 }}>
            <Avatar sx={{ bgcolor: 'secondary.main', mr: 2 }}>
                <Person />
            </Avatar>
            <Box sx={{ flexGrow: 1 }}>
                <Typography variant="subtitle2" fontWeight="bold">
                    {data.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                    {data.count} co-occurrences
                </Typography>
            </Box>
            <Chip
                size="small"
                label={`${data.count}x`}
                color="primary"
                variant={data.count > 5 ? "filled" : "outlined"}
            />
        </Card>
    );
}

export default function GraphPage() {
    const [searchTerm, setSearchTerm] = useState('');
    const [targetName, setTargetName] = useState('Unknown');

    const statsQuery = useQuery({
        queryKey: ['graph', 'stats'],
        queryFn: getGraphStats,
    });

    const socialQuery = useQuery({
        queryKey: ['graph', 'social', targetName],
        queryFn: () => getSocialGraph(targetName),
        enabled: !!targetName && targetName !== 'Unknown',
        retry: false,
    });

    const handleSearch = () => {
        if (searchTerm.trim()) {
            setTargetName(searchTerm.trim());
        }
    };

    return (
        <Box>
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="h4" fontWeight={700} gutterBottom>
                        Knowledge Graph
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Explore relationships and timelines extracted from your media library.
                    </Typography>
                </Box>
                <Hub sx={{ fontSize: 40, color: 'text.secondary', opacity: 0.5 }} />
            </Box>

            {/* Stats Overview */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid size={{ xs: 12, md: 4 }}>
                    <Paper sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="h3" color="primary.main" fontWeight="bold">
                            {statsQuery.data?.stats?.identities || statsQuery.data?.stats?.total_identities || 0}
                        </Typography>
                        <Typography variant="overline" color="text.secondary">
                            Tracked Identities
                        </Typography>
                    </Paper>
                </Grid>
                <Grid size={{ xs: 12, md: 4 }}>
                    <Paper sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="h3" color="secondary.main" fontWeight="bold">
                            {statsQuery.data?.stats?.scenes || statsQuery.data?.stats?.total_scenes || 0}
                        </Typography>
                        <Typography variant="overline" color="text.secondary">
                            Analyzed Scenes
                        </Typography>
                    </Paper>
                </Grid>
                <Grid size={{ xs: 12, md: 4 }}>
                    <Paper sx={{ p: 3, textAlign: 'center' }}>
                        <Typography variant="h3" fontWeight="bold">
                            {statsQuery.data?.stats?.scene_transitions || statsQuery.data?.stats?.total_relationships || 0}
                        </Typography>
                        <Typography variant="overline" color="text.secondary">
                            Social Links
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>

            {/* Social Graph Search */}
            <Paper sx={{ p: 3, mb: 4 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Hub sx={{ mr: 1, color: 'primary.main' }} />
                    <Typography variant="h6">Social Network Explorer</Typography>
                </Box>

                <Box sx={{ display: 'flex', gap: 2, mb: 4 }}>
                    <TextField
                        fullWidth
                        placeholder="Enter a person's name (e.g., 'Mom', 'John')"
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    />
                    <Button
                        variant="contained"
                        startIcon={<Search />}
                        onClick={handleSearch}
                    >
                        Explore
                    </Button>
                </Box>

                {socialQuery.isLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                        <CircularProgress />
                    </Box>
                )}

                {socialQuery.isError && (
                    <Paper sx={{ p: 3, bgcolor: 'error.light', color: 'error.contrastText' }}>
                        <Typography>
                            Could not find graph data for "{targetName}". Try indexing more videos or naming faces first.
                        </Typography>
                    </Paper>
                )}

                {socialQuery.data && (
                    <Grid container spacing={4}>
                        {/* Center Node */}
                        <Grid size={{ xs: 12, md: 4 }}>
                            <Card sx={{ height: '100%', bgcolor: 'primary.dark', color: 'white', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', p: 3 }}>
                                <Avatar sx={{ width: 80, height: 80, bgcolor: 'white', color: 'primary.main', mb: 2, fontSize: 32 }}>
                                    <Person fontSize="inherit" />
                                </Avatar>
                                <Typography variant="h5" fontWeight="bold">
                                    {socialQuery.data.center_person}
                                </Typography>
                                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                    Cluster ID: {socialQuery.data.center_cluster_id}
                                </Typography>
                            </Card>
                        </Grid>

                        <Grid size={{ xs: 12, md: 1 }} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <ArrowForward sx={{ fontSize: 40, color: 'text.disabled', transform: { xs: 'rotate(90deg)', md: 'rotate(0)' } }} />
                        </Grid>

                        {/* Connections */}
                        <Grid size={{ xs: 12, md: 7 }}>
                            <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                                APPEARS OFTEN WITH
                            </Typography>
                            <Box sx={{ maxHeight: 400, overflowY: 'auto' }}>
                                {socialQuery.data.connections.map((conn) => (
                                    <SocialNode key={conn.cluster_id} data={conn} />
                                ))}
                                {socialQuery.data.connections.length === 0 && (
                                    <Typography color="text.disabled" fontStyle="italic">
                                        No strong social connections found yet.
                                    </Typography>
                                )}
                            </Box>
                        </Grid>
                    </Grid>
                )}
            </Paper>
        </Box>
    );
}
