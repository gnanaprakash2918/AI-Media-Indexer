import React, { useState, useEffect } from 'react';
import {
    Container,
    Typography,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Alert,
    IconButton,
    Tooltip,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField,
    Box,
    CircularProgress,
    Badge,
    Tabs,
    Tab,
    Collapse,
    List,
    ListItemButton,
    ListItemText,
    Divider,
} from '@mui/material';
import { RecordVoiceOver, Delete, Edit, AutoAwesome, Groups, ExpandMore, ExpandLess, MoveUp, MergeType } from '@mui/icons-material';
import {
    getVoiceSegments,
    deleteVoiceSegment,
    renameVoiceSpeaker,
    triggerVoiceClustering,
    getVoiceClusters,
    moveVoiceToCluster,
    createNewVoiceCluster,
    nameVoiceCluster,
    mergeVoiceClusters,
} from '../api/client';

interface VoiceSegment {
    id: string;
    media_path: string;
    start: number;
    end: number;
    speaker_label: string;
    speaker_name?: string;
    audio_path?: string;
    voice_cluster_id?: number;
}

interface VoiceCluster {
    cluster_id: number;
    speaker_name: string | null;
    segment_count: number;
    representative: VoiceSegment | null;
    segments: VoiceSegment[];
}

export const Voices: React.FC = () => {
    const [segments, setSegments] = useState<VoiceSegment[]>([]);
    const [clusters, setClusters] = useState<VoiceCluster[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isClustering, setIsClustering] = useState(false);
    const [tab, setTab] = useState(0);
    const [renameDialogOpen, setRenameDialogOpen] = useState(false);
    const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
    const [selectedClusterId, setSelectedClusterId] = useState<number | null>(null);
    const [isClusterRename, setIsClusterRename] = useState(false);
    const [newName, setNewName] = useState('');
    const [isRenaming, setIsRenaming] = useState(false);
    const [moveDialogOpen, setMoveDialogOpen] = useState(false);
    const [moveSegmentId, setMoveSegmentId] = useState<string | null>(null);
    const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set());
    const [mergeSourceId, setMergeSourceId] = useState<number | null>(null);
    const [isMerging, setIsMerging] = useState(false);

    useEffect(() => { loadData(); }, []);

    const loadData = async () => {
        try {
            const [segData, clusterData] = await Promise.all([getVoiceSegments(undefined, 500), getVoiceClusters()]);
            setSegments(segData.segments || []);
            setClusters(clusterData.clusters || []);
        } catch (err) { console.error(err); setError('Failed to load voice segments'); }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('Delete this voice segment?')) return;
        try {
            await deleteVoiceSegment(id);
            setSegments(segments.filter(s => s.id !== id));
            const clusterData = await getVoiceClusters();
            setClusters(clusterData.clusters || []);
        } catch (err) { console.error(err); setError('Failed to delete segment'); }
    };

    const handleRename = async () => {
        if (!newName.trim()) return;

        setIsRenaming(true);
        try {
            if (isClusterRename && selectedClusterId !== null) {
                await nameVoiceCluster(selectedClusterId, newName.trim());
            } else if (selectedSegmentId) {
                await renameVoiceSpeaker(selectedSegmentId, newName.trim());
            }

            await loadData();
            setRenameDialogOpen(false);
            setSelectedSegmentId(null);
            setSelectedClusterId(null);
            setNewName('');
        } catch (err) { console.error(err); setError('Failed to rename'); }
        finally { setIsRenaming(false); }
    };

    const openRenameDialog = (segmentId: string, currentName?: string) => {
        setSelectedSegmentId(segmentId);
        setSelectedClusterId(null);
        setIsClusterRename(false);
        setNewName(currentName || '');
        setRenameDialogOpen(true);
    };

    const openClusterRenameDialog = (clusterId: number, currentName?: string) => {
        setSelectedClusterId(clusterId);
        setSelectedSegmentId(null);
        setIsClusterRename(true);
        setNewName(currentName || '');
        setRenameDialogOpen(true);
    };

    const handleTriggerClustering = async () => {
        setIsClustering(true);
        try { await triggerVoiceClustering(); await loadData(); }
        catch (err) { console.error(err); setError('Failed to cluster voices'); }
        finally { setIsClustering(false); }
    };

    const handleMoveToCluster = async (clusterId: number) => {
        if (!moveSegmentId) return;
        try { await moveVoiceToCluster(moveSegmentId, clusterId); await loadData(); setMoveDialogOpen(false); setMoveSegmentId(null); }
        catch (err) { console.error(err); setError('Failed to move segment'); }
    };

    const handleCreateNewCluster = async () => {
        if (!moveSegmentId) return;
        try { await createNewVoiceCluster([moveSegmentId]); await loadData(); setMoveDialogOpen(false); setMoveSegmentId(null); }
        catch (err) { console.error(err); setError('Failed to create new cluster'); }
    };

    const handleMergeClusters = async (toClusterId: number) => {
        if (!mergeSourceId || isMerging) return;
        setIsMerging(true);
        try {
            await mergeVoiceClusters(mergeSourceId, toClusterId);
            await loadData();
            setMergeSourceId(null);
        } catch (err) {
            console.error(err);
            setError('Failed to merge clusters');
        } finally {
            setIsMerging(false);
        }
    };

    const toggleClusterExpand = (clusterId: number) => {
        const newExpanded = new Set(expandedClusters);
        if (newExpanded.has(clusterId)) newExpanded.delete(clusterId);
        else newExpanded.add(clusterId);
        setExpandedClusters(newExpanded);
    };

    const openMoveDialog = (segmentId: string) => { setMoveSegmentId(segmentId); setMoveDialogOpen(true); };

    const SegmentRow = ({ seg }: { seg: VoiceSegment }) => (
        <TableRow>
            <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{seg.media_path.split(/[/\\]/).pop()}</TableCell>
            <TableCell>{seg.start.toFixed(2)}s - {seg.end.toFixed(2)}s</TableCell>
            <TableCell><Chip label={seg.speaker_label} size="small" variant="outlined" /></TableCell>
            <TableCell>{seg.speaker_name ? <Chip label={seg.speaker_name} color="success" size="small" /> : <Typography variant="caption" color="text.secondary">Unnamed</Typography>}</TableCell>
            <TableCell>{seg.audio_path ? <audio controls src={`http://localhost:8000${seg.audio_path}`} style={{ height: 28, width: 160 }} /> : <Typography variant="caption" color="text.secondary">No audio</Typography>}</TableCell>
            <TableCell>
                <Tooltip title="Rename"><IconButton onClick={() => openRenameDialog(seg.id, seg.speaker_name)} size="small" color="primary"><Edit fontSize="small" /></IconButton></Tooltip>
                <Tooltip title="Move"><IconButton onClick={() => openMoveDialog(seg.id)} size="small"><MoveUp fontSize="small" /></IconButton></Tooltip>
                <Tooltip title="Delete"><IconButton onClick={() => handleDelete(seg.id)} size="small" color="error"><Delete fontSize="small" /></IconButton></Tooltip>
            </TableCell>
        </TableRow>
    );

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                <RecordVoiceOver fontSize="large" color="primary" />
                <Typography variant="h4">Voice Intelligence</Typography>
                <Tooltip title="Run DBSCAN clustering">
                    <Button variant="outlined" startIcon={isClustering ? <CircularProgress size={16} /> : <AutoAwesome />} onClick={handleTriggerClustering} disabled={isClustering} size="small">
                        {isClustering ? 'Clustering...' : 'Cluster Voices'}
                    </Button>
                </Tooltip>
            </Box>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>View all voice segments or grouped clusters.</Typography>

            {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>{error}</Alert>}

            <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
                <Tab label={`All Segments (${segments.length})`} />
                <Tab label={`Clusters (${clusters.length})`} icon={<Groups fontSize="small" />} iconPosition="start" />
            </Tabs>

            {tab === 0 ? (
                segments.length === 0 ? (
                    <Alert severity="info">No voice segments found. Ingest media with "Voice Analysis" enabled.</Alert>
                ) : (
                    <TableContainer component={Paper}>
                        <Table size="small">
                            <TableHead>
                                <TableRow>
                                    <TableCell>Media</TableCell>
                                    <TableCell>Time Range</TableCell>
                                    <TableCell>Speaker ID</TableCell>
                                    <TableCell>Name</TableCell>
                                    <TableCell>Audio</TableCell>
                                    <TableCell>Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>{segments.map((seg) => <SegmentRow key={seg.id} seg={seg} />)}</TableBody>
                        </Table>
                    </TableContainer>
                )
            ) : (
                clusters.length === 0 ? (
                    <Paper sx={{ p: 4, textAlign: 'center' }}>
                        <Groups sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                        <Typography variant="h6" color="text.secondary">No voice clusters found</Typography>
                        <Button variant="contained" startIcon={<AutoAwesome />} onClick={handleTriggerClustering} disabled={isClustering} sx={{ mt: 2 }}>Cluster Voices</Button>
                    </Paper>
                ) : (
                    <Box>
                        {clusters.map((cluster) => (
                            <Paper key={cluster.cluster_id} sx={{ mb: 2, overflow: 'hidden' }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', p: 2, cursor: 'pointer', '&:hover': { bgcolor: 'action.hover' } }} onClick={() => toggleClusterExpand(cluster.cluster_id)}>
                                    <Badge badgeContent={cluster.segment_count} color="primary" sx={{ mr: 2 }}><RecordVoiceOver /></Badge>
                                    <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Box>
                                            {cluster.speaker_name ? <Chip label={cluster.speaker_name} color="success" size="small" /> : <Typography variant="body1" fontWeight={600}>Voice Cluster #{cluster.cluster_id}</Typography>}
                                            <Typography variant="caption" color="text.secondary" display="block">{cluster.segment_count} segment{cluster.segment_count !== 1 ? 's' : ''}</Typography>
                                        </Box>
                                        <Button
                                            size="small"
                                            startIcon={<Edit />}
                                            onClick={(e) => { e.stopPropagation(); openClusterRenameDialog(cluster.cluster_id, cluster.speaker_name || ''); }}
                                            sx={{ mr: 1 }}
                                        >
                                            Rename
                                        </Button>
                                        <Button
                                            size="small"
                                            variant="outlined"
                                            color="warning"
                                            startIcon={<MergeType />}
                                            onClick={(e) => { e.stopPropagation(); setMergeSourceId(cluster.cluster_id); }}
                                        >
                                            Merge
                                        </Button>
                                    </Box>
                                    {cluster.representative?.audio_path && <audio controls src={`http://localhost:8000${cluster.representative.audio_path}`} style={{ height: 28, width: 140, marginRight: 16 }} onClick={(e) => e.stopPropagation()} />}
                                    {expandedClusters.has(cluster.cluster_id) ? <ExpandLess /> : <ExpandMore />}
                                </Box>
                                <Collapse in={expandedClusters.has(cluster.cluster_id)}>
                                    <Divider />
                                    <TableContainer>
                                        <Table size="small">
                                            <TableHead><TableRow><TableCell>Media</TableCell><TableCell>Time Range</TableCell><TableCell>Speaker ID</TableCell><TableCell>Name</TableCell><TableCell>Audio</TableCell><TableCell>Actions</TableCell></TableRow></TableHead>
                                            <TableBody>{cluster.segments.map((seg) => <SegmentRow key={seg.id} seg={seg} />)}</TableBody>
                                        </Table>
                                    </TableContainer>
                                </Collapse>
                            </Paper>
                        ))}
                    </Box>
                )
            )}

            <Dialog open={renameDialogOpen} onClose={() => setRenameDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>{isClusterRename ? 'Rename Voice Cluster' : 'Rename Speaker'}</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {isClusterRename
                            ? 'Enter a name for this entire voice cluster. This will update all segments in this cluster.'
                            : 'Enter a name for this speaker segment.'}
                    </Typography>
                    <TextField fullWidth autoFocus label="Speaker Name" value={newName} onChange={(e) => setNewName(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleRename()} />
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setRenameDialogOpen(false)}>Cancel</Button>
                    <Button variant="contained" onClick={handleRename} disabled={!newName.trim() || isRenaming}>{isRenaming ? 'Saving...' : 'Save'}</Button>
                </DialogActions>
            </Dialog>

            <Dialog open={moveDialogOpen} onClose={() => setMoveDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle>Move to Cluster</DialogTitle>
                <DialogContent>
                    <List>
                        <ListItemButton onClick={handleCreateNewCluster}><ListItemText primary="+ Create New Cluster" secondary="Create new speaker identity" /></ListItemButton>
                        <Divider />
                        {clusters.map((c) => (
                            <ListItemButton key={c.cluster_id} onClick={() => handleMoveToCluster(c.cluster_id)}>
                                <ListItemText primary={c.speaker_name || `Cluster #${c.cluster_id}`} secondary={`${c.segment_count} segments`} />
                            </ListItemButton>
                        ))}
                    </List>
                </DialogContent>
                <DialogActions><Button onClick={() => setMoveDialogOpen(false)}>Cancel</Button></DialogActions>
            </Dialog>

            {/* Merge Clusters Dialog */}
            <Dialog open={mergeSourceId !== null} onClose={() => setMergeSourceId(null)} maxWidth="sm" fullWidth>
                <DialogTitle>Merge Voice Cluster</DialogTitle>
                <DialogContent>
                    <Typography variant="body2" sx={{ mb: 2 }}>
                        Select a target cluster to merge into. Named clusters are shown first.
                    </Typography>
                    {isMerging ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : (
                        <List>
                            {clusters
                                .filter(c => c.cluster_id !== mergeSourceId)
                                .sort((a, b) => {
                                    // Named clusters first, then by segment count
                                    if (a.speaker_name && !b.speaker_name) return -1;
                                    if (!a.speaker_name && b.speaker_name) return 1;
                                    return (b.segment_count || 0) - (a.segment_count || 0);
                                })
                                .map((c) => (
                                    <ListItemButton
                                        key={c.cluster_id}
                                        onClick={() => handleMergeClusters(c.cluster_id)}
                                        sx={{
                                            border: c.speaker_name ? '2px solid' : 'none',
                                            borderColor: c.speaker_name ? 'primary.main' : 'transparent',
                                            borderRadius: 1,
                                            mb: 0.5,
                                            bgcolor: c.speaker_name ? 'action.selected' : 'transparent',
                                        }}
                                    >
                                        <ListItemText
                                            primary={
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    {c.speaker_name || `Voice Cluster #${c.cluster_id}`}
                                                    {c.speaker_name && <Chip size="small" label="Named" color="primary" variant="outlined" />}
                                                </Box>
                                            }
                                            secondary={`${c.segment_count} segments`}
                                        />
                                    </ListItemButton>
                                ))}
                        </List>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setMergeSourceId(null)}>Cancel</Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
};

export default Voices;