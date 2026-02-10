import { mergeFaceClusters } from '../api/client';
import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Chip,
  Avatar,
  Skeleton,
  Alert,
  Tabs,
  Tab,
  InputAdornment,
  IconButton,
  Badge,
  CircularProgress,
  Tooltip,
  Collapse,
  List,
  ListItemButton,
  ListItemAvatar,
  ListItemText,
  Divider,
  Card,
  CardContent,
  Autocomplete,
} from '@mui/material';
import {
  Face,
  Search,
  Check,
  Edit,
  Delete,
  ZoomIn,
  AutoAwesome,
  Groups,
  ExpandMore,
  ExpandLess,
  MoveUp,
  Star,
  StarBorder,
  Lightbulb,
  Close,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import {
  getUnresolvedFaces,
  getNamedFaces,
  getFaceClusters,
  triggerFaceClustering,
  nameFaceCluster,
  nameSingleFace,
  deleteFace,
  moveFaceToCluster,
  createNewFaceCluster,
  setFaceMain,
  getAllNames,
  apiClient,
  deleteFaceCluster,
} from '../api/client';
import type { FaceData, FaceClusterData, IdentitySuggestion } from '../types';

function FaceCard({
  face,
  onLabel,
  onDelete,
  onZoom,
  onMove,
  compact = false,
}: {
  face: FaceData;
  onLabel: (faceId: string) => void;
  onDelete: (faceId: string) => void;
  onZoom: (face: FaceData) => void;
  onMove?: (faceId: string) => void;
  compact?: boolean;
}) {
  const [imageError, setImageError] = useState(false);
  const thumbUrl = face.thumbnail_path
    ? `http://localhost:8000${face.thumbnail_path}`
    : null;
  const hasValidImage = thumbUrl && !imageError;
  const size = compact ? 64 : 96;

  return (
    <Paper
      component={motion.div}
      whileHover={{ scale: 1.02 }}
      sx={{
        p: compact ? 1 : 2,
        borderRadius: 2,
        cursor: 'pointer',
        textAlign: 'center',
        position: 'relative',
        minWidth: compact ? 100 : 140,
      }}
      onClick={() => onLabel(face.id)}
    >
      <IconButton
        size="small"
        sx={{
          position: 'absolute',
          top: 2,
          right: 2,
          opacity: 0.6,
          '&:hover': { opacity: 1, color: 'error.main' },
        }}
        onClick={e => {
          e.stopPropagation();
          onDelete(face.id);
        }}
      >
        <Delete fontSize="small" />
      </IconButton>

      {hasValidImage && (
        <IconButton
          size="small"
          sx={{
            position: 'absolute',
            top: 2,
            left: 2,
            opacity: 0.6,
            '&:hover': { opacity: 1 },
          }}
          onClick={e => {
            e.stopPropagation();
            onZoom(face);
          }}
        >
          <ZoomIn fontSize="small" />
        </IconButton>
      )}

      {onMove && (
        <IconButton
          size="small"
          sx={{
            position: 'absolute',
            bottom: 2,
            right: 2,
            opacity: 0.6,
            '&:hover': { opacity: 1, color: 'primary.main' },
          }}
          onClick={e => {
            e.stopPropagation();
            onMove(face.id);
          }}
        >
          <MoveUp fontSize="small" />
        </IconButton>
      )}

      <Avatar
        src={hasValidImage ? thumbUrl : undefined}
        onError={() => setImageError(true)}
        sx={{
          width: size,
          height: size,
          mx: 'auto',
          mb: 1,
          border: '2px solid',
          borderColor: face.name ? 'success.main' : 'grey.400',
        }}
      >
        {!hasValidImage && <Face />}
      </Avatar>

      {face.name ? (
        <Chip label={face.name} color="success" size="small" />
      ) : (
        <Typography variant="caption" color="text.secondary">
          {face.timestamp !== undefined
            ? `${face.timestamp.toFixed(1)}s`
            : 'Unnamed'}
        </Typography>
      )}
    </Paper>
  );
}

function ClusterCard({
  cluster,
  onLabelCluster,
  onLabelFace,
  onDeleteFace,
  onZoom,
  onMoveFace,
  onMerge,
  onSetMain,
  onDeleteCluster,
  onIdentify,
}: {
  cluster: FaceClusterData;
  onLabelCluster: (clusterId: number) => void;
  onLabelFace: (faceId: string) => void;
  onDeleteFace: (faceId: string) => void;
  onZoom: (face: FaceData) => void;
  onMoveFace: (faceId: string) => void;
  onMerge?: () => void;
  onSetMain?: (clusterId: number, isMain: boolean) => void;
  onDeleteCluster?: (clusterId: number) => void;
  onIdentify?: (clusterId: number) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const representative = cluster.representative;
  const thumbUrl = representative?.thumbnail_path
    ? `http://localhost:8000${representative.thumbnail_path}`
    : null;
  const hasValidImage = thumbUrl && !imageError;

  return (
    <Paper
      sx={{
        mb: 2,
        overflow: 'hidden',
        border: cluster.is_main ? '2px solid' : 'none',
        borderColor: 'warning.main',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          p: 2,
          cursor: 'pointer',
          '&:hover': { bgcolor: 'action.hover' },
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <Avatar
          src={hasValidImage ? thumbUrl : undefined}
          onError={() => setImageError(true)}
          sx={{ width: 56, height: 56, mr: 2 }}
        >
          <Face />
        </Avatar>
        <Box sx={{ flex: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {cluster.is_main && (
              <Star sx={{ color: 'warning.main', fontSize: 18 }} />
            )}
            {cluster.name ? (
              <Chip label={cluster.name} color="success" size="small" />
            ) : (
              <Typography variant="body1" fontWeight={600}>
                Cluster #{cluster.cluster_id}
              </Typography>
            )}
          </Box>
          <Typography variant="caption" color="text.secondary" display="block">
            {cluster.face_count} occurrence{cluster.face_count !== 1 ? 's' : ''}
            {cluster.is_main ? ' | Main Character' : ''}
          </Typography>
        </Box>
        <Badge badgeContent={cluster.face_count} color="primary" sx={{ mr: 2 }}>
          <Groups />
        </Badge>

        {onSetMain && (
          <Tooltip
            title={cluster.is_main ? 'Remove Main' : 'Mark as Main Character'}
          >
            <IconButton
              size="small"
              onClick={e => {
                e.stopPropagation();
                onSetMain(cluster.cluster_id, !cluster.is_main);
              }}
              sx={{ mr: 1, color: cluster.is_main ? 'warning.main' : 'inherit' }}
            >
              {cluster.is_main ? <Star /> : <StarBorder />}
            </IconButton>
          </Tooltip>
        )}

        {onMerge && (
          <Button
            size="small"
            variant="outlined"
            color="warning"
            onClick={e => {
              e.stopPropagation();
              onMerge();
            }}
            sx={{ mr: 1 }}
          >
            <MoveUp fontSize="small" sx={{ mr: 0.5 }} /> Merge
          </Button>
        )}

        <Button
          size="small"
          variant="outlined"
          color="secondary"
          onClick={e => {
            e.stopPropagation();
            onIdentify?.(cluster.cluster_id);
          }}
          sx={{ mr: 1 }}
        >
          <AutoAwesome fontSize="small" sx={{ mr: 0.5 }} /> Auto ID
        </Button>

        <Button
          size="small"
          variant="outlined"
          onClick={e => {
            e.stopPropagation();
            onLabelCluster(cluster.cluster_id);
          }}
          sx={{ mr: 1 }}
        >
          <Edit fontSize="small" sx={{ mr: 0.5 }} /> Label All
        </Button>
        {onDeleteCluster && (
          <Button
            size="small"
            variant="outlined"
            color="error"
            onClick={e => {
              e.stopPropagation();
              onDeleteCluster(cluster.cluster_id);
            }}
            sx={{ mr: 1 }}
          >
            <Delete fontSize="small" sx={{ mr: 0.5 }} /> Delete
          </Button>
        )}
        {expanded ? <ExpandLess /> : <ExpandMore />}
      </Box>
      <Collapse in={expanded}>
        <Divider />
        <Box sx={{ p: 2, bgcolor: 'action.hover' }}>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mb: 1, display: 'block' }}
          >
            All faces in this cluster:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {cluster.faces.map(face => (
              <FaceCard
                key={face.id}
                face={face}
                onLabel={onLabelFace}
                onDelete={onDeleteFace}
                onZoom={onZoom}
                onMove={onMoveFace}
                compact
              />
            ))}
          </Box>
        </Box>
      </Collapse>
    </Paper>
  );
}

function LabelDialog({
  open,
  isCluster,
  onClose,
  onSave,
  isLoading,
  existingNames = [],
}: {
  open: boolean;
  isCluster: boolean;
  onClose: () => void;
  onSave: (name: string) => void;
  isLoading: boolean;
  existingNames?: string[];
}) {
  const [name, setName] = useState('');


  const handleSave = () => {
    if (name.trim()) {
      onSave(name.trim());
      setName('');
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {isCluster ? 'Label Entire Cluster' : 'Label Face'}
      </DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {isCluster
            ? 'All faces in this cluster will be labeled.'
            : 'Label this specific face.'}
        </Typography>
        <Autocomplete
          freeSolo
          options={existingNames}
          inputValue={name}
          onInputChange={(_, newValue) => setName(newValue)}
          renderInput={params => (
            <TextField
              {...params}
              autoFocus
              label="Name"
              placeholder="Enter or select name"
              onKeyDown={e => e.key === 'Enter' && handleSave()}
            />
          )}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={!name.trim() || isLoading}
        >
          {isLoading ? 'Saving...' : 'Save'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function MoveDialog({
  open,
  clusters,
  onClose,
  onMove,
  onNewCluster,
}: {
  open: boolean;
  clusters: FaceClusterData[];
  onClose: () => void;
  onMove: (clusterId: number) => void;
  onNewCluster: () => void;
}) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Move Face to Cluster</DialogTitle>
      <DialogContent>
        <List>
          <ListItemButton onClick={onNewCluster}>
            <ListItemText
              primary="+ Create New Cluster"
              secondary="Create a new separate identity"
            />
          </ListItemButton>
          <Divider />
          {clusters.map(c => (
            <ListItemButton
              key={c.cluster_id}
              onClick={() => onMove(c.cluster_id)}
            >
              <ListItemAvatar>
                <Avatar
                  src={
                    c.representative?.thumbnail_path
                      ? `http://localhost:8000${c.representative.thumbnail_path}`
                      : undefined
                  }
                >
                  <Face />
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={c.name || `Cluster #${c.cluster_id}`}
                secondary={`${c.face_count} faces`}
              />
            </ListItemButton>
          ))}
        </List>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
      </DialogActions>
    </Dialog>
  );
}

export default function FacesPage() {
  const [tab, setTab] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedClusterId, setSelectedClusterId] = useState<number | null>(
    null,
  );
  const [selectedFaceId, setSelectedFaceId] = useState<string | null>(null);
  const [moveFaceId, setMoveFaceId] = useState<string | null>(null);
  const [zoomFace, setZoomFace] = useState<FaceData | null>(null);
  const [isClustering, setIsClustering] = useState(false);
  const [dismissedSuggestions, setDismissedSuggestions] = useState<Set<string>>(
    new Set(),
  );
  const queryClient = useQueryClient();

  const clustersQuery = useQuery({
    queryKey: ['faces', 'clusters'],
    queryFn: () => getFaceClusters(),
  });
  const unresolvedQuery = useQuery({
    queryKey: ['faces', 'unresolved'],
    queryFn: () => getUnresolvedFaces(500),
  });
  const namedQuery = useQuery({
    queryKey: ['faces', 'named'],
    queryFn: () => getNamedFaces(),
  });

  const allNamesQuery = useQuery({
    queryKey: ['identities', 'names'],
    queryFn: () => getAllNames(),
  });

  const suggestionsQuery = useQuery({
    queryKey: ['identity', 'suggestions'],
    queryFn: async () => {
      const res = await apiClient.get<{ suggestions: IdentitySuggestion[] }>(
        '/identity/suggestions',
      );
      return res.data.suggestions || [];
    },
    refetchInterval: 60000,
  });

  const labelClusterMutation = useMutation({
    mutationFn: ({ clusterId, name }: { clusterId: number; name: string }) =>
      nameFaceCluster(clusterId, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setSelectedClusterId(null);
    },
  });

  const labelFaceMutation = useMutation({
    mutationFn: ({ faceId, name }: { faceId: string; name: string }) =>
      nameSingleFace(faceId, name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setSelectedFaceId(null);
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (faceId: string) => deleteFace(faceId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
  });

  const moveMutation = useMutation({
    mutationFn: ({ faceId, clusterId }: { faceId: string; clusterId: number }) =>
      moveFaceToCluster(faceId, clusterId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setMoveFaceId(null);
    },
  });

  const newClusterMutation = useMutation({
    mutationFn: (faceIds: string[]) => createNewFaceCluster(faceIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setMoveFaceId(null);
    },
  });

  const setMainMutation = useMutation({
    mutationFn: ({ clusterId, isMain }: { clusterId: number; isMain: boolean }) =>
      setFaceMain(clusterId, isMain),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
  });

  const handleTriggerClustering = async () => {
    setIsClustering(true);
    try {
      await triggerFaceClustering();
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    } finally {
      setIsClustering(false);
    }
  };

  const handleIdentifyCluster = async (clusterId: number) => {
    try {
      // Optimistic UI or loading state could be added here
      const result = await import('../api/client').then(m => m.identifyFaceCluster(clusterId));
      if (result.status === 'success' && result.match) {
        if (confirm(`Match Found: ${result.match.name} (${Math.round(result.match.confidence * 100)}%)\n\nApply this name to the cluster?`)) {
          labelClusterMutation.mutate({ clusterId, name: result.match.name });
        }
      } else {
        alert('No confident match found for this cluster.');
      }
    } catch (e) {
      console.error('Identification failed:', e);
      alert('Face identification failed. See console for details.');
    }
  };

  const handleDeleteFace = (faceId: string) => {
    if (confirm('Delete this face?')) deleteMutation.mutate(faceId);
  };

  const deleteClusterMutation = useMutation({
    mutationFn: (clusterId: number) => deleteFaceCluster(clusterId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
  });

  const handleDeleteCluster = (clusterId: number) => {
    if (confirm('Delete this entire face cluster and all its faces?')) {
      deleteClusterMutation.mutate(clusterId);
    }
  };

  const clusters: FaceClusterData[] = clustersQuery.data?.clusters || [];
  const allFaces: FaceData[] = [
    ...(unresolvedQuery.data?.faces || []),
    ...(namedQuery.data?.faces || []),
  ];
  const unresolvedFaces: FaceData[] = unresolvedQuery.data?.faces || [];
  const namedFaces: FaceData[] = namedQuery.data?.faces || [];
  const isLoading = clustersQuery.isLoading || unresolvedQuery.isLoading;

  const filterFaces = (faces: FaceData[]) => {
    if (!searchQuery) return faces;
    const q = searchQuery.toLowerCase();
    return faces.filter(
      f =>
        f.name?.toLowerCase().includes(q) ||
        f.media_path?.toLowerCase().includes(q),
    );
  };

  const displayFaces =
    tab === 0 ? allFaces : tab === 1 ? unresolvedFaces : namedFaces;
  const filteredFaces = filterFaces(displayFaces);

  const [mergeSourceId, setMergeSourceId] = useState<number | null>(null);


  // ... inside component ...

  const mergeClustersMutation = useMutation({
    mutationFn: ({ from, to }: { from: number; to: number }) =>
      mergeFaceClusters(from, to),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setMergeSourceId(null);
    },
  });

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
        <Typography variant="h4" fontWeight={700}>
          Face Gallery
        </Typography>
        <Tooltip title="Run DBSCAN clustering">
          <Button
            variant="outlined"
            startIcon={
              isClustering ? <CircularProgress size={16} /> : <AutoAwesome />
            }
            onClick={handleTriggerClustering}
            disabled={isClustering}
            size="small"
          >
            {isClustering ? 'Clustering...' : 'Cluster Faces'}
          </Button>
        </Tooltip>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        View all face occurrences or grouped clusters.
      </Typography>

      {/* Smart Suggestions Panel */}
      {/* Smart Suggestions Panel */}
      {suggestionsQuery.data &&
        suggestionsQuery.data.filter(
          s => !dismissedSuggestions.has(`${s.type}-${s.source}-${s.target}`),
        ).length > 0 && (
          <Paper
            sx={{
              mb: 3,
              bgcolor: 'warning.main',
              color: 'warning.contrastText',
              borderRadius: 2,
              overflow: 'hidden',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 1,
                p: 2,
                cursor: 'pointer',
                '&:hover': { bgcolor: 'warning.dark' },
              }}
              onClick={() => {
                // Toggle logic (using local state which I need to add, or just always show summary)
                // For now, let's just make it compact horizontal scroll
              }}
            >
              <Lightbulb />
              <Typography variant="subtitle1" fontWeight={600}>
                Smart Suggestions
              </Typography>
              <Chip
                label={
                  suggestionsQuery.data.filter(
                    s =>
                      !dismissedSuggestions.has(
                        `${s.type}-${s.source}-${s.target}`,
                      ),
                  ).length
                }
                size="small"
                sx={{ bgcolor: 'white', color: 'warning.main', fontWeight: 'bold' }}
              />
              <Box sx={{ flex: 1 }} />
              <Typography variant="caption" sx={{ mr: 1 }}>
                Automated merge proposals based on timestamps and names
              </Typography>
            </Box>

            <Box
              sx={{
                display: 'flex',
                gap: 2,
                p: 2,
                pt: 0,
                overflowX: 'auto',
                pb: 2,
                '::-webkit-scrollbar': { height: 8 },
                '::-webkit-scrollbar-thumb': { bgcolor: 'rgba(255,255,255,0.3)', borderRadius: 4 }
              }}
            >
              {suggestionsQuery.data
                .filter(
                  s =>
                    !dismissedSuggestions.has(
                      `${s.type}-${s.source}-${s.target}`,
                    ),
                )
                .map((s, i) => (
                  <Card
                    key={i}
                    sx={{
                      minWidth: 260,
                      maxWidth: 260,
                      flexShrink: 0,
                      bgcolor: 'background.paper',
                      boxShadow: 3
                    }}
                  >
                    <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Typography
                          variant="caption"
                          sx={{
                            color: 'primary.main',
                            fontWeight: 'bold',
                            textTransform: 'uppercase',
                            fontSize: '0.7rem'
                          }}
                        >
                          {s.type === 'merge_face_voice'
                            ? 'Face ↔ Voice'
                            : s.type === 'tmdb_match'
                              ? 'TMDB Match'
                              : 'Face ↔ Face'}
                        </Typography>
                        <Chip
                          label={`${Math.round(s.confidence * 100)}%`}
                          size="small"
                          color={s.confidence > 0.8 ? "success" : "warning"}
                          sx={{ height: 20, fontSize: '0.7rem' }}
                        />
                      </Box>

                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                        <Typography variant="body2" fontWeight={700} noWrap title={s.source}>
                          {s.source}
                        </Typography>
                        <MoveUp sx={{ transform: 'rotate(90deg)', color: 'text.secondary', fontSize: 16 }} />
                        <Typography variant="body2" fontWeight={700} noWrap title={s.target}>
                          {s.target}
                        </Typography>
                      </Box>

                      <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1.5, height: 20, overflow: 'hidden' }}>
                        {s.reason}
                      </Typography>

                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Button
                          fullWidth
                          size="small"
                          variant="contained"
                          color="success"
                          onClick={() => {
                            if (
                              s.type === 'merge_face_face' &&
                              s.source_id &&
                              s.target_id
                            ) {
                              mergeClustersMutation.mutate({
                                from: s.source_id,
                                to: s.target_id,
                              });
                            }
                            setDismissedSuggestions(prev => new Set([...prev, `${s.type}-${s.source}-${s.target}`]));
                          }}
                        >
                          Accept
                        </Button>
                        <IconButton
                          size="small"
                          onClick={() =>
                            setDismissedSuggestions(prev => new Set([...prev, `${s.type}-${s.source}-${s.target}`]))
                          }
                        >
                          <Close fontSize="small" />
                        </IconButton>
                      </Box>
                    </CardContent>
                  </Card>
                ))}
            </Box>
          </Paper>
        )}
      <Box
        sx={{
          display: 'flex',
          gap: 2,
          mb: 3,
          flexWrap: 'wrap',
          alignItems: 'center',
        }}
      >
        <Tabs value={tab} onChange={(_, v) => setTab(v)}>
          <Tab label={`All (${allFaces.length})`} />
          <Tab
            label={`Unresolved (${unresolvedFaces.length})`}
            icon={<Edit fontSize="small" />}
            iconPosition="start"
          />
          <Tab
            label={`Named (${namedFaces.length})`}
            icon={<Check fontSize="small" />}
            iconPosition="start"
          />
          <Tab
            label={`Clusters (${clusters.length})`}
            icon={<Groups fontSize="small" />}
            iconPosition="start"
          />
        </Tabs>
        <Box sx={{ flex: 1 }} />
        <TextField
          size="small"
          placeholder="Search..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          slotProps={{
            input: {
              startAdornment: (
                <InputAdornment position="start">
                  <Search fontSize="small" />
                </InputAdornment>
              ),
            },
          }}
          sx={{ minWidth: 200 }}
        />
      </Box>

      {isLoading ? (
        <Grid container spacing={2}>
          {Array.from({ length: 8 }).map((_, i) => (
            <Grid key={i} size={{ xs: 6, sm: 4, md: 3, lg: 2 }}>
              <Skeleton variant="rounded" height={160} />
            </Grid>
          ))}
        </Grid>
      ) : tab === 3 ? (
        clusters.length > 0 ? (
          <Box>
            {clusters.map(cluster => (
              <ClusterCard
                key={cluster.cluster_id}
                cluster={cluster}
                onLabelCluster={setSelectedClusterId}
                onLabelFace={setSelectedFaceId}
                onDeleteFace={handleDeleteFace}
                onZoom={setZoomFace}
                onMoveFace={setMoveFaceId}
                onMerge={() => setMergeSourceId(cluster.cluster_id)}
                onSetMain={(cid, isMain) =>
                  setMainMutation.mutate({ clusterId: cid, isMain })
                }
                onDeleteCluster={handleDeleteCluster}
                onIdentify={handleIdentifyCluster}
              />
            ))}
          </Box>
        ) : (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Groups sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No clusters yet
            </Typography>
            <Button
              variant="contained"
              startIcon={<AutoAwesome />}
              onClick={handleTriggerClustering}
              sx={{ mt: 2 }}
            >
              Run Face Clustering
            </Button>
          </Paper>
        )
      ) : filteredFaces.length > 0 ? (
        <Grid container spacing={2}>
          {filteredFaces.map((face, idx) => (
            <Grid key={face.id || idx} size={{ xs: 6, sm: 4, md: 3, lg: 2 }}>
              <FaceCard
                face={face}
                onLabel={setSelectedFaceId}
                onDelete={handleDeleteFace}
                onZoom={setZoomFace}
                onMove={setMoveFaceId}
              />
            </Grid>
          ))}
        </Grid>
      ) : (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Face sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            No faces found
          </Typography>
        </Paper>
      )}

      <LabelDialog
        key={selectedFaceId || 'face-dialog'}
        open={!!selectedFaceId}
        isCluster={false}
        isLoading={labelFaceMutation.isPending}
        existingNames={allNamesQuery.data || []}
        onClose={() => setSelectedFaceId(null)}
        onSave={name => {
          if (selectedFaceId)
            labelFaceMutation.mutate({ faceId: selectedFaceId, name });
        }}
      />
      <LabelDialog
        key={selectedClusterId ?? 'cluster-dialog'}
        open={selectedClusterId !== null}
        isCluster={true}
        isLoading={labelClusterMutation.isPending}
        existingNames={allNamesQuery.data || []}
        onClose={() => setSelectedClusterId(null)}
        onSave={name => {
          if (selectedClusterId !== null)
            labelClusterMutation.mutate({ clusterId: selectedClusterId, name });
        }}
      />
      <MoveDialog
        open={moveFaceId !== null}
        clusters={clusters}
        onClose={() => setMoveFaceId(null)}
        onMove={clusterId =>
          moveMutation.mutate({ faceId: moveFaceId!, clusterId })
        }
        onNewCluster={() => newClusterMutation.mutate([moveFaceId!])}
      />

      <Dialog
        open={mergeSourceId !== null}
        onClose={() => setMergeSourceId(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Merge Cluster</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Select a target cluster to merge into. Named clusters are shown
            first.
          </Typography>
          <List>
            {clusters
              .filter(c => c.cluster_id !== mergeSourceId)
              .sort((a, b) => {
                // Named clusters first, then by face count
                if (a.name && !b.name) return -1;
                if (!a.name && b.name) return 1;
                return (b.face_count || 0) - (a.face_count || 0);
              })
              .map(c => (
                <ListItemButton
                  key={c.cluster_id}
                  onClick={() =>
                    mergeClustersMutation.mutate({
                      from: mergeSourceId!,
                      to: c.cluster_id,
                    })
                  }
                  sx={{
                    border: c.name ? '2px solid' : 'none',
                    borderColor: c.name ? 'primary.main' : 'transparent',
                    borderRadius: 1,
                    mb: 0.5,
                    bgcolor: c.name ? 'action.selected' : 'transparent',
                  }}
                >
                  <ListItemAvatar>
                    <Avatar
                      src={
                        c.representative?.thumbnail_path
                          ? `http://localhost:8000${c.representative.thumbnail_path}`
                          : undefined
                      }
                    >
                      <Face />
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        {c.name || `Cluster #${c.cluster_id}`}
                        {c.name && (
                          <Chip
                            size="small"
                            label="Named"
                            color="primary"
                            variant="outlined"
                          />
                        )}
                      </Box>
                    }
                    secondary={`${c.face_count} faces`}
                  />
                </ListItemButton>
              ))}
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMergeSourceId(null)}>Cancel</Button>
        </DialogActions>
      </Dialog>

      <Dialog
        open={zoomFace !== null}
        onClose={() => setZoomFace(null)}
        maxWidth="md"
      >
        <Box sx={{ p: 2, textAlign: 'center' }}>
          {zoomFace?.thumbnail_path && (
            <Box
              component="img"
              src={`http://localhost:8000${zoomFace.thumbnail_path}`}
              sx={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 2 }}
            />
          )}
          <Typography variant="body2" sx={{ mt: 1 }}>
            {zoomFace?.media_path?.split(/[/\\]/).pop()}
            {zoomFace?.timestamp !== undefined &&
              ` @ ${zoomFace.timestamp.toFixed(1)}s`}
          </Typography>
        </Box>
      </Dialog>

      {(labelClusterMutation.isError ||
        labelFaceMutation.isError ||
        deleteMutation.isError ||
        mergeClustersMutation.isError) && (
          <Alert severity="error" sx={{ mt: 2 }}>
            Operation failed.
          </Alert>
        )}
    </Box>
  );
}
