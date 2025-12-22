import { useState } from 'react';
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
} from '@mui/material';
import { Face, Search, Check, Edit, Delete, ZoomIn } from '@mui/icons-material';
import { motion } from 'framer-motion';

import { getUnresolvedFaces, getNamedFaces, nameSingleFace, deleteFace } from '../api/client';

interface FaceCluster {
    id: string;
    cluster_id: number;
    name: string | null;
    media_path?: string;
    timestamp?: number;
    thumbnail_path?: string;
}

function FaceCard({
    face,
    onLabel,
    onDelete,
    onZoom,
}: {
    face: FaceCluster;
    onLabel: (faceId: string) => void;
    onDelete: (faceId: string) => void;
    onZoom: (face: FaceCluster) => void;
}) {
    const [imageLoaded, setImageLoaded] = useState(false);
    const [imageError, setImageError] = useState(false);

    const thumbUrl = face.thumbnail_path
        ? `http://localhost:8000${face.thumbnail_path}`
        : null;

    const hasValidImage = thumbUrl && !imageError;

    return (
        <Paper
            component={motion.div}
            whileHover={{ scale: 1.02, boxShadow: '0 8px 30px rgba(0,0,0,0.12)' }}
            sx={{
                p: 2,
                borderRadius: 3,
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                position: 'relative',
                background: (theme) =>
                    theme.palette.mode === 'dark'
                        ? 'linear-gradient(145deg, rgba(30,30,40,0.9), rgba(20,20,30,0.95))'
                        : 'linear-gradient(145deg, rgba(255,255,255,0.95), rgba(245,245,250,0.9))',
                '&:hover': {
                    boxShadow: 6,
                },
            }}
            onClick={() => onLabel(face.id)}
        >
            <IconButton
                size="small"
                sx={{
                    position: 'absolute',
                    top: 4,
                    right: 4,
                    opacity: 0,
                    transition: 'opacity 0.2s',
                    '.MuiPaper-root:hover &': {
                        opacity: 0.8,
                    },
                    '&:hover': {
                        opacity: 1,
                        color: 'error.main',
                    },
                }}
                onClick={(e) => {
                    e.stopPropagation();
                    onDelete(face.id);
                }}
            >
                <Delete fontSize="small" />
            </IconButton>

            {/* Zoom button */}
            {hasValidImage && (
                <IconButton
                    size="small"
                    sx={{
                        position: 'absolute',
                        top: 4,
                        left: 4,
                        opacity: 0,
                        transition: 'opacity 0.2s',
                        bgcolor: 'rgba(0,0,0,0.5)',
                        '.MuiPaper-root:hover &': {
                            opacity: 0.8,
                        },
                        '&:hover': {
                            opacity: 1,
                            bgcolor: 'primary.main',
                        },
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                        onZoom(face);
                    }}
                >
                    <ZoomIn fontSize="small" sx={{ color: 'white' }} />
                </IconButton>
            )}

            {/* Face Thumbnail - Google Photos Style */}
            <Box
                sx={{
                    position: 'relative',
                    width: 96,
                    height: 96,
                    mx: 'auto',
                    mb: 2,
                }}
            >
                {/* Loading skeleton */}
                {hasValidImage && !imageLoaded && (
                    <Skeleton
                        variant="circular"
                        width={96}
                        height={96}
                        sx={{ position: 'absolute', top: 0, left: 0 }}
                    />
                )}

                {/* Actual face image or fallback */}
                <Avatar
                    src={hasValidImage ? thumbUrl : undefined}
                    onLoad={() => setImageLoaded(true)}
                    onError={() => setImageError(true)}
                    sx={{
                        width: 96,
                        height: 96,
                        border: '3px solid',
                        borderColor: face.name ? 'success.main' : 'primary.main',
                        boxShadow: '0 4px 14px rgba(0,0,0,0.15)',
                        bgcolor: 'grey.800',
                        fontSize: 40,
                        opacity: hasValidImage && !imageLoaded ? 0 : 1,
                        transition: 'opacity 0.3s ease, transform 0.3s ease',
                        '& img': {
                            objectFit: 'cover',
                        },
                    }}
                >
                    {!hasValidImage && <Face fontSize="large" />}
                </Avatar>

                {/* Subtle glow effect for named faces */}
                {face.name && (
                    <Box
                        sx={{
                            position: 'absolute',
                            top: -4,
                            left: -4,
                            right: -4,
                            bottom: -4,
                            borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(76,175,80,0.2) 0%, transparent 70%)',
                            pointerEvents: 'none',
                        }}
                    />
                )}
            </Box>

            {face.name ? (
                <Chip
                    label={face.name}
                    color="success"
                    size="small"
                    icon={<Check />}
                    sx={{
                        fontWeight: 600,
                        maxWidth: '100%',
                        '& .MuiChip-label': {
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                        },
                    }}
                />
            ) : (
                <Chip
                    label={face.timestamp !== undefined ? `${face.timestamp.toFixed(1)}s` : `#${face.cluster_id}`}
                    variant="outlined"
                    size="small"
                    sx={{
                        borderColor: 'primary.light',
                        color: 'text.secondary',
                    }}
                />
            )}

            <Typography
                variant="caption"
                display="block"
                color="text.secondary"
                sx={{
                    mt: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    maxWidth: '100%',
                }}
            >
                {face.media_path?.split(/[/\\]/).pop() || 'Unknown source'}
            </Typography>
        </Paper>
    );
}

function LabelDialog({
    open,
    faceId: _faceId,
    onClose,
    onSave,
    isLoading,
}: {
    open: boolean;
    faceId: string | null;
    onClose: () => void;
    onSave: (name: string) => void;
    isLoading: boolean;
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
            <DialogTitle>Label Face</DialogTitle>
            <DialogContent>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Enter a name for this person. All faces in this cluster will be labeled with this name.
                </Typography>
                <TextField
                    fullWidth
                    autoFocus
                    label="Person's Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                    placeholder="e.g., John Doe"
                />
            </DialogContent>
            <DialogActions>
                <Button onClick={onClose}>Cancel</Button>
                <Button
                    variant="contained"
                    onClick={handleSave}
                    disabled={!name.trim() || isLoading}
                >
                    {isLoading ? 'Saving...' : 'Save Label'}
                </Button>
            </DialogActions>
        </Dialog>
    );
}

export default function FacesPage() {
    const [tab, setTab] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedFaceId, setSelectedFaceId] = useState<string | null>(null);
    const [zoomFace, setZoomFace] = useState<FaceCluster | null>(null);
    const queryClient = useQueryClient();

    const unresolvedFaces = useQuery({
        queryKey: ['faces', 'unresolved'],
        queryFn: () => getUnresolvedFaces(),
    });

    const namedFaces = useQuery({
        queryKey: ['faces', 'named'],
        queryFn: () => getNamedFaces(),
    });

    const labelMutation = useMutation({
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

    const handleLabel = (name: string) => {
        if (selectedFaceId !== null) {
            labelMutation.mutate({ faceId: selectedFaceId, name });
        }
    };

    const handleDelete = (faceId: string) => {
        if (confirm('Delete this face?')) {
            deleteMutation.mutate(faceId);
        }
    };

    const faces = tab === 0 ? unresolvedFaces.data?.faces : namedFaces.data?.faces;
    const isLoading = tab === 0 ? unresolvedFaces.isLoading : namedFaces.isLoading;

    const filteredFaces = faces?.filter((f: FaceCluster) => {
        if (!searchQuery) return true;
        const query = searchQuery.toLowerCase();
        return (
            f.name?.toLowerCase().includes(query) ||
            f.media_path?.toLowerCase().includes(query) ||
            String(f.cluster_id).includes(query)
        );
    });

    return (
        <Box>
            <Typography variant="h4" fontWeight={700} gutterBottom>
                Face Gallery
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
                Review detected faces and assign names for better search results.
            </Typography>

            {/* Tabs and Search */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap', alignItems: 'center' }}>
                <Tabs value={tab} onChange={(_, v) => setTab(v)}>
                    <Tab
                        label={`Unresolved (${unresolvedFaces.data?.faces?.length || 0})`}
                        icon={<Edit fontSize="small" />}
                        iconPosition="start"
                    />
                    <Tab
                        label={`Named (${namedFaces.data?.faces?.length || 0})`}
                        icon={<Check fontSize="small" />}
                        iconPosition="start"
                    />
                </Tabs>
                <Box sx={{ flex: 1 }} />
                <TextField
                    size="small"
                    placeholder="Search faces..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
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

            {/* Grid */}
            {isLoading ? (
                <Grid container spacing={2}>
                    {Array.from({ length: 8 }).map((_, i) => (
                        <Grid key={i} size={{ xs: 6, sm: 4, md: 3, lg: 2 }}>
                            <Skeleton variant="rounded" height={160} />
                        </Grid>
                    ))}
                </Grid>
            ) : filteredFaces?.length > 0 ? (
                <Grid container spacing={2}>
                    {filteredFaces.map((face: FaceCluster, idx: number) => (
                        <Grid key={face.id || idx} size={{ xs: 6, sm: 4, md: 3, lg: 2 }}>
                            <FaceCard
                                face={face}
                                onLabel={setSelectedFaceId}
                                onDelete={handleDelete}
                                onZoom={setZoomFace}
                            />
                        </Grid>
                    ))}
                </Grid>
            ) : (
                <Paper sx={{ p: 6, textAlign: 'center', borderRadius: 3, bgcolor: 'action.hover' }}>
                    <Face sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">
                        {tab === 0 ? 'No unresolved faces' : 'No named faces yet'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                        {tab === 0
                            ? 'All detected faces have been labeled'
                            : 'Label some faces from the Unresolved tab'}
                    </Typography>
                </Paper>
            )}

            {/* Label Dialog */}
            <LabelDialog
                open={selectedFaceId !== null}
                faceId={selectedFaceId}
                onClose={() => setSelectedFaceId(null)}
                onSave={handleLabel}
                isLoading={labelMutation.isPending}
            />

            {/* Zoom Dialog for full-size preview */}
            <Dialog
                open={zoomFace !== null}
                onClose={() => setZoomFace(null)}
                maxWidth="md"
                PaperProps={{
                    sx: {
                        bgcolor: 'transparent',
                        boxShadow: 'none',
                        overflow: 'visible',
                    },
                }}
            >
                <Box
                    sx={{
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        p: 2,
                    }}
                >
                    {zoomFace?.thumbnail_path && (
                        <Box
                            component="img"
                            src={`http://localhost:8000${zoomFace.thumbnail_path}`}
                            alt={zoomFace.name || 'Face'}
                            sx={{
                                maxWidth: '80vw',
                                maxHeight: '70vh',
                                borderRadius: 3,
                                boxShadow: '0 8px 40px rgba(0,0,0,0.5)',
                                objectFit: 'contain',
                            }}
                        />
                    )}
                    <Box sx={{ mt: 2, textAlign: 'center' }}>
                        {zoomFace?.name && (
                            <Chip
                                label={zoomFace.name}
                                color="success"
                                icon={<Check />}
                                sx={{ mb: 1 }}
                            />
                        )}
                        <Typography
                            variant="body2"
                            color="text.secondary"
                            sx={{ color: 'white', textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}
                        >
                            {zoomFace?.media_path?.split(/[/\\]/).pop()}
                            {zoomFace?.timestamp !== undefined && ` @ ${zoomFace.timestamp.toFixed(1)}s`}
                        </Typography>
                    </Box>
                </Box>
            </Dialog>

            {labelMutation.isError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                    Failed to save label. Please try again.
                </Alert>
            )}
        </Box>
    );
}
