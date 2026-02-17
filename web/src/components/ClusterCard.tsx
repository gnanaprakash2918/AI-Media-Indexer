import { useState } from 'react';
import {
    Box,
    Typography,
    Paper,
    Button,
    Chip,
    Avatar,
    IconButton,
    Badge,
    Tooltip,
    Collapse,
    Divider,
} from '@mui/material';
import {
    Face,
    Edit,
    Delete,
    AutoAwesome,
    Groups,
    ExpandMore,
    ExpandLess,
    MoveUp,
    Star,
    StarBorder,
} from '@mui/icons-material';

import FaceCard from './FaceCard';
import { API_BASE } from '../api/client';
import type { FaceData, FaceClusterData } from '../types';

export default function ClusterCard({
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
        ? `${API_BASE}${representative.thumbnail_path}`
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
