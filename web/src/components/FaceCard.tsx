import { useState } from 'react';
import {
    Paper,
    Typography,
    Chip,
    Avatar,
    IconButton,
} from '@mui/material';
import {
    Face,
    Delete,
    ZoomIn,
    MoveUp,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import type { FaceData } from '../types';

export default function FaceCard({
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
