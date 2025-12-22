
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
} from '@mui/material';
import { RecordVoiceOver, Delete } from '@mui/icons-material';
import { getVoiceSegments, deleteVoiceSegment } from '../api/client';
import { IconButton, Tooltip } from '@mui/material';

export const Voices: React.FC = () => {
    const [segments, setSegments] = useState<any[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const data = await getVoiceSegments(undefined, 200);
            setSegments(data.segments);
        } catch (err) {
            console.error(err);
            setError('Failed to load voice segments');
        }
    };

    const handleDelete = async (id: string) => {
        try {
            await deleteVoiceSegment(id);
            setSegments(segments.filter(s => s.id !== id));
        } catch (err) {
            console.error(err);
            setError('Failed to delete segment');
        }
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <RecordVoiceOver fontSize="large" color="primary" />
                Voice Intelligence
            </Typography>

            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

            {segments.length === 0 ? (
                <Alert severity="info">No voice segments found. Ensure ingestion is running and "Voice Analysis" is enabled.</Alert>
            ) : (
                <TableContainer component={Paper}>
                    <Table size="small">
                        <TableHead>
                            <TableRow>
                                <TableCell>Media</TableCell>
                                <TableCell>Time Range</TableCell>
                                <TableCell>Speaker</TableCell>
                                <TableCell>Audio</TableCell>
                                <TableCell>Confidence</TableCell>
                                <TableCell>Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {segments.map((seg, idx) => (
                                <TableRow key={idx}>
                                    <TableCell>{seg.media_path.split(/[/\\]/).pop()}</TableCell>
                                    <TableCell>{seg.start.toFixed(2)}s - {seg.end.toFixed(2)}s</TableCell>
                                    <TableCell>
                                        <Chip
                                            label={seg.speaker_label}
                                            color="primary"
                                            variant="outlined"
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>
                                        {seg.audio_path ? (
                                            <audio controls src={`http://localhost:8000${seg.audio_path}`} style={{ height: 32, width: 200 }} />
                                        ) : (
                                            <Typography variant="caption" color="text.secondary">
                                                No audio
                                            </Typography>
                                        )}
                                    </TableCell>
                                    <TableCell>
                                        {(1.0).toFixed(2)} {/* Confidence placeholder */}
                                    </TableCell>
                                    <TableCell>
                                        <Tooltip title="Delete Segment">
                                            <IconButton onClick={() => handleDelete(seg.id)} size="small" color="error">
                                                <Delete />
                                            </IconButton>
                                        </Tooltip>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </TableContainer>
            )}
        </Container>
    );
};

export default Voices;
