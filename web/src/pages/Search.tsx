import { useState, useCallback } from 'react';
import { useMutation } from '@tanstack/react-query';
import {
    Box,
    Typography,
    TextField,
    InputAdornment,
    Grid,
    Paper,
    Chip,
    CircularProgress,
    IconButton,
    Alert,
} from '@mui/material';
import { Search as SearchIcon, PlayArrow } from '@mui/icons-material';

import { searchMedia } from '../api/client';
import { MediaCard } from '../components/media/MediaCard';

interface SearchStats {
    query: string;
    dialogue_count: number;
    visual_count: number;
    total_frames_scanned: number;
    returned_count: number;
    score_range?: {
        min: number;
        max: number;
        avg: number;
    };
}

interface SearchResponse {
    results: any[];
    stats: SearchStats;
}

export default function SearchPage() {
    const [query, setQuery] = useState('');
    const [lastQuery, setLastQuery] = useState('');

    // Use mutation instead of query to require explicit trigger
    const search = useMutation({
        mutationFn: async (q: string): Promise<SearchResponse> => {
            const response = await searchMedia(q);
            // Handle both old array format and new {results, stats} format
            if (Array.isArray(response)) {
                return { results: response, stats: { query: q, dialogue_count: 0, visual_count: response.length, total_frames_scanned: 0, returned_count: response.length } };
            }
            return response as SearchResponse;
        },
    });

    const handleSearch = useCallback(() => {
        if (query.trim().length > 0) {
            setLastQuery(query);
            search.mutate(query);
        }
    }, [query, search]);

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    };

    const handleChipClick = (suggestion: string) => {
        setQuery(suggestion);
        setLastQuery(suggestion);
        search.mutate(suggestion);
    };

    const stats = search.data?.stats;
    const results = search.data?.results || [];

    return (
        <Box>
            {/* Hero */}
            <Box sx={{ textAlign: 'center', mb: 6, mt: 4 }}>
                <Typography
                    variant="h3"
                    component="h1"
                    sx={{ fontWeight: 800, mb: 2, letterSpacing: '-0.02em' }}
                >
                    Find moments,{' '}
                    <Box component="span" sx={{ color: 'text.secondary', fontWeight: 400 }}>
                        not just files.
                    </Box>
                </Typography>
                <Typography
                    variant="body1"
                    color="text.secondary"
                    sx={{ maxWidth: 500, mx: 'auto', mb: 4 }}
                >
                    Search by visual description, spoken dialogue, or specific faces.
                    <br />
                    <strong>Press Enter or click the button to search.</strong>
                </Typography>

                {/* Search Bar - Now requires Enter */}
                <Box sx={{ maxWidth: 600, mx: 'auto' }}>
                    <TextField
                        fullWidth
                        placeholder="Search your media library... (press Enter)"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        slotProps={{
                            input: {
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <SearchIcon color="action" />
                                    </InputAdornment>
                                ),
                                endAdornment: (
                                    <InputAdornment position="end">
                                        {search.isPending ? (
                                            <CircularProgress size={24} />
                                        ) : (
                                            <IconButton
                                                onClick={handleSearch}
                                                color="primary"
                                                disabled={!query.trim()}
                                            >
                                                <PlayArrow />
                                            </IconButton>
                                        )}
                                    </InputAdornment>
                                ),
                            },
                        }}
                        sx={{
                            '& .MuiOutlinedInput-root': {
                                borderRadius: 3,
                                bgcolor: 'background.paper',
                            },
                        }}
                    />
                </Box>
            </Box>

            {/* Stats Bar */}
            {stats && (
                <Alert severity="info" sx={{ mb: 3, borderRadius: 2 }}>
                    <strong>"{stats.query}"</strong>: {stats.returned_count} results
                    ({stats.visual_count} visual, {stats.dialogue_count} dialogue)
                    {stats.score_range && (
                        <> | Scores: {stats.score_range.min.toFixed(2)} - {stats.score_range.max.toFixed(2)}</>
                    )}
                </Alert>
            )}

            {/* Results */}
            {lastQuery && (
                <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <Typography variant="body2" color="text.secondary">
                            {results.length} results for "{lastQuery}"
                        </Typography>
                        {search.isPending && <CircularProgress size={16} />}
                    </Box>

                    <Grid container spacing={3}>
                        {results.map((item: any, idx: number) => (
                            <Grid key={item.id || idx} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                                <MediaCard item={item} />
                            </Grid>
                        ))}
                    </Grid>

                    {search.isSuccess && results.length === 0 && (
                        <Paper
                            sx={{
                                p: 6,
                                textAlign: 'center',
                                bgcolor: 'action.hover',
                                borderRadius: 3,
                            }}
                        >
                            <Typography color="text.secondary">
                                No matching moments found. Try a different query.
                            </Typography>
                        </Paper>
                    )}
                </Box>
            )}

            {/* Empty State */}
            {!lastQuery && (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                        Try searching for:
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap', mt: 2 }}>
                        {['person on bike', 'bowling strike', 'red jacket', 'two people talking', 'outdoor scene'].map((s) => (
                            <Chip
                                key={s}
                                label={s}
                                variant="outlined"
                                onClick={() => handleChipClick(s)}
                                sx={{ cursor: 'pointer' }}
                            />
                        ))}
                    </Box>
                </Box>
            )}
        </Box>
    );
}

