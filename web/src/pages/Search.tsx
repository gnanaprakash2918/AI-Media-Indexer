import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    Box,
    Typography,
    TextField,
    InputAdornment,
    Grid,
    Paper,
    Chip,
    CircularProgress,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

import { searchMedia } from '../api/client';
import { MediaCard } from '../components/media/MediaCard';

export default function SearchPage() {
    const [query, setQuery] = useState('');
    const [debouncedQuery, setDebouncedQuery] = useState('');

    const handleSearch = (value: string) => {
        setQuery(value);
        // Simple debounce
        const timer = setTimeout(() => setDebouncedQuery(value), 300);
        return () => clearTimeout(timer);
    };

    const search = useQuery({
        queryKey: ['search', debouncedQuery],
        queryFn: () => searchMedia(debouncedQuery),
        enabled: debouncedQuery.length > 2,
    });

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
                </Typography>

                {/* Search Bar */}
                <Box sx={{ maxWidth: 600, mx: 'auto' }}>
                    <TextField
                        fullWidth
                        placeholder="Search your media library..."
                        value={query}
                        onChange={(e) => handleSearch(e.target.value)}
                        slotProps={{
                            input: {
                                startAdornment: (
                                    <InputAdornment position="start">
                                        <SearchIcon color="action" />
                                    </InputAdornment>
                                ),
                                endAdornment: search.isLoading ? (
                                    <InputAdornment position="end">
                                        <CircularProgress size={20} />
                                    </InputAdornment>
                                ) : null,
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

            {/* Results */}
            {debouncedQuery && (
                <Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                        <Typography variant="body2" color="text.secondary">
                            {search.data?.length ?? 0} results for "{debouncedQuery}"
                        </Typography>
                        {search.isLoading && <CircularProgress size={16} />}
                    </Box>

                    <Grid container spacing={3}>
                        {search.data?.map((item: any, idx: number) => (
                            <Grid key={item.id || idx} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                                <MediaCard item={item} />
                            </Grid>
                        ))}
                    </Grid>

                    {search.isSuccess && search.data?.length === 0 && (
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
            {!debouncedQuery && (
                <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                        Start typing to search
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap', mt: 2 }}>
                        {['red car', 'beach sunset', 'talking about money', 'person laughing'].map((s) => (
                            <Chip
                                key={s}
                                label={s}
                                variant="outlined"
                                onClick={() => handleSearch(s)}
                                sx={{ cursor: 'pointer' }}
                            />
                        ))}
                    </Box>
                </Box>
            )}
        </Box>
    );
}
