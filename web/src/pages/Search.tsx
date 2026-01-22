import { useState, useCallback } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tooltip,
} from '@mui/material';
import { Search as SearchIcon, PlayArrow } from '@mui/icons-material';

import { searchHybrid, searchGranular, getLibrary, type SearchResult } from '../api/client';
import { MediaCard } from '../components/media/MediaCard';
import { SearchDebugPanel } from '../components/search/SearchDebugPanel';

interface SearchStats {
  total: number;
  duration_seconds: number;
}

interface PipelineStep {
  step: string;
  status: 'completed' | 'running' | 'pending' | 'skipped';
  detail: string;
  data?: Record<string, unknown>;
}

interface SearchResponse {
  query: string;
  video_filter: string | null;
  results: SearchResult[];
  stats: SearchStats;
  // Debug/Transparency fields
  pipeline_steps?: PipelineStep[];
  reasoning_chain?: Record<string, string>;
  search_text?: string;
  fallback_used?: string | null;
}

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [lastQuery, setLastQuery] = useState('');
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [deepSearch, setDeepSearch] = useState(false);
  const [useReranking, setUseReranking] = useState(true);

  // Overlay visibility toggles
  const [overlayToggles, setOverlayToggles] = useState({
    faces: true,
    text: true,
    objects: true,
    speakers: true,
  });

  // Fetch indexed videos for filter dropdown
  const { data: libraryData } = useQuery({
    queryKey: ['library'],
    queryFn: getLibrary,
  });

  const videos: string[] = (libraryData?.media || []).map(
    (m: { video_path: string }) => m.video_path,
  );

  // Use hybrid or granular search
  const search = useMutation({
    mutationFn: async (q: string): Promise<SearchResponse> => {
      if (deepSearch) {
        return (await searchGranular(q, selectedVideo || undefined, 10, useReranking)) as SearchResponse;
      }
      const response = await searchHybrid(q, selectedVideo || undefined, 20, useReranking);
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
          sx={{ maxWidth: 500, mx: 'auto', mb: 2 }}
        >
          Search by visual description, spoken dialogue, or specific faces.
          <br />
          <strong>Press Enter or click the button to search.</strong>
        </Typography>
        <Typography
          variant="caption"
          color="text.disabled"
          sx={{ maxWidth: 600, mx: 'auto', mb: 3, display: 'block' }}
        >
          💡 Try: "Blue shirt", "Running fast", "Clock 10:00 AM", "Crowd cheering", "Person speaking"
        </Typography>

        {/* Search Bar - Now requires Enter */}
        <Box sx={{ maxWidth: 600, mx: 'auto' }}>
          <TextField
            fullWidth
            placeholder="Search your media library... (press Enter)"
            value={query}
            onChange={e => setQuery(e.target.value)}
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

        {/* Video Filter & Deep Search */}
        <Box sx={{ maxWidth: 600, mx: 'auto', mt: 2, display: 'flex', gap: 2, justifyContent: 'center', alignItems: 'center' }}>
          {videos.length > 0 && (
            <FormControl size="small" sx={{ minWidth: 200 }}>
              <InputLabel>Filter by Video</InputLabel>
              <Select
                value={selectedVideo}
                onChange={e => setSelectedVideo(e.target.value)}
                label="Filter by Video"
              >
                <MenuItem value="">All Videos</MenuItem>
                {videos.map((v: string) => (
                  <MenuItem key={v} value={v}>
                    {v.split(/[/\\]/).pop()}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">Deep Search (AI Reasoning)</Typography>
            <Chip
              label={deepSearch ? "ON" : "OFF"}
              color={deepSearch ? "primary" : "default"}
              variant={deepSearch ? "filled" : "outlined"}
              size="small"
              onClick={() => setDeepSearch(!deepSearch)}
              sx={{ fontWeight: 700 }}
            />
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2" color="text.secondary">LLM Rerank</Typography>
            <Tooltip title="Verify results with Vision LLM (Slower but more accurate)">
              <Chip
                label={useReranking ? "ON" : "OFF"}
                color={useReranking ? "secondary" : "default"}
                variant={useReranking ? "filled" : "outlined"}
                size="small"
                onClick={() => setUseReranking(!useReranking)}
                sx={{ fontWeight: 700 }}
              />
            </Tooltip>
          </Box>
        </Box>

        {/* Overlay Toggle Buttons */}
        <Box sx={{ maxWidth: 600, mx: 'auto', mt: 2, display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Typography variant="caption" color="text.secondary" sx={{ width: '100%', textAlign: 'center', mb: 0.5 }}>
            Video Overlays (shown during playback):
          </Typography>
          <Tooltip title="Show green boxes around detected faces with names" arrow>
            <Chip
              label="🟢 Faces"
              size="small"
              color={overlayToggles.faces ? "success" : "default"}
              variant={overlayToggles.faces ? "filled" : "outlined"}
              onClick={() => setOverlayToggles(prev => ({ ...prev, faces: !prev.faces }))}
              sx={{ fontWeight: 600 }}
            />
          </Tooltip>
          <Tooltip title="Show blue boxes around detected text (OCR)" arrow>
            <Chip
              label="🔵 Text/OCR"
              size="small"
              color={overlayToggles.text ? "info" : "default"}
              variant={overlayToggles.text ? "filled" : "outlined"}
              onClick={() => setOverlayToggles(prev => ({ ...prev, text: !prev.text }))}
              sx={{ fontWeight: 600 }}
            />
          </Tooltip>
          <Tooltip title="Show red boxes around detected objects" arrow>
            <Chip
              label="🔴 Objects"
              size="small"
              color={overlayToggles.objects ? "error" : "default"}
              variant={overlayToggles.objects ? "filled" : "outlined"}
              onClick={() => setOverlayToggles(prev => ({ ...prev, objects: !prev.objects }))}
              sx={{ fontWeight: 600 }}
            />
          </Tooltip>
          <Tooltip title="Show yellow dashed boxes around active speakers" arrow>
            <Chip
              label="🟡 Speakers"
              size="small"
              color={overlayToggles.speakers ? "warning" : "default"}
              variant={overlayToggles.speakers ? "filled" : "outlined"}
              onClick={() => setOverlayToggles(prev => ({ ...prev, speakers: !prev.speakers }))}
              sx={{ fontWeight: 600 }}
            />
          </Tooltip>
        </Box>
      </Box>

      {/* Stats Bar */}
      {stats && (
        <Alert severity="info" sx={{ mb: 2, borderRadius: 2 }}>
          <strong>"{lastQuery}"</strong>: {stats.total} results
          {selectedVideo && (
            <> (filtered to {selectedVideo.split(/[/\\]/).pop()})</>
          )}
          {stats.duration_seconds && (
            <> | Search time: {(stats.duration_seconds * 1000).toFixed(0)}ms</>
          )}
        </Alert>
      )}

      {/* Search Debug Panel - Shows pipeline transparency */}
      {search.data?.pipeline_steps && search.data.pipeline_steps.length > 0 && (
        <SearchDebugPanel
          pipelineSteps={search.data.pipeline_steps}
          reasoningChain={search.data.reasoning_chain}
          searchText={search.data.search_text}
          fallbackUsed={search.data.fallback_used}
          durationMs={stats ? stats.duration_seconds * 1000 : undefined}
          isLoading={search.isPending}
        />
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
            {results.map((item: SearchResult, idx: number) => (
              <Grid key={item.id || idx} size={{ xs: 12, sm: 6, md: 4, lg: 3 }}>
                <MediaCard item={item} searchQuery={lastQuery} overlayToggles={overlayToggles} />
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
          <Box
            sx={{
              display: 'flex',
              gap: 1,
              justifyContent: 'center',
              flexWrap: 'wrap',
              mt: 2,
            }}
          >
            {[
              'person on bike',
              'bowling strike',
              'red jacket',
              'two people talking',
              'outdoor scene',
            ].map(s => (
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
