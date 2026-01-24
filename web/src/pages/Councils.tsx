import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Card,
    CardContent,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Switch,
    Slider,
    Chip,
    IconButton,
    Tooltip,
    Alert,
} from '@mui/material';
import {
    Gavel,
    AutoAwesome,
    Memory,
    Science,
    Business,
    Info,
} from '@mui/icons-material';

import {
    getCouncilsConfig,
    setCouncilMode,
    updateCouncilModel,
    type CouncilConfig,
    type ModelSpec,
} from '../api/client';

export default function CouncilsPage() {
    const queryClient = useQueryClient();

    const { data: config, isLoading, error } = useQuery({
        queryKey: ['councils'],
        queryFn: getCouncilsConfig,
    });

    const modeMutation = useMutation({
        mutationFn: setCouncilMode,
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['councils'] }),
    });

    const modelMutation = useMutation({
        mutationFn: (vars: { council: string; model: string; update: any }) =>
            updateCouncilModel(vars.council, vars.model, vars.update),
        onSuccess: () => queryClient.invalidateQueries({ queryKey: ['councils'] }),
    });

    if (isLoading) return <Typography>Loading Councils...</Typography>;
    if (error) return <Alert severity="error">Failed to load councils</Alert>;
    if (!config) return null;

    return (
        <Box>
            <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="h4" fontWeight={700} gutterBottom>
                        Discovery Councils
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                        Configure the AI voting blocs that power search and ranking.
                    </Typography>
                </Box>
                <Gavel sx={{ fontSize: 40, color: 'text.secondary', opacity: 0.5 }} />
            </Box>

            {/* Global Mode Control */}
            <Paper sx={{ p: 3, mb: 4, borderRadius: 2, bgcolor: 'background.default', border: 1, borderColor: 'divider' }}>
                <Grid container spacing={3} alignItems="center">
                    <Grid size={{ xs: 12, md: 6 }}>
                        <Typography variant="h6" gutterBottom>Operating Mode</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Determines which models are eligible to vote.
                        </Typography>
                    </Grid>
                    <Grid size={{ xs: 12, md: 6 }}>
                        <FormControl fullWidth>
                            <InputLabel>Council Mode</InputLabel>
                            <Select
                                value={config.mode}
                                label="Council Mode"
                                onChange={(e) => modeMutation.mutate(e.target.value)}
                            >
                                <MenuItem value="oss_only">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Science fontSize="small" /> OSS Only (Local/Free)
                                    </Box>
                                </MenuItem>
                                <MenuItem value="commercial_only">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <Business fontSize="small" /> Commercial Only (API/Paid)
                                    </Box>
                                </MenuItem>
                                <MenuItem value="combined">
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        <AutoAwesome fontSize="small" /> Combined (Best Quality)
                                    </Box>
                                </MenuItem>
                            </Select>
                        </FormControl>
                    </Grid>
                </Grid>
            </Paper>

            {/* Council Lists */}
            <Grid container spacing={3}>
                {Object.entries(config.councils).map(([name, council]: [string, any]) => (
                    <Grid size={{ xs: 12 }} key={name}>
                        <Paper sx={{ p: 0, overflow: 'hidden', borderRadius: 2 }}>
                            <Box sx={{ p: 2, bgcolor: 'primary.main', color: 'white' }}>
                                <Typography variant="h6" sx={{ textTransform: 'uppercase', letterSpacing: 1 }}>
                                    {name} Council
                                </Typography>
                            </Box>
                            <Box sx={{ p: 3 }}>
                                <Grid container spacing={2}>
                                    {council.models.map((model: ModelSpec) => (
                                        <Grid size={{ xs: 12, md: 6, lg: 4 }} key={model.name}>
                                            <Card variant="outlined" sx={{
                                                opacity: model.enabled ? 1 : 0.6,
                                                borderColor: model.enabled ? 'primary.light' : 'divider'
                                            }}>
                                                <CardContent>
                                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                                        <Chip
                                                            label={model.model_type}
                                                            size="small"
                                                            color={model.model_type === 'oss' ? 'success' : 'secondary'}
                                                            variant="outlined"
                                                        />
                                                        <Switch
                                                            checked={model.enabled}
                                                            size="small"
                                                            onChange={(e) => modelMutation.mutate({
                                                                council: name,
                                                                model: model.name,
                                                                update: { enabled: e.target.checked }
                                                            })}
                                                        />
                                                    </Box>

                                                    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                                                        {model.name}
                                                    </Typography>

                                                    <Typography variant="caption" color="text.secondary" display="block" paragraph>
                                                        {model.description}
                                                    </Typography>

                                                    {model.vram_gb > 0 && (
                                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 2 }}>
                                                            <Memory fontSize="inherit" color="disabled" />
                                                            <Typography variant="caption" color="text.disabled">
                                                                {model.vram_gb} GB VRAM
                                                            </Typography>
                                                        </Box>
                                                    )}

                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                        <Typography variant="caption">Weight:</Typography>
                                                        <Slider
                                                            size="small"
                                                            value={model.weight}
                                                            min={0}
                                                            max={2}
                                                            step={0.1}
                                                            onChangeCommitted={(_, v) => modelMutation.mutate({
                                                                council: name,
                                                                model: model.name,
                                                                update: { weight: v as number }
                                                            })}
                                                            disabled={!model.enabled}
                                                            sx={{ flex: 1 }}
                                                        />
                                                        <Typography variant="caption" fontWeight="bold">
                                                            {model.weight.toFixed(1)}
                                                        </Typography>
                                                    </Box>
                                                </CardContent>
                                            </Card>
                                        </Grid>
                                    ))}
                                </Grid>
                            </Box>
                        </Paper>
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
}
