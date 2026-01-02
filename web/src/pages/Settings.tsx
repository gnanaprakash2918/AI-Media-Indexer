import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Chip,
    Divider,
    List,
    ListItem,
    ListItemText,
    Alert,
    CircularProgress,
    TextField,
    Button,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Switch,
    FormControlLabel,
} from '@mui/material';
import {
    Memory,
    Videocam,
    CloudQueue,
    Settings as SettingsIcon,
    Check,
    Warning,
    Save,
    Key,
} from '@mui/icons-material';

import { getConfig, healthCheck, apiClient } from '../api/client';

interface ConfigData {
    device: string;
    compute_type: string;
    qdrant_backend: string;
    qdrant_host: string;
    qdrant_port: number;
    frame_interval: number;
    frame_sample_ratio: number;
    face_detection_threshold: number;
    face_detection_resolution: number;
    language: string;
    llm_provider: string;
    enable_voice_analysis: boolean;
    enable_resource_monitoring: boolean;
    max_cpu_percent: number;
    max_ram_percent: number;
    google_api_key?: string;
    ollama_base_url?: string;
    ollama_model?: string;
    hf_token?: string;
}

function ConfigSection({
    title,
    icon,
    children,
}: {
    title: string;
    icon: React.ReactNode;
    children: React.ReactNode;
}) {
    return (
        <Paper sx={{ p: 2, borderRadius: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                {icon}
                <Typography variant="subtitle1" fontWeight={600}>
                    {title}
                </Typography>
            </Box>
            <Divider sx={{ mb: 1.5 }} />
            {children}
        </Paper>
    );
}

export default function SettingsPage() {
    const queryClient = useQueryClient();
    const [editMode, setEditMode] = useState(false);
    const [formData, setFormData] = useState<Partial<ConfigData>>({});
    const [saved, setSaved] = useState(false);

    const config = useQuery({
        queryKey: ['config'],
        queryFn: getConfig,
    });

    const health = useQuery({
        queryKey: ['health'],
        queryFn: healthCheck,
    });

    useEffect(() => {
        if (config.data) {
            setFormData(config.data);
        }
    }, [config.data]);

    const saveMutation = useMutation({
        mutationFn: async (data: Partial<ConfigData>) => {
            const res = await apiClient.post('/config', data);
            return res.data;
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['config'] });
            setSaved(true);
            setTimeout(() => setSaved(false), 3000);
        },
    });

    const handleChange = (key: keyof ConfigData, value: any) => {
        setFormData(prev => ({ ...prev, [key]: value }));
        setEditMode(true);
    };

    const handleSave = () => {
        saveMutation.mutate(formData);
        setEditMode(false);
    };

    if (config.isLoading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                <CircularProgress size={32} />
            </Box>
        );
    }

    if (config.isError) {
        return (
            <Alert severity="error" sx={{ m: 2 }}>
                Failed to load configuration. Make sure the backend is running.
            </Alert>
        );
    }

    return (
        <Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                <Box>
                    <Typography variant="h5" fontWeight={700}>Settings</Typography>
                    <Typography variant="body2" color="text.secondary">
                        Configure API keys and runtime settings
                    </Typography>
                </Box>
                {editMode && (
                    <Button
                        variant="contained"
                        startIcon={<Save />}
                        onClick={handleSave}
                        disabled={saveMutation.isPending}
                    >
                        {saveMutation.isPending ? 'Saving...' : 'Save Changes'}
                    </Button>
                )}
            </Box>

            {saved && (
                <Alert severity="success" sx={{ mb: 2 }}>
                    Settings saved. Some changes require a backend restart.
                </Alert>
            )}

            <Grid container spacing={2}>
                {/* Connection Status */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <ConfigSection title="Status" icon={<CloudQueue color="primary" fontSize="small" />}>
                        <List disablePadding dense>
                            <ListItem sx={{ px: 0 }}>
                                <ListItemText primary="Backend API" />
                                <Chip
                                    icon={health.isSuccess ? <Check /> : <Warning />}
                                    label={health.isSuccess ? 'Online' : 'Offline'}
                                    color={health.isSuccess ? 'success' : 'error'}
                                    size="small"
                                />
                            </ListItem>
                            <ListItem sx={{ px: 0 }}>
                                <ListItemText primary="Qdrant" secondary={`${formData.qdrant_host}:${formData.qdrant_port}`} />
                                <Chip
                                    label={health.data?.qdrant === 'connected' ? 'Connected' : 'Offline'}
                                    color={health.data?.qdrant === 'connected' ? 'success' : 'warning'}
                                    size="small"
                                />
                            </ListItem>
                        </List>
                    </ConfigSection>
                </Grid>

                {/* API Keys */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <ConfigSection title="API Keys" icon={<Key color="primary" fontSize="small" />}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>LLM Provider</InputLabel>
                                <Select
                                    value={formData.llm_provider || 'ollama'}
                                    label="LLM Provider"
                                    onChange={(e) => handleChange('llm_provider', e.target.value)}
                                >
                                    <MenuItem value="ollama">Ollama (Local)</MenuItem>
                                    <MenuItem value="gemini">Google Gemini</MenuItem>
                                    <MenuItem value="openai">OpenAI</MenuItem>
                                </Select>
                            </FormControl>

                            {formData.llm_provider === 'gemini' && (
                                <TextField
                                    size="small"
                                    label="Google API Key"
                                    type="password"
                                    value={formData.google_api_key || ''}
                                    onChange={(e) => handleChange('google_api_key', e.target.value)}
                                    placeholder="AIza..."
                                />
                            )}

                            {formData.llm_provider === 'ollama' && (
                                <>
                                    <TextField
                                        size="small"
                                        label="Ollama URL"
                                        value={formData.ollama_base_url || 'http://localhost:11434'}
                                        onChange={(e) => handleChange('ollama_base_url', e.target.value)}
                                    />
                                    <TextField
                                        size="small"
                                        label="Ollama Model"
                                        value={formData.ollama_model || 'llava:7b'}
                                        onChange={(e) => handleChange('ollama_model', e.target.value)}
                                        placeholder="llava:7b, llama3, etc."
                                        helperText="Model must be pulled via 'ollama pull <model>'"
                                    />
                                </>
                            )}

                            <TextField
                                size="small"
                                label="HuggingFace Token"
                                type="password"
                                value={formData.hf_token || ''}
                                onChange={(e) => handleChange('hf_token', e.target.value)}
                                placeholder="hf_..."
                            />
                        </Box>
                    </ConfigSection>
                </Grid>

                {/* Hardware */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <ConfigSection title="Hardware" icon={<Memory color="primary" fontSize="small" />}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Device</InputLabel>
                                <Select
                                    value={formData.device || 'cuda'}
                                    label="Device"
                                    onChange={(e) => handleChange('device', e.target.value)}
                                >
                                    <MenuItem value="cuda">CUDA (GPU)</MenuItem>
                                    <MenuItem value="mps">MPS (Apple)</MenuItem>
                                    <MenuItem value="cpu">CPU</MenuItem>
                                </Select>
                            </FormControl>
                            <FormControl size="small" fullWidth>
                                <InputLabel>Compute Type</InputLabel>
                                <Select
                                    value={formData.compute_type || 'float16'}
                                    label="Compute Type"
                                    onChange={(e) => handleChange('compute_type', e.target.value)}
                                >
                                    <MenuItem value="float32">float32 (High Quality)</MenuItem>
                                    <MenuItem value="float16">float16 (Balanced)</MenuItem>
                                    <MenuItem value="int8">int8 (Fast)</MenuItem>
                                </Select>
                            </FormControl>
                        </Box>
                    </ConfigSection>
                </Grid>

                {/* Processing */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <ConfigSection title="Processing" icon={<Videocam color="primary" fontSize="small" />}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            <TextField
                                size="small"
                                label="Frame Interval (sec)"
                                type="number"
                                inputProps={{ step: 0.1, min: 0 }}
                                value={formData.frame_interval ?? 1}
                                onChange={(e) => {
                                    const val = parseFloat(e.target.value);
                                    handleChange('frame_interval', isNaN(val) ? 0 : val);
                                }}
                                error={formData.frame_interval === 0}
                                helperText={
                                    formData.frame_interval === 0
                                        ? "⚠️ EXTREME LOAD: Extracting every frame (24-60 FPS)"
                                        : (formData.frame_interval || 1) < 0.5
                                            ? "High Load: >2 FPS"
                                            : "Recommended: 0.5s - 2.0s"
                                }
                                sx={{
                                    '& .MuiFormHelperText-root': {
                                        color: formData.frame_interval === 0 ? 'error.main' : 'text.secondary',
                                        fontWeight: formData.frame_interval === 0 ? 700 : 400
                                    }
                                }}
                            />
                            <TextField
                                size="small"
                                label="Frame Sample Ratio"
                                type="number"
                                value={formData.frame_sample_ratio || 2}
                                onChange={(e) => handleChange('frame_sample_ratio', parseInt(e.target.value) || 2)}
                                helperText="Process every Nth frame (1=all, 2=half, lower = more processing)"
                            />
                            <TextField
                                size="small"
                                label="Face Detection Threshold"
                                type="number"
                                inputProps={{ step: 0.1, min: 0.3, max: 0.9 }}
                                value={formData.face_detection_threshold || 0.5}
                                onChange={(e) => handleChange('face_detection_threshold', parseFloat(e.target.value) || 0.5)}
                                helperText="Confidence threshold (0.3-0.9, lower = more faces)"
                            />
                            <FormControl size="small" fullWidth>
                                <InputLabel>Face Detection Resolution</InputLabel>
                                <Select
                                    value={formData.face_detection_resolution || 640}
                                    label="Face Detection Resolution"
                                    onChange={(e) => handleChange('face_detection_resolution', e.target.value)}
                                >
                                    <MenuItem value={320}>320px (Fast)</MenuItem>
                                    <MenuItem value={640}>640px (Balanced)</MenuItem>
                                    <MenuItem value={960}>960px (Accurate)</MenuItem>
                                </Select>
                            </FormControl>
                            <TextField
                                size="small"
                                label="Language"
                                value={formData.language || ''}
                                onChange={(e) => handleChange('language', e.target.value)}
                                placeholder="auto, en, ta, etc."
                                helperText="Language for transcription (auto, en, ta, etc.)"
                            />
                        </Box>
                    </ConfigSection>
                </Grid>

                {/* Features */}
                <Grid size={{ xs: 12 }}>
                    <ConfigSection title="Features" icon={<SettingsIcon color="primary" fontSize="small" />}>
                        <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={formData.enable_voice_analysis ?? true}
                                        onChange={(e) => handleChange('enable_voice_analysis', e.target.checked)}
                                    />
                                }
                                label="Voice Analysis"
                            />
                            <FormControlLabel
                                control={
                                    <Switch
                                        checked={formData.enable_resource_monitoring ?? true}
                                        onChange={(e) => handleChange('enable_resource_monitoring', e.target.checked)}
                                    />
                                }
                                label="Resource Monitoring"
                            />
                        </Box>
                    </ConfigSection>
                </Grid>
            </Grid>
        </Box>
    );
}
