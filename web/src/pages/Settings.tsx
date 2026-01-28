import { useState } from 'react';
import { useQuery, useMutation, useQueryClient, type UseQueryResult } from '@tanstack/react-query';
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
  Slider,
  LinearProgress,
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
  Speed,
  Bolt,
} from '@mui/icons-material';

import { getConfig, healthCheck, apiClient, type HealthResponse } from '../api/client';

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
  enable_vlm_reranking?: boolean;
  enable_hybrid_search?: boolean;
  enable_frame_vlm?: boolean;
  enable_video_embeddings?: boolean;
  enable_hybrid_vlm?: boolean;
  max_cpu_percent: number;
  max_ram_percent: number;
  google_api_key?: string;
  ollama_base_url?: string;
  ollama_model?: string;
  hf_token?: string;
}

interface SystemProfile {
  vram_gb: number;
  ram_gb: number;
  tier: 'low' | 'medium' | 'high';
  embedding_model: string;
  embedding_dim: number;
  vision_model: string;
  batch_size: number;
  max_concurrent_jobs: number;
  frame_batch_size: number;
  lazy_unload: boolean;
  aggressive_cleanup: boolean;
}

interface SystemConfigResponse {
  profile: SystemProfile;
  vram_used_gb: number;
  vram_free_gb: number;
  settings: {
    embedding_model: string;
    vision_model: string;
    batch_size: number;
    max_concurrent_jobs: number;
    lazy_unload: boolean;
    high_performance_mode: boolean;
  };
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

interface SettingsContentProps {
  initialConfig: ConfigData;
  initialSystemConfig: SystemConfigResponse;
  health: UseQueryResult<HealthResponse>;
}

function SettingsContent({
  initialConfig,
  initialSystemConfig,
  health,
}: SettingsContentProps) {
  const queryClient = useQueryClient();
  const [editMode, setEditMode] = useState(false);
  const [formData, setFormData] = useState<Partial<ConfigData>>(initialConfig);
  const [saved, setSaved] = useState(false);

  // System config for Hardware & Performance
  const [perfSettings, setPerfSettings] = useState({
    batch_size: initialSystemConfig.settings.batch_size,
    max_concurrent_jobs: initialSystemConfig.settings.max_concurrent_jobs,
    lazy_unload: initialSystemConfig.settings.lazy_unload,
    high_performance_mode: initialSystemConfig.settings.high_performance_mode,
  });

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

  const saveSystemMutation = useMutation({
    mutationFn: async (data: typeof perfSettings) => {
      const res = await apiClient.post('/config/system', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['systemConfig'] });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    },
  });

  const handleChange = (key: keyof ConfigData, value: unknown) => {
    setFormData(prev => ({ ...prev, [key]: value }));
    setEditMode(true);
  };

  const handlePerfChange = (key: keyof typeof perfSettings, value: unknown) => {
    setPerfSettings(prev => ({ ...prev, [key]: value }));
    setEditMode(true);
  };

  const handleSave = () => {
    saveMutation.mutate(formData);
    setEditMode(false);
  };

  const handleSavePerf = () => {
    saveSystemMutation.mutate(perfSettings);
    setEditMode(false);
  };

  const vramPercent =
    (initialSystemConfig.vram_used_gb /
      (initialSystemConfig.vram_used_gb + initialSystemConfig.vram_free_gb)) *
    100;

  return (
    <Box>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Box>
          <Typography variant="h5" fontWeight={700}>
            Settings
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Configure API keys, hardware, and runtime settings
          </Typography>
        </Box>
        {editMode && (
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="contained"
              startIcon={<Save />}
              onClick={handleSave}
              disabled={saveMutation.isPending}
            >
              Save Config
            </Button>
            <Button
              variant="outlined"
              startIcon={<Speed />}
              onClick={handleSavePerf}
              disabled={saveSystemMutation.isPending}
            >
              Save Performance
            </Button>
          </Box>
        )}
      </Box>

      {saved && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Settings saved. Some changes require a backend restart.
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* Hardware & Performance (NEW) */}
        <Grid size={{ xs: 12 }}>
          <ConfigSection
            title="Hardware & Performance"
            icon={<Bolt color="warning" fontSize="small" />}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {/* VRAM Status */}
              <Box>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mb: 0.5,
                  }}
                >
                  <Typography variant="body2" fontWeight={600}>
                    VRAM: {initialSystemConfig.vram_used_gb.toFixed(1)} /{' '}
                    {(
                      initialSystemConfig.vram_used_gb +
                      initialSystemConfig.vram_free_gb
                    ).toFixed(1)}{' '}
                    GB
                  </Typography>
                  <Chip
                    label={initialSystemConfig.profile.tier.toUpperCase()}
                    size="small"
                    color={
                      initialSystemConfig.profile.tier === 'high'
                        ? 'success'
                        : initialSystemConfig.profile.tier === 'medium'
                          ? 'warning'
                          : 'error'
                    }
                  />
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={vramPercent}
                  color={
                    vramPercent > 80
                      ? 'error'
                      : vramPercent > 50
                        ? 'warning'
                        : 'primary'
                  }
                  sx={{ height: 8, borderRadius: 1 }}
                />
              </Box>

              <Grid container spacing={2}>
                {/* Embedding Model (Read-only) */}
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Embedding Model (SOTA - Locked)"
                    value={initialSystemConfig.profile.embedding_model}
                    disabled
                    helperText={`${initialSystemConfig.profile.embedding_dim}d vectors - Never downgraded`}
                    InputProps={{
                      sx: { bgcolor: 'action.disabledBackground' },
                    }}
                  />
                </Grid>

                {/* Vision Model (Read-only) */}
                <Grid size={{ xs: 12, md: 6 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Vision Model"
                    value={initialSystemConfig.profile.vision_model}
                    disabled
                    helperText="Auto-selected based on VRAM"
                    InputProps={{
                      sx: { bgcolor: 'action.disabledBackground' },
                    }}
                  />
                </Grid>

                {/* Batch Size (Editable) */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Batch Size"
                    type="number"
                    inputProps={{ min: 1, max: 64 }}
                    value={perfSettings.batch_size}
                    onChange={e =>
                      handlePerfChange(
                        'batch_size',
                        parseInt(e.target.value) || 1,
                      )
                    }
                    helperText="Higher = faster, more VRAM"
                  />
                </Grid>

                {/* Max Concurrent Jobs (Slider) */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <Typography variant="body2" gutterBottom>
                    Max Concurrent Jobs: {perfSettings.max_concurrent_jobs}
                  </Typography>
                  <Slider
                    value={perfSettings.max_concurrent_jobs}
                    onChange={(_, val) =>
                      handlePerfChange('max_concurrent_jobs', val as number)
                    }
                    min={1}
                    max={4}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>

                {/* Lazy Unload (Toggle) */}
                <Grid size={{ xs: 12, md: 4 }}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={perfSettings.lazy_unload}
                        onChange={e =>
                          handlePerfChange('lazy_unload', e.target.checked)
                        }
                      />
                    }
                    label="Lazy Unload Models"
                  />
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    display="block"
                  >
                    Free VRAM after each operation
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          </ConfigSection>
        </Grid>

        {/* Connection Status */}
        <Grid size={{ xs: 12, md: 6 }}>
          <ConfigSection
            title="Status"
            icon={<CloudQueue color="primary" fontSize="small" />}
          >
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
                <ListItemText
                  primary="Qdrant"
                  secondary={`${formData.qdrant_host}:${formData.qdrant_port}`}
                />
                <Chip
                  label={
                    health.data?.qdrant === 'connected'
                      ? 'Connected'
                      : 'Offline'
                  }
                  color={
                    health.data?.qdrant === 'connected' ? 'success' : 'warning'
                  }
                  size="small"
                />
              </ListItem>
            </List>
          </ConfigSection>
        </Grid>

        {/* API Keys */}
        <Grid size={{ xs: 12, md: 6 }}>
          <ConfigSection
            title="API Keys"
            icon={<Key color="primary" fontSize="small" />}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <FormControl size="small" fullWidth>
                <InputLabel>LLM Provider</InputLabel>
                <Select
                  value={formData.llm_provider || 'ollama'}
                  label="LLM Provider"
                  onChange={e => handleChange('llm_provider', e.target.value)}
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
                  onChange={e => handleChange('google_api_key', e.target.value)}
                  placeholder="AIza..."
                />
              )}

              {formData.llm_provider === 'ollama' && (
                <>
                  <TextField
                    size="small"
                    label="Ollama URL"
                    value={formData.ollama_base_url || 'http://localhost:11434'}
                    onChange={e =>
                      handleChange('ollama_base_url', e.target.value)
                    }
                  />
                  <TextField
                    size="small"
                    label="Ollama Model"
                    value={formData.ollama_model || 'llava:7b'}
                    onChange={e => handleChange('ollama_model', e.target.value)}
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
                onChange={e => handleChange('hf_token', e.target.value)}
                placeholder="hf_..."
              />
            </Box>
          </ConfigSection>
        </Grid>

        {/* Hardware */}
        <Grid size={{ xs: 12, md: 6 }}>
          <ConfigSection
            title="Device"
            icon={<Memory color="primary" fontSize="small" />}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <FormControl size="small" fullWidth>
                <InputLabel>Device</InputLabel>
                <Select
                  value={formData.device || 'cuda'}
                  label="Device"
                  onChange={e => handleChange('device', e.target.value)}
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
                  onChange={e => handleChange('compute_type', e.target.value)}
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
          <ConfigSection
            title="Processing"
            icon={<Videocam color="primary" fontSize="small" />}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
              <TextField
                size="small"
                label="Frame Interval (sec)"
                type="number"
                inputProps={{ step: 0.1, min: 0 }}
                value={formData.frame_interval ?? 1}
                onChange={e =>
                  handleChange(
                    'frame_interval',
                    parseFloat(e.target.value) || 0,
                  )
                }
                helperText="Recommended: 0.5s - 2.0s"
              />
              <TextField
                size="small"
                label="Face Detection Threshold"
                type="number"
                inputProps={{ step: 0.1, min: 0.3, max: 0.9 }}
                value={formData.face_detection_threshold || 0.5}
                onChange={e =>
                  handleChange(
                    'face_detection_threshold',
                    parseFloat(e.target.value) || 0.5,
                  )
                }
                helperText="0.3-0.9, lower = more faces"
              />
              <TextField
                size="small"
                label="Language"
                value={formData.language || ''}
                onChange={e => handleChange('language', e.target.value)}
                placeholder="auto, en, ta, etc."
                helperText="Language for transcription"
              />
            </Box>
          </ConfigSection>
        </Grid>

        {/* Features */}
        <Grid size={{ xs: 12 }}>
          <ConfigSection
            title="Features"
            icon={<SettingsIcon color="primary" fontSize="small" />}
          >
            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_voice_analysis ?? true}
                    onChange={e =>
                      handleChange('enable_voice_analysis', e.target.checked)
                    }
                  />
                }
                label="Voice Analysis"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_resource_monitoring ?? true}
                    onChange={e =>
                      handleChange(
                        'enable_resource_monitoring',
                        e.target.checked,
                      )
                    }
                  />
                }
                label="Resource Monitoring"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_vlm_reranking ?? true}
                    onChange={e =>
                      handleChange('enable_vlm_reranking', e.target.checked)
                    }
                  />
                }
                label="VLM Reranking (may use more VRAM)"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_hybrid_search ?? true}
                    onChange={e =>
                      handleChange('enable_hybrid_search', e.target.checked)
                    }
                  />
                }
                label="Hybrid Search"
              />
            </Box>
            <Divider sx={{ my: 2 }} />
            <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
              Video Understanding (Hybrid VLM)
            </Typography>
            <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_frame_vlm ?? true}
                    onChange={e =>
                      handleChange('enable_frame_vlm', e.target.checked)
                    }
                  />
                }
                label="Frame VLM (faces, text, objects)"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_video_embeddings ?? true}
                    onChange={e =>
                      handleChange('enable_video_embeddings', e.target.checked)
                    }
                  />
                }
                label="Video Embeddings (actions, motion)"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.enable_hybrid_vlm ?? true}
                    onChange={e =>
                      handleChange('enable_hybrid_vlm', e.target.checked)
                    }
                  />
                }
                label="Hybrid VLM (combine both - recommended)"
              />
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Frame VLM: Fine-grained per-frame details. Video Embeddings: Action/motion understanding (InternVideo, LanguageBind).
            </Typography>
          </ConfigSection>
        </Grid>
      </Grid>
    </Box>
  );
}

export default function SettingsPage() {
  const config = useQuery({
    queryKey: ['config'],
    queryFn: getConfig,
  });

  const health = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
  });

  const systemConfigQuery = useQuery({
    queryKey: ['systemConfig'],
    queryFn: async () => {
      const res = await apiClient.get<SystemConfigResponse>('/config/system');
      return res.data;
    },
  });

  if (config.isLoading || systemConfigQuery.isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress size={32} />
      </Box>
    );
  }

  if (
    config.isError ||
    systemConfigQuery.isError ||
    !config.data ||
    !systemConfigQuery.data
  ) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        Failed to load configuration. Make sure the backend is running.
      </Alert>
    );
  }

  return (
    <SettingsContent
      initialConfig={config.data}
      initialSystemConfig={systemConfigQuery.data}
      health={health}
    />
  );
}
