import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000,  // 5 minutes - search with LLM reranking can be slow
});

// Health & Status
export interface HealthResponse {
  device: string;
  qdrant: string;
  pipeline: string;
}

export const healthCheck = async (): Promise<HealthResponse> => {
  const res = await apiClient.get<HealthResponse>('/health');
  return res.data;
};

export const getStats = async () => {
  const res = await apiClient.get('/stats');
  return res.data;
};

export const getConfig = async () => {
  const res = await apiClient.get('/config');
  return res.data;
};

// Search
export const searchMedia = async (
  query: string,
  limit = 20,
): Promise<MediaItem[]> => {
  const res = await apiClient.get('/search', { params: { q: query, limit } });
  return res.data;
};

// Unified Search — all search flows route here with configurable toggles
export const searchHybrid = async (
  query: string,
  videoPath?: string,
  limit = 20,
  useReranking = false,
  useReasoning = false,
  useVlm = true,
  useDeepReasoning = false,
) => {
  const res = await apiClient.get('/search/unified', {
    params: {
      q: query,
      video_path: videoPath,
      limit,
      enable_expansion: useReasoning,
      enable_reranking: useReranking,
      enable_vlm: useVlm,
      enable_deep_reasoning: useDeepReasoning,
    },
  });
  return res.data;
};

// Ingestion
export const ingestMedia = async (
  path: string,
  hint: string = 'unknown',
  startTime?: number,
  endTime?: number,
  enableChunking = true,
) => {
  const res = await apiClient.post('/ingest', {
    path,
    media_type_hint: hint,
    start_time: startTime,
    end_time: endTime,
    enable_chunking: enableChunking
  });
  return res.data;
};

export const scanDirectory = async (directory: string, recursive = true) => {
  const res = await apiClient.post('/scan', { directory, recursive });
  return res.data;
};

// System
export const browseFileSystem = async (): Promise<string | null> => {
  const res = await apiClient.get('/system/browse');
  return res.data.path;
};

// Jobs
export interface StageStats {
  status: 'pending' | 'running' | 'completed' | 'skipped' | 'failed';
  start?: number;
  end?: number;
  duration?: number;
  items_processed?: number;
  items_total?: number;
  error?: string;
  retries?: number;
}

export interface JobSpeed {
  fps: number;
  speed_ratio: number;
}

export interface Job {
  job_id: string;
  status: string;
  progress: number;
  weighted_progress: number;
  file_path: string;
  media_type: string;
  current_stage: string;
  pipeline_stage: string;
  message: string;
  started_at: number;
  completed_at: number | null;
  error: string | null;
  total_frames: number;
  processed_frames: number;
  current_item_index: number;
  total_items: number;
  timestamp: number;
  duration: number;
  last_heartbeat: number;
  stage_stats: Record<string, StageStats>;
  eta_seconds: number | null;
  speed: JobSpeed;
  checkpoint_data?: Record<string, unknown>;
}

export interface JobsResponse {
  jobs: Job[];
}

export const getJobs = async (): Promise<JobsResponse> => {
  const res = await apiClient.get<JobsResponse>('/jobs');
  return res.data;
};

export const getJob = async (jobId: string) => {
  const res = await apiClient.get(`/jobs/${jobId}`);
  return res.data;
};

export const cancelJob = async (jobId: string) => {
  const res = await apiClient.post(`/jobs/${jobId}/cancel`);
  return res.data;
};

export const pauseJob = async (jobId: string) => {
  const res = await apiClient.post(`/jobs/${jobId}/pause`);
  return res.data;
};

export const resumeJob = async (jobId: string) => {
  const res = await apiClient.post(`/jobs/${jobId}/resume`);
  return res.data;
};

export const deleteJob = async (jobId: string) => {
  const res = await apiClient.delete(`/jobs/${jobId}`);
  return res.data;
};

// Faces (HITL)
export const getUnresolvedFaces = async (limit = 50) => {
  const res = await apiClient.get('/faces/unresolved', { params: { limit } });
  return res.data;
};

// ... (skipping unchanged parts)

export const getNamedFaces = async () => {
  const res = await apiClient.get('/faces/named');
  return res.data;
};

export const getAllNames = async () => {
  const res = await apiClient.get<{ names: string[] }>('/identities/names');
  return res.data.names || [];
};

export const nameFaceCluster = async (clusterId: number, name: string) => {
  const res = await apiClient.post(`/faces/cluster/${clusterId}/name`, { name });
  return res.data;
};

export const nameSingleFace = async (faceId: string, name: string) => {
  const res = await apiClient.put(`/faces/${faceId}/name`, { name });
  return res.data;
};

export const deleteFace = async (faceId: string) => {
  const res = await apiClient.delete(`/faces/${faceId}`);
  return res.data;
};

// Voice Segments
export const getVoiceSegments = async (mediaPath?: string, limit = 100) => {
  const res = await apiClient.get('/voices', {
    params: { media_path: mediaPath, limit },
  });
  return res.data;
};

export const deleteVoiceSegment = async (segmentId: string) => {
  const res = await apiClient.delete(`/voices/${segmentId}`);
  return res.data;
};

// Face Clustering
export const triggerFaceClustering = async () => {
  const res = await apiClient.post('/faces/cluster');
  return res.data;
};

export const getFaceClusters = async () => {
  const res = await apiClient.get('/faces/clusters');
  return res.data;
};

export const mergeFaceClusters = async (
  sourceClusterId: number,
  targetClusterId: number,
) => {
  const res = await apiClient.post('/faces/merge', {
    source_cluster_id: sourceClusterId,
    target_cluster_id: targetClusterId,
  });
  return res.data;
};

export const setFaceMain = async (clusterId: number, isMain: boolean) => {
  const res = await apiClient.post(`/faces/cluster/${clusterId}/main`, null, {
    params: { is_main: isMain },
  });
  return res.data;
};

// Voice Clustering
export const triggerVoiceClustering = async () => {
  const res = await apiClient.post('/voices/cluster');
  return res.data;
};

export const getVoiceClusters = async () => {
  const res = await apiClient.get('/voices/clusters');
  return res.data;
};

export const nameVoiceCluster = async (clusterId: number, name: string) => {
  const res = await apiClient.post(`/voices/cluster/${clusterId}/name`, { name });
  return res.data;
};

export const renameVoiceSpeaker = async (segmentId: string, name: string) => {
  const res = await apiClient.put(`/voices/${segmentId}/name`, { name });
  return res.data;
};

export const mergeVoiceClusters = async (
  sourceClusterId: number,
  targetClusterId: number,
) => {
  const res = await apiClient.post('/voices/merge', {
    source_cluster_id: sourceClusterId,
    target_cluster_id: targetClusterId,
  });
  return res.data;
};

// Manual Cluster Management
export const moveFaceToCluster = async (faceId: string, clusterId: number) => {
  const res = await apiClient.put(`/faces/${faceId}/cluster`, null, {
    params: { cluster_id: clusterId },
  });
  return res.data;
};

export const createNewFaceCluster = async (faceIds: string[]) => {
  const res = await apiClient.post('/faces/new-cluster', faceIds);
  return res.data;
};

export const moveVoiceToCluster = async (
  segmentId: string,
  clusterId: number,
) => {
  const res = await apiClient.put(`/voices/${segmentId}/cluster`, null, {
    params: { cluster_id: clusterId },
  });
  return res.data;
};

export const createNewVoiceCluster = async (segmentIds: string[]) => {
  const res = await apiClient.post('/voices/new-cluster', segmentIds);
  return res.data;
};

// Delete entire clusters
export const deleteVoiceCluster = async (clusterId: number) => {
  const res = await apiClient.delete(`/voices/cluster/${clusterId}`);
  return res.data;
};

export const deleteFaceCluster = async (clusterId: number) => {
  const res = await apiClient.delete(`/faces/cluster/${clusterId}`);
  return res.data;
};

// Name-Based Search
export const searchByName = async (name: string, limit = 20) => {
  const res = await apiClient.get('/search/by-name', { params: { name, limit } });
  return res.data;
};

// Granular / Agentic Search — unified endpoint with expansion + reranking
export const searchGranular = async (
  query: string,
  videoPath?: string,
  limit = 10,
  enableRerank = true,
) => {
  const res = await apiClient.get('/search/unified', {
    params: {
      q: query,
      video_path: videoPath,
      limit,
      enable_expansion: true,
      enable_reranking: enableRerank,
      enable_vlm: enableRerank,
      enable_deep_reasoning: true,
    },
  });
  return res.data;
};

// Grounding (SAM)
export const triggerGrounding = async (videoPath: string, concepts?: string[]) => {
  const res = await apiClient.post('/api/grounding/trigger', {
    video_path: videoPath,
    concepts,
  });
  return res.data;
};

export const updateMasklet = async (maskletId: string, updates: Record<string, unknown>) => {
  const res = await apiClient.patch(`/api/masklets/${maskletId}`, updates);
  return res.data;
};

export const getMasklets = async (videoPath: string, startTime?: number, endTime?: number) => {
  const res = await apiClient.get('/api/media/masklets', {
    params: { video_path: videoPath, start_time: startTime, end_time: endTime }
  });
  return res.data;
};

export const getVideoSummary = async (videoPath: string) => {
  const res = await apiClient.get('/api/media/summary', {
    params: { path: videoPath }
  });
  return res.data;
};

// Overlays for video visualization
export interface OverlayItem {
  timestamp: number;
  bbox: number[];
  label?: string;
  text?: string;
  color: string;
  confidence?: number;
}

export interface VideoOverlays {
  video_id: string;
  faces: OverlayItem[];
  text_regions: OverlayItem[];
  objects: OverlayItem[];
  active_speakers: OverlayItem[];
  clothing?: OverlayItem[];
  loudness_events?: Array<{
    timestamp: number;
    spl_db: number;
    lufs: number;
    category: string;
  }>;
  // Voice diarization segments for timeline visualization
  voice_diarization?: Array<{
    start_time: number;
    end_time: number;
    speaker_label: string;
    speaker_name?: string | null;
    voice_cluster_id: number;
    color: string;
  }>;
}


export const getOverlays = async (
  videoPath: string,
  startTime?: number,
  endTime?: number,
): Promise<VideoOverlays> => {
  const videoId = encodeURIComponent(videoPath);
  const res = await apiClient.get(`/overlays/${videoId}`, {
    params: { start_time: startTime, end_time: endTime }
  });
  return res.data;
};

// Library
export const getLibrary = async () => {
  const res = await apiClient.get('/library');
  return res.data;
};

export const deleteLibraryItem = async (path: string) => {
  const res = await apiClient.delete('/library', { params: { path } });
  return res.data;
};

// Alias for backward compatibility if needed
export const deleteFromLibrary = deleteLibraryItem;

// HITL: Manual Description Correction
export const updateFrameDescription = async (
  frameId: string,
  description: string,
) => {
  const res = await apiClient.put(`/frames/${frameId}/description`, {
    description,
  });
  return res.data;
};

// SSE Event Source helper
export function createEventSource(
  onMessage: (event: unknown) => void,
  onError?: () => void,
): EventSource {
  const es = new EventSource(`${API_BASE}/events`);

  es.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      // Heartbeat or invalid JSON, ignore
    }
  };

  es.onerror = () => {
    es.close();
    onError?.();
  };

  return es;
}

// Types
export interface MediaItem {
  id: string;
  filename: string;
  video_path: string;
  metadata: {
    title?: string;
    duration?: number;
    year?: number;
  };
  score?: number;
  thumbnail_url?: string;
}

// Job interface is defined at the top of the file with full stats

export interface FaceCluster {
  id: string;
  cluster_id: number;
  name: string | null;
  media_path?: string;
  timestamp?: number;
  thumbnail_path?: string;
}

// NEW: Enhanced Search Result with RRF explainability
export interface SearchResult {
  id: string;
  video_path: string;
  timestamp: number;
  score: number;
  rrf_score?: number;
  vector_score?: number;
  keyword_score?: number;
  action?: string;
  dialogue?: string;
  match_reasons: string[];
  matched_identity?: string;
  thumbnail_path?: string;
  face_cluster_ids?: number[];
  face_names?: string[];
}

// Get list of indexed videos for filter dropdown
export const getIndexedVideos = async (): Promise<string[]> => {
  const res = await apiClient.get('/library');
  const items = res.data.media || res.data || [];
  return items.map(
    (item: { path?: string; video_path?: string }) =>
      item.path || item.video_path || '',
  );
};

// Ingest with resume support
export const ingestWithResume = async (
  path: string,
  resumeJobId?: string,
  hint: string = 'unknown',
) => {
  const res = await apiClient.post('/ingest', {
    path,
    media_type_hint: hint,
    resume_job_id: resumeJobId,
  });
  return res.data;
};

// ========== HITL Power APIs: Link ==========

export const linkFaceVoice = async (
  faceClusterId: number,
  voiceClusterId: number,
  name?: string,
) => {
  const res = await apiClient.post('/identities/link', {
    face_cluster_id: faceClusterId,
    voice_cluster_id: voiceClusterId,
    name: name,
  });
  return res.data;
};

export const getIdentitySuggestions = async (videoPath?: string) => {
  const res = await apiClient.get('/identities/suggestions', {
    params: { video_path: videoPath },
  });
  return res.data;
};

// ========== HITL Search Feedback APIs ==========

export interface SearchFeedback {
  query: string;
  result_id: string;
  video_path: string;
  timestamp: number;
  is_relevant: boolean;
  feedback_type?: 'binary' | 'rating' | 'correction';
  rating?: number;
  correction?: string;
  notes?: string;
}

export const submitSearchFeedback = async (feedback: SearchFeedback) => {
  const res = await apiClient.post('/search/feedback', feedback);
  return res.data;
};

export const getSearchFeedbackStats = async () => {
  const res = await apiClient.get('/search/feedback/stats');
  return res.data;
};

// ========== Voice Overlay APIs ==========

export interface VoiceOverlay {
  type: string;
  start_time: number;
  end_time: number;
  duration: number;
  speaker_label: string;
  speaker_name: string | null;
  cluster_id: number;
  has_audio: boolean;
  audio_url: string | null;
  color: string;
}

export interface VoiceOverlayResponse {
  video_id: string;
  timestamp: number | null;
  voice_segments: VoiceOverlay[];
  total_segments: number;
  active_speaker: {
    speaker_label: string;
    speaker_name: string;
    start_time: number;
    end_time: number;
  } | null;
}

export const getVoiceOverlays = async (
  videoPath: string,
  timestamp?: number,
): Promise<VoiceOverlayResponse> => {
  const res = await apiClient.get('/overlays/voice', {
    params: { video_id: videoPath, timestamp },
  });
  return res.data;
};

// ========== Explainable Search — unified endpoint with deep reasoning ==========

export const searchExplainable = async (query: string, limit = 10) => {
  const res = await apiClient.get('/search/unified', {
    params: {
      q: query,
      limit,
      enable_expansion: true,
      enable_reranking: true,
      enable_vlm: true,
      enable_deep_reasoning: true,
    },
  });
  return res.data;
};

// ========== Manipulation (Inpainting/Redaction) ==========

export interface RegionRequest {
  video_path: string;
  start_time: number;
  end_time: number;
  bbox: number[]; // [x, y, w, h]
}

export interface ManipulationJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  result_path?: string;
  error?: string;
}

export const triggerInpaint = async (request: RegionRequest) => {
  const res = await apiClient.post<ManipulationJob>('/manipulation/inpaint', request);
  return res.data;
};

export const triggerRedact = async (request: RegionRequest) => {
  const res = await apiClient.post<ManipulationJob>('/manipulation/redact', request);
  return res.data;
};

export const getManipulationJob = async (jobId: string) => {
  const res = await apiClient.get<ManipulationJob>(`/manipulation/jobs/${jobId}`);
  return res.data;
};

// ========== Councils API ==========

export interface CouncilConfig {
  mode: 'oss_only' | 'commercial_only' | 'combined';
  councils: Record<string, Council>;
}

export interface Council {
  models: ModelSpec[];
  enabled: boolean;
}

export interface ModelSpec {
  name: string;
  model_type: 'oss' | 'commercial';
  model_id: string;
  enabled: boolean;
  weight: number;
  vram_gb: number;
  description: string;
}

export const getCouncilsConfig = async () => {
  const res = await apiClient.get<CouncilConfig>('/councils');
  return res.data;
};

export const setCouncilMode = async (mode: string) => {
  const res = await apiClient.put('/councils/mode', { mode });
  return res.data;
};

export const updateCouncilModel = async (
  councilName: string,
  modelName: string,
  update: { enabled?: boolean; weight?: number }
) => {
  const res = await apiClient.patch(`/councils/${councilName}/models/${modelName}`, update);
  return res.data;
};

// ========== GraphRAG & Timeline API ==========

export interface CoOccurrence {
  name: string;
  cluster_id: number;
  count: number;
  relationship_strength: number;
}

export interface SocialGraphResponse {
  center_person: string;
  center_cluster_id: number;
  connections: CoOccurrence[];
}

export interface SceneNode {
  type: 'scene' | 'action';
  id: string;
  timestamp: number;
  description?: string;
  characters?: string[];
  thumbnail?: string;
}

export const getSocialGraph = async (name?: string, clusterId?: number) => {
  const res = await apiClient.get<SocialGraphResponse>('/graph/social', {
    params: { name, cluster_id: clusterId },
  });
  return res.data;
};

export const getSceneTimeline = async (videoPath: string) => {
  // Use encodeURIComponent to handle paths safely
  const res = await apiClient.get<{ timeline: SceneNode[] }>(`/graph/timeline/${encodeURIComponent(videoPath)}`);
  return res.data;
};

export const getGraphStats = async () => {
  const res = await apiClient.get<{ stats: any }>('/graph/stats');
  return res.data;
};

export const identifyFaceCluster = async (clusterId: number) => {
  const res = await apiClient.post(`/faces/cluster/${clusterId}/identify`);
  return res.data;
};
