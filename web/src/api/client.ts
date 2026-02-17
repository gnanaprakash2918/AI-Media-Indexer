/**
 * API Client — core Axios instance, health/status, jobs, ingestion, SSE.
 *
 * Domain-specific endpoints are in separate modules:
 *   - ./search.ts  — search, feedback
 *   - ./faces.ts   — face detection, clustering, identity
 *   - ./voices.ts  — voice segments, clustering, overlays
 *   - ./media.ts   — library, overlays, grounding, manipulation, councils, graph
 *
 * All exports are re-exported here for backward compatibility.
 */
import axios from 'axios';

export const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 300000,  // 5 minutes - search with LLM reranking can be slow
});

// Standardized API error handling — all callers get consistent error structure
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status ?? 0;
    const detail = error.response?.data?.detail ?? error.message ?? 'Unknown error';
    const message =
      status === 401 ? 'Authentication required' :
        status === 404 ? `Not found: ${error.config?.url}` :
          status >= 500 ? `Server error (${status}): ${detail}` :
            status > 0 ? `Request failed (${status}): ${detail}` :
              `Network error: ${detail}`;

    const apiError = Object.assign(new Error(message), { status, detail });
    return Promise.reject(apiError);
  },
);

// ========== Health & Status ==========

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

// ========== Ingestion ==========

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

// ========== System ==========

export const browseFileSystem = async (): Promise<string | null> => {
  const res = await apiClient.get('/system/browse');
  return res.data.path;
};

// ========== Jobs ==========

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

// ========== SSE ==========

export function createEventSource(
  onMessage: (event: unknown) => void,
  onError?: () => void,
): EventSource {
  const url = `${API_BASE}/events`;
  let retryCount = 0;
  const maxRetries = 5;

  const es = new EventSource(url);

  const setupHandlers = (source: EventSource) => {
    source.onmessage = event => {
      retryCount = 0; // Reset on successful message
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch {
        // Heartbeat or invalid JSON, ignore
      }
    };

    source.onerror = () => {
      source.close();
      if (retryCount < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
        retryCount++;
        setTimeout(() => {
          const newSource = new EventSource(url);
          setupHandlers(newSource);
        }, delay);
      } else {
        onError?.();
      }
    };
  };

  setupHandlers(es);
  return es;
}

// ========== Re-exports for backward compatibility ==========
// Domain-specific modules are the source of truth; these re-exports
// ensure existing `import { X } from '../api/client'` statements keep working.

export {
  searchHybrid, searchByName, searchGranular, searchExplainable,
  submitSearchFeedback, getSearchFeedbackStats,
} from './search';
export type { SearchResult, SearchFeedback } from './search';

export {
  getUnresolvedFaces, getNamedFaces, getAllNames,
  nameFaceCluster, nameSingleFace, deleteFace,
  triggerFaceClustering, getFaceClusters, mergeFaceClusters,
  setFaceMain, moveFaceToCluster, createNewFaceCluster,
  deleteFaceCluster, identifyFaceCluster,
  linkFaceVoice, getIdentitySuggestions,
} from './faces';
export type { FaceCluster } from './faces';

export {
  getVoiceSegments, deleteVoiceSegment,
  triggerVoiceClustering, getVoiceClusters,
  nameVoiceCluster, renameVoiceSpeaker, mergeVoiceClusters,
  moveVoiceToCluster, createNewVoiceCluster, deleteVoiceCluster,
  getVoiceOverlays,
} from './voices';
export type { VoiceOverlay, VoiceOverlayResponse } from './voices';

export {
  getLibrary, deleteLibraryItem, getIndexedVideos,
  getOverlays, triggerGrounding, updateMasklet, getMasklets, getVideoSummary,
  updateFrameDescription,
  triggerInpaint, triggerRedact, getManipulationJob,
  getCouncilsConfig, setCouncilMode, updateCouncilModel,
  getSocialGraph, getSceneTimeline, getGraphStats,
} from './media';
export type {
  MediaItem, OverlayItem, VideoOverlays,
  RegionRequest, ManipulationJob,
  CouncilConfig, Council, ModelSpec,
  CoOccurrence, SocialGraphResponse, SceneNode,
} from './media';
