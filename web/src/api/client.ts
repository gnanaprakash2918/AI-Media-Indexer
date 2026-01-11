import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const apiClient = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000,
});

// Health & Status
export const healthCheck = async () => {
    const res = await apiClient.get('/health');
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
export const searchMedia = async (query: string, limit = 20): Promise<any[]> => {
    const res = await apiClient.get('/search', { params: { q: query, limit } });
    return res.data;
};

// Hybrid Search (100% accuracy with HITL identity integration)
export const searchHybrid = async (query: string, videoPath?: string, limit = 20) => {
    const res = await apiClient.get('/search/hybrid', {
        params: { q: query, video_path: videoPath, limit }
    });
    return res.data;
};

// Ingestion
export const ingestMedia = async (
    path: string,
    hint: string = 'unknown',
    startTime?: number,
    endTime?: number
) => {
    const res = await apiClient.post('/ingest', {
        path,
        media_type_hint: hint,
        start_time: startTime,
        end_time: endTime,
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
export const getJobs = async () => {
    const res = await apiClient.get('/jobs');
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

export const nameFaceCluster = async (clusterId: number, name: string) => {
    const res = await apiClient.post(`/faces/${clusterId}/name`, { name });
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
    const res = await apiClient.get('/voices', { params: { media_path: mediaPath, limit } });
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

export const mergeFaceClusters = async (sourceClusterId: number, targetClusterId: number) => {
    const res = await apiClient.post('/faces/merge', {
        source_cluster_id: sourceClusterId,
        target_cluster_id: targetClusterId,
    });
    return res.data;
};

export const setFaceMain = async (clusterId: number, isMain: boolean) => {
    const res = await apiClient.post(`/faces/${clusterId}/main`, null, { params: { is_main: isMain } });
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
    const res = await apiClient.post(`/voices/${clusterId}/name`, { name });
    return res.data;
};

export const renameVoiceSpeaker = async (segmentId: string, name: string) => {
    const res = await apiClient.put(`/voices/${segmentId}/name`, { name });
    return res.data;
};

export const mergeVoiceClusters = async (sourceClusterId: number, targetClusterId: number) => {
    const res = await apiClient.post('/voices/merge', {
        source_cluster_id: sourceClusterId,
        target_cluster_id: targetClusterId,
    });
    return res.data;
};

// Manual Cluster Management
export const moveFaceToCluster = async (faceId: string, clusterId: number) => {
    const res = await apiClient.put(`/faces/${faceId}/cluster`, null, { params: { cluster_id: clusterId } });
    return res.data;
};

export const createNewFaceCluster = async (faceIds: string[]) => {
    const res = await apiClient.post('/faces/new-cluster', faceIds);
    return res.data;
};

export const moveVoiceToCluster = async (segmentId: string, clusterId: number) => {
    const res = await apiClient.put(`/voices/${segmentId}/cluster`, null, { params: { cluster_id: clusterId } });
    return res.data;
};

export const createNewVoiceCluster = async (segmentIds: string[]) => {
    const res = await apiClient.post('/voices/new-cluster', segmentIds);
    return res.data;
};

// Name-Based Search
export const searchByName = async (name: string, limit = 20) => {
    const res = await apiClient.get('/search/by-name', { params: { name, limit } });
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
export const updateFrameDescription = async (frameId: string, description: string) => {
    const res = await apiClient.put(`/frames/${frameId}/description`, { description });
    return res.data;
};

// SSE Event Source helper
export function createEventSource(onMessage: (event: any) => void, onError?: () => void): EventSource {
    const es = new EventSource(`${API_BASE}/events`);

    es.onmessage = (event) => {
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

export interface Job {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
    progress: number;
    file_path: string;
    media_type: string;
    current_stage: string;
    message: string;
    started_at: number;
    completed_at: number | null;
    error: string | null;
    total_frames?: number;
    processed_frames?: number;
    timestamp?: number;
    duration?: number;
}

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
    return items.map((item: { path?: string; video_path?: string }) => item.path || item.video_path || '');
};

// Ingest with resume support
export const ingestWithResume = async (
    path: string,
    resumeJobId?: string,
    hint: string = 'unknown'
) => {
    const res = await apiClient.post('/ingest', {
        path,
        media_type_hint: hint,
        resume_job_id: resumeJobId,
    });
    return res.data;
};

// ========== HITL Power APIs: Link ==========

export const linkFaceVoice = async (faceClusterId: number, voiceClusterId: number, name?: string) => {
    const res = await apiClient.post('/identities/link', {
        face_cluster_id: faceClusterId,
        voice_cluster_id: voiceClusterId,
        name: name,
    });
    return res.data;
};

export const getIdentitySuggestions = async (videoPath?: string) => {
    const res = await apiClient.get('/identities/suggestions', {
        params: { video_path: videoPath }
    });
    return res.data;
};
