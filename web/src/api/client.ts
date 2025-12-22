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

// Ingestion
export const ingestMedia = async (path: string, hint: string = 'unknown') => {
    const res = await apiClient.post('/ingest', { path, media_type_hint: hint });
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

// Faces (HITL)
export const getUnresolvedFaces = async (limit = 50) => {
    const res = await apiClient.get('/faces/unresolved', { params: { limit } });
    return res.data;
};

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

// Library
export const getLibrary = async () => {
    const res = await apiClient.get('/library');
    return res.data;
};

export const deleteFromLibrary = async (path: string) => {
    const res = await apiClient.delete(`/library/${encodeURIComponent(path)}`);
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
    status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
    progress: number;
    file_path: string;
    media_type: string;
    current_stage: string;
    message: string;
    started_at: number;
    completed_at: number | null;
    error: string | null;
}

export interface FaceCluster {
    id: string;
    cluster_id: number;
    name: string | null;
    media_path?: string;
    timestamp?: number;
    thumbnail_path?: string;
}
