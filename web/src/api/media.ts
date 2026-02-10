/**
 * Media API — library, overlays, grounding, manipulation, and councils.
 */
import { apiClient } from './client';

// ========== Media Types ==========

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
    voice_diarization?: Array<{
        start_time: number;
        end_time: number;
        speaker_label: string;
        speaker_name?: string | null;
        voice_cluster_id: number;
        color: string;
    }>;
}

export interface RegionRequest {
    video_path: string;
    start_time: number;
    end_time: number;
    bbox: number[];
}

export interface ManipulationJob {
    job_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    result_path?: string;
    error?: string;
}

// ========== Councils Types ==========

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

// ========== Graph Types ==========

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

// ========== Library Endpoints ==========

export const getLibrary = async () => {
    const res = await apiClient.get('/library');
    return res.data;
};

export const deleteLibraryItem = async (path: string) => {
    const res = await apiClient.delete('/library', { params: { path } });
    return res.data;
};

export const getIndexedVideos = async (): Promise<string[]> => {
    const res = await apiClient.get('/library');
    const items = res.data.media || res.data || [];
    return items.map(
        (item: { path?: string; video_path?: string }) =>
            item.path || item.video_path || '',
    );
};

// ========== Overlays ==========

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

// ========== Grounding (SAM) ==========

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

// ========== HITL: Frame Description ==========

export const updateFrameDescription = async (
    frameId: string,
    description: string,
) => {
    const res = await apiClient.put(`/frames/${frameId}/description`, {
        description,
    });
    return res.data;
};

// ========== Manipulation ==========

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

// ========== Councils ==========

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

// ========== Graph ==========

export const getSocialGraph = async (name?: string, clusterId?: number) => {
    const res = await apiClient.get<SocialGraphResponse>('/graph/social', {
        params: { name, cluster_id: clusterId },
    });
    return res.data;
};

export const getSceneTimeline = async (videoPath: string) => {
    const res = await apiClient.get<{ timeline: SceneNode[] }>(`/graph/timeline/${encodeURIComponent(videoPath)}`);
    return res.data;
};

export const getGraphStats = async () => {
    const res = await apiClient.get<{ stats: unknown }>('/graph/stats');
    return res.data;
};
