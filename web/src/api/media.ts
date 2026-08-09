/**
 * Media API — library, overlays, and grounding endpoints.
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
