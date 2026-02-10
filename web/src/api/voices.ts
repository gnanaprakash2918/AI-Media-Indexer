/**
 * Voices API — voice segments, clustering, and overlay endpoints.
 */
import { apiClient } from './client';

// ========== Voice Types ==========

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

// ========== Voice Endpoints ==========

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

export const deleteVoiceCluster = async (clusterId: number) => {
    const res = await apiClient.delete(`/voices/cluster/${clusterId}`);
    return res.data;
};

// ========== Voice Overlays ==========

export const getVoiceOverlays = async (
    videoPath: string,
    timestamp?: number,
): Promise<VoiceOverlayResponse> => {
    const res = await apiClient.get('/overlays/voice', {
        params: { video_id: videoPath, timestamp },
    });
    return res.data;
};
