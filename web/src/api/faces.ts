/**
 * Faces API — face detection, clustering, and identity management.
 */
import { apiClient } from './client';

// ========== Face Types ==========

export interface FaceCluster {
    id: string;
    cluster_id: number;
    name: string | null;
    media_path?: string;
    timestamp?: number;
    thumbnail_path?: string;
}

// ========== Face Endpoints ==========

export const getUnresolvedFaces = async (limit = 50) => {
    const res = await apiClient.get('/faces/unresolved', { params: { limit } });
    return res.data;
};

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

export const deleteFaceCluster = async (clusterId: number) => {
    const res = await apiClient.delete(`/faces/cluster/${clusterId}`);
    return res.data;
};

export const identifyFaceCluster = async (clusterId: number) => {
    const res = await apiClient.post(`/faces/cluster/${clusterId}/identify`);
    return res.data;
};

// ========== Identity APIs ==========

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
