/**
 * Shared type definitions used across multiple frontend components.
 *
 * Consolidates duplicated interfaces from Faces.tsx, Voices page, and
 * client.ts into a single source of truth.
 */

// ─── Face Types ────────────────────────────────────────────────────────────────

export interface FaceData {
    id: string;
    cluster_id: number;
    name: string | null;
    media_path?: string;
    timestamp?: number;
    thumbnail_path?: string;
}

export interface FaceClusterData {
    cluster_id: number;
    name: string | null;
    face_count: number;
    representative: FaceData | null;
    faces: FaceData[];
    is_main?: boolean;
}

export interface IdentitySuggestion {
    type: string;
    source: string;
    target: string;
    reason: string;
    confidence: number;
    source_id?: number;
    target_id?: number;
}

// ─── Voice Types ───────────────────────────────────────────────────────────────

export interface VoiceSegment {
    id: string;
    cluster_id: number;
    speaker_name: string | null;
    media_path?: string;
    start_time?: number;
    end_time?: number;
    text?: string;
}

export interface VoiceClusterData {
    cluster_id: number;
    name: string | null;
    segment_count: number;
    segments: VoiceSegment[];
}

// ─── Search Types ──────────────────────────────────────────────────────────────

export interface SearchResult {
    media_path: string;
    score: number;
    timestamp?: number;
    end_time?: number;
    description?: string;
    transcript?: string;
    source_type?: string;
    thumbnail_path?: string;
}

export interface SearchResponse {
    results: SearchResult[];
    total: number;
    search_degraded?: boolean;
    degradation_reason?: string;
}

// ─── Media Types ───────────────────────────────────────────────────────────────

export interface MediaItem {
    path: string;
    title?: string;
    media_type?: string;
    duration?: number;
    thumbnail?: string;
    indexed_at?: number;
}

// ─── Graph Types ───────────────────────────────────────────────────────────────

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

// ─── API Error ─────────────────────────────────────────────────────────────────

export interface ApiError {
    status: number;
    message: string;
    detail?: string;
}
