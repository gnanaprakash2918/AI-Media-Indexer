/**
 * Search API — all search-related endpoints and types.
 */
import { apiClient } from './client';

// ========== Search Types ==========

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

// ========== Search Endpoints ==========

/** Unified search with configurable toggles */
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

/** Name-based search */
export const searchByName = async (name: string, limit = 20) => {
    const res = await apiClient.get('/search/by-name', { params: { name, limit } });
    return res.data;
};

/** Granular / Agentic Search — unified endpoint with expansion + reranking */
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

/** Explainable search with all toggles enabled */
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

// ========== Feedback ==========

export const submitSearchFeedback = async (feedback: SearchFeedback) => {
    const res = await apiClient.post('/search/feedback', feedback);
    return res.data;
};

export const getSearchFeedbackStats = async () => {
    const res = await apiClient.get('/search/feedback/stats');
    return res.data;
};
