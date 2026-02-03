import { useState, memo } from 'react';
import {
  PlayArrow,
  AccessTime,
  Videocam,
  GraphicEq,
  TextFields,
  Face,
  Mic,
  LocalOffer,
  Room,
  Visibility,
  Edit,
  ThumbUp,
  ThumbDown,
  ExpandMore,
  BugReport,
} from '@mui/icons-material';
import {
  Card,
  CardMedia,
  CardContent,
  Typography,
  Box,
  Chip,
  IconButton,
  alpha,
  useTheme,
  Tooltip,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  CircularProgress,
  Snackbar,
} from '@mui/material';
import { VideoPlayer } from './VideoPlayer';
import { updateFrameDescription, submitSearchFeedback } from '../../api/client';

interface MediaResult {
  score: number;
  base_score?: number;
  keyword_boost?: number;
  video_path: string;
  start?: number;
  end?: number;
  timestamp?: number;
  // Scene-level timestamps (from scenes collection)
  start_time?: number;
  end_time?: number;
  text?: string;
  type?: string;
  action?: string;
  description?: string;
  thumbnail_url?: string;
  id?: string;
  // HITL Identity Fields
  face_names?: string[];
  speaker_names?: string[];
  face_cluster_ids?: number[];
  // Structured Analysis Fields
  entities?: string[];
  visible_text?: string[];
  scene_location?: string;
  identity_text?: string;
  // Agentic Search Explainability
  match_reason?: string;
  agent_thought?: string;
  matched_constraints?: string[];
  // RRF Hybrid Search Explainability
  match_reasons?: string[];
  rrf_score?: number;
  vector_score?: number;
  keyword_score?: number;
  matched_identity?: string;
  // Deep debug info from backend
  _debug?: {
    modalities_used?: string[];
    models_contributed?: string[];
    raw_fused_score?: number;
    normalized_score?: number;
    match_type?: string;
  };
  modality_sources?: string[];
  fused_score?: number;
  person_names?: string[];
  visual_summary?: string;
  dialogue_summary?: string;
}

interface MediaCardProps {
  item: MediaResult;
  searchQuery?: string; // For HITL feedback
  overlayToggles?: {
    faces: boolean;
    text: boolean;
    objects: boolean;
    speakers: boolean;
  };
}

export const MediaCard = memo(function MediaCard({ item, searchQuery, overlayToggles }: MediaCardProps) {
  const theme = useTheme();
  const [playerOpen, setPlayerOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editDescription, setEditDescription] = useState(
    item.action || item.description || '',
  );
  const [saving, setSaving] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<'up' | 'down' | null>(null);
  const [snackOpen, setSnackOpen] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  // HITL Feedback handler
  const handleFeedback = async (isRelevant: boolean) => {
    if (!searchQuery || feedbackGiven) return;
    try {
      await submitSearchFeedback({
        query: searchQuery,
        result_id: item.id || '',
        video_path: item.video_path,
        timestamp: item.timestamp ?? item.start ?? 0,
        is_relevant: isRelevant,
        feedback_type: 'binary',
      });
      setFeedbackGiven(isRelevant ? 'up' : 'down');
      setSnackOpen(true);
    } catch (_err) {
      console.error('Feedback failed', _err);
    }
  };

  // Handle both video_path and media_path (scenelets use media_path)
  const videoPath = item.video_path || (item as MediaResult & { media_path?: string }).media_path || '';

  // Extract filename for thumbnail
  const filename = videoPath ? videoPath.split(/[/\\]/).pop() : 'unknown';
  const baseName = filename?.replace(/\.[^/.]+$/, '') || '';

  // Use dynamic thumbnail if available, else fallback to static or placeholder
  const thumbnailUrl = item.thumbnail_url
    ? `http://localhost:8000${item.thumbnail_url}`
    : `http://localhost:8000/thumbnails/${filename}.jpg`;

  // Format timestamp
  const formatTime = (s?: number) => {
    if (s === undefined) return '';
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getIcon = () => {
    if (item.type === 'dialogue') return <GraphicEq fontSize="small" />;
    if (item.type === 'visual') return <Videocam fontSize="small" />;
    return <TextFields fontSize="small" />;
  };

  const handlePlay = () => setPlayerOpen(true);

  const handleSaveDescription = async () => {
    if (!item.id || !editDescription.trim()) return;
    setSaving(true);
    try {
      await updateFrameDescription(item.id, editDescription.trim());
      item.action = editDescription.trim();
      item.description = editDescription.trim();
      setEditOpen(false);
    } catch (err) {
      console.error('Failed to update description', err);
    } finally {
      setSaving(false);
    }
  };

  // Check if we have HITL data
  const hasFaces = item.face_names && item.face_names.length > 0;
  const hasSpeakers = item.speaker_names && item.speaker_names.length > 0;
  const hasEntities = item.entities && item.entities.length > 0;
  const hasVisibleText = item.visible_text && item.visible_text.length > 0;
  const hasScene = item.scene_location && item.scene_location.length > 0;
  const hasDetails =
    hasFaces || hasSpeakers || hasEntities || hasVisibleText || hasScene;

  // Use timestamp or start
  const timeValue = item.timestamp ?? item.start;

  return (
    <>
      <Card
        sx={{
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          position: 'relative',
        }}
        onClick={() => hasDetails && setExpanded(!expanded)}
      >
        <Box
          sx={{
            position: 'relative',
            paddingTop: '56.25%' /* 16:9 Aspect Ratio */,
          }}
        >
          <CardMedia
            component="img"
            image={thumbnailUrl}
            alt={filename}
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
            }}
            onError={e => {
              (e.target as HTMLImageElement).src =
                'https://placehold.co/600x400/18181b/52525b?text=No+Preview';
            }}
          />

          {/* Overlay Gradient */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              width: '100%',
              height: '50%',
              background:
                'linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 100%)',
            }}
          />

          {/* Timestamp Badge */}
          {timeValue !== undefined && (
            <Box
              sx={{
                position: 'absolute',
                bottom: 8,
                right: 8,
                bgcolor: 'rgba(0, 0, 0, 0.75)',
                color: 'white',
                px: 1,
                py: 0.5,
                borderRadius: 1,
                display: 'flex',
                alignItems: 'center',
                gap: 0.5,
                fontSize: '0.75rem',
                fontWeight: 600,
                backdropFilter: 'blur(4px)',
              }}
            >
              <AccessTime sx={{ fontSize: 14 }} />
              {formatTime(timeValue)}
            </Box>
          )}

          {/* Face Names Badge (top-left) */}
          {hasFaces && (
            <Box
              sx={{
                position: 'absolute',
                top: 8,
                left: 8,
                display: 'flex',
                gap: 0.5,
                flexWrap: 'wrap',
              }}
            >
              {item.face_names!.slice(0, 2).map((name, i) => (
                <Chip
                  key={i}
                  icon={<Face sx={{ fontSize: 14 }} />}
                  label={name}
                  size="small"
                  sx={{
                    bgcolor: alpha(theme.palette.info.main, 0.9),
                    color: 'white',
                    fontWeight: 600,
                    height: 24,
                  }}
                />
              ))}
            </Box>
          )}

          {/* Play Button Overlay (Hover) */}
          <Box
            className="play-overlay"
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              opacity: 0,
              transition: 'opacity 0.2s',
              bgcolor: 'rgba(0,0,0,0.3)',
              '&:hover': { opacity: 1 },
            }}
          >
            <IconButton
              onClick={e => {
                e.stopPropagation();
                handlePlay();
              }}
              sx={{
                bgcolor: 'rgba(255,255,255,0.2)',
                backdropFilter: 'blur(8px)',
                '&:hover': { bgcolor: 'rgba(255,255,255,0.3)' },
                width: 56,
                height: 56,
              }}
            >
              <PlayArrow sx={{ color: 'white', fontSize: 32 }} />
            </IconButton>
            {item.id && (
              <Tooltip title="Edit Description (HITL)">
                <IconButton
                  onClick={e => {
                    e.stopPropagation();
                    setEditOpen(true);
                  }}
                  sx={{
                    bgcolor: 'rgba(255,255,255,0.2)',
                    backdropFilter: 'blur(8px)',
                    '&:hover': { bgcolor: 'rgba(255,200,100,0.4)' },
                    ml: 1,
                  }}
                >
                  <Edit sx={{ color: 'white' }} />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>

        <CardContent sx={{ flexGrow: 1, p: 2 }}>
          {/* Score and Type Row */}
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 1,
            }}
          >
            <Chip
              icon={getIcon()}
              label={item.type || 'Match'}
              size="small"
              sx={{
                height: 24,
                fontSize: '0.75rem',
                fontWeight: 600,
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                color: theme.palette.primary.main,
              }}
            />
            <Tooltip
              title={
                item.base_score != null
                  ? `Base: ${((item.base_score ?? 0) * 100).toFixed(0)}% + Keyword: ${((item.keyword_boost ?? 0) * 100).toFixed(0)}%`
                  : 'Confidence Score'
              }
            >
              <Typography
                variant="caption"
                sx={{
                  color: 'success.main',
                  fontWeight: 700,
                  fontFamily: 'monospace',
                }}
              >
                {((item.score ?? 0) * 100).toFixed(0)}%
              </Typography>
            </Tooltip>

            {/* HITL Feedback Buttons */}
            {searchQuery && (
              <Box sx={{ display: 'flex', gap: 0.5, ml: 1 }}>
                <Tooltip title="Relevant result">
                  <IconButton
                    size="small"
                    onClick={(e) => { e.stopPropagation(); handleFeedback(true); }}
                    disabled={feedbackGiven !== null}
                    sx={{
                      width: 24,
                      height: 24,
                      color: feedbackGiven === 'up' ? 'success.main' : 'action.disabled',
                      '&:hover': { color: 'success.main' },
                    }}
                  >
                    <ThumbUp sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
                <Tooltip title="Not relevant">
                  <IconButton
                    size="small"
                    onClick={(e) => { e.stopPropagation(); handleFeedback(false); }}
                    disabled={feedbackGiven !== null}
                    sx={{
                      width: 24,
                      height: 24,
                      color: feedbackGiven === 'down' ? 'error.main' : 'action.disabled',
                      '&:hover': { color: 'error.main' },
                    }}
                  >
                    <ThumbDown sx={{ fontSize: 16 }} />
                  </IconButton>
                </Tooltip>
              </Box>
            )}
          </Box>

          {/* Speaker Names */}
          {hasSpeakers && (
            <Box sx={{ display: 'flex', gap: 0.5, mb: 1, flexWrap: 'wrap' }}>
              {item.speaker_names!.map((name, i) => (
                <Chip
                  key={i}
                  icon={<Mic sx={{ fontSize: 12 }} />}
                  label={name}
                  size="small"
                  variant="outlined"
                  sx={{ height: 20, fontSize: '0.7rem' }}
                />
              ))}
            </Box>
          )}

          {/* Subtitle with time range */}
          {item.text && (
            <Box sx={{ bgcolor: 'action.hover', p: 1, borderRadius: 1, mb: 1 }}>
              <Typography
                variant="body2"
                sx={{ fontStyle: 'italic', fontWeight: 500 }}
              >
                "{item.text}"
              </Typography>
              {item.start !== undefined && item.end !== undefined && (
                <Typography variant="caption" color="text.secondary">
                  {formatTime(item.start)} - {formatTime(item.end)}
                </Typography>
              )}
            </Box>
          )}

          {!item.text && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                overflow: 'hidden',
                lineHeight: 1.6,
                fontWeight: 500,
              }}
            >
              {item.action ||
                item.description ||
                'Visual match detected in scene.'}
            </Typography>
          )}

          {/* Search Explainability - Match Reason */}
          {(item.match_reason || item.agent_thought || (item.match_reasons && item.match_reasons.length > 0)) && (
            <Box
              sx={{
                mt: 1,
                p: 1,
                bgcolor: alpha(theme.palette.success.main, 0.08),
                borderRadius: 1,
                border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`,
              }}
            >
              <Typography
                variant="caption"
                sx={{
                  fontWeight: 600,
                  color: 'success.main',
                  display: 'block',
                  mb: 0.5,
                }}
              >
                🎯 Match Reasoning
              </Typography>

              {item.match_reasons ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {item.match_reasons.map((reason, i) => (
                    <Typography
                      key={i}
                      variant="caption"
                      color="text.secondary"
                      sx={{ lineHeight: 1.4, display: 'flex', alignItems: 'flex-start', gap: 0.5 }}
                    >
                      <span style={{ color: theme.palette.success.main }}>•</span>
                      {reason}
                    </Typography>
                  ))}
                </Box>
              ) : (
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ lineHeight: 1.4 }}
                >
                  {item.match_reason || item.agent_thought}
                </Typography>
              )}

              {item.matched_constraints &&
                item.matched_constraints.length > 0 && (
                  <Box
                    sx={{ mt: 0.5, display: 'flex', gap: 0.5, flexWrap: 'wrap' }}
                  >
                    {item.matched_constraints.map((c, i) => (
                      <Chip
                        key={i}
                        label={c}
                        size="small"
                        sx={{
                          height: 18,
                          fontSize: '0.6rem',
                          bgcolor: alpha(theme.palette.success.main, 0.15),
                          color: 'success.dark',
                        }}
                      />
                    ))}
                  </Box>
                )}
            </Box>
          )}

          {/* 🔍 DETAILED DEBUG PANEL - Toggle to see ALL internal data */}
          <Box sx={{ mt: 1 }}>
            <Button
              size="small"
              startIcon={<BugReport sx={{ fontSize: 14 }} />}
              endIcon={<ExpandMore sx={{
                transform: showDebug ? 'rotate(180deg)' : 'rotate(0deg)',
                transition: 'transform 0.2s'
              }} />}
              onClick={(e) => { e.stopPropagation(); setShowDebug(!showDebug); }}
              sx={{
                fontSize: '0.65rem',
                textTransform: 'none',
                color: 'text.secondary',
                '&:hover': { bgcolor: alpha(theme.palette.info.main, 0.1) }
              }}
            >
              {showDebug ? 'Hide Debug Info' : 'Show All Details'}
            </Button>

            <Collapse in={showDebug}>
              <Box
                sx={{
                  mt: 1,
                  p: 1.5,
                  bgcolor: alpha(theme.palette.info.main, 0.05),
                  borderRadius: 1,
                  border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`,
                  fontSize: '0.7rem',
                }}
              >
                {/* Score Breakdown */}
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                    📊 SCORE DETAILS
                  </Typography>
                  <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5 }}>
                    <Typography variant="caption">Final Score: <strong>{((item.score ?? 0) * 100).toFixed(1)}%</strong></Typography>
                    {item.fused_score != null && <Typography variant="caption">Fused: {item.fused_score.toFixed(4)}</Typography>}
                    {item.vector_score != null && <Typography variant="caption">Vector: {(item.vector_score * 100).toFixed(1)}%</Typography>}
                    {item.keyword_score != null && <Typography variant="caption">Keyword: {(item.keyword_score * 100).toFixed(1)}%</Typography>}
                    {item._debug?.raw_fused_score != null && <Typography variant="caption">Raw Fused: {item._debug.raw_fused_score.toFixed(4)}</Typography>}
                    {item._debug?.match_type && <Typography variant="caption">Match Type: {item._debug.match_type}</Typography>}
                  </Box>
                </Box>

                {/* Modalities Used */}
                {(item._debug?.modalities_used || item.modality_sources) && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      🔗 MODALITIES SEARCHED
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {(item._debug?.modalities_used || item.modality_sources || []).map((mod, i) => (
                        <Chip key={i} label={mod} size="small" sx={{ height: 18, fontSize: '0.6rem' }} color="info" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* AI Models Used */}
                {item._debug?.models_contributed && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      🤖 AI MODELS USED
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {item._debug.models_contributed.map((model, i) => (
                        <Chip key={i} label={model} size="small" sx={{ height: 18, fontSize: '0.6rem' }} color="secondary" variant="outlined" />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* Visual Summary */}
                {(item.visual_summary || item.action || item.description) && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      👁️ VISUAL DESCRIPTION (from VLM)
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', bgcolor: 'background.paper', p: 0.5, borderRadius: 0.5 }}>
                      {item.visual_summary || item.action || item.description || 'No visual description available'}
                    </Typography>
                  </Box>
                )}

                {/* Dialogue Summary */}
                {item.dialogue_summary && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      🎤 DIALOGUE (from ASR/Whisper)
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', bgcolor: 'background.paper', p: 0.5, borderRadius: 0.5, fontStyle: 'italic' }}>
                      "{item.dialogue_summary}"
                    </Typography>
                  </Box>
                )}

                {/* All People Detected */}
                {(item.face_names || item.person_names) && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      👤 ALL PEOPLE DETECTED
                    </Typography>
                    <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                      {[...(item.face_names || []), ...(item.person_names || [])].filter((v, i, a) => a.indexOf(v) === i).map((name, i) => (
                        <Chip key={i} icon={<Face sx={{ fontSize: 10 }} />} label={name} size="small" sx={{ height: 18, fontSize: '0.6rem' }} />
                      ))}
                    </Box>
                  </Box>
                )}

                {/* Face Cluster IDs */}
                {item.face_cluster_ids && item.face_cluster_ids.length > 0 && (
                  <Box sx={{ mb: 1.5 }}>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      🆔 FACE CLUSTER IDs
                    </Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      [{item.face_cluster_ids.join(', ')}]
                    </Typography>
                  </Box>
                )}

                {/* Raw ID */}
                {item.id && (
                  <Box>
                    <Typography variant="caption" fontWeight={700} color="info.main" sx={{ display: 'block', mb: 0.5 }}>
                      🔑 RESULT ID
                    </Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {item.id}
                    </Typography>
                  </Box>
                )}
              </Box>
            </Collapse>
          </Box>

          <Collapse in={expanded}>
            <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
              {/* Face Names - SHOW ALL detected faces */}
              {hasFaces && (
                <Box sx={{ mb: 1 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      mb: 0.5,
                    }}
                  >
                    <Face sx={{ fontSize: 12, color: 'primary.main' }} />
                    <Typography variant="caption" color="primary.main" fontWeight={600}>
                      People Detected
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {item.face_names!.map((name, i) => (
                      <Chip
                        key={i}
                        icon={<Face sx={{ fontSize: 10 }} />}
                        label={name}
                        size="small"
                        color="primary"
                        variant="outlined"
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Entities */}
              {hasEntities && (
                <Box sx={{ mb: 1 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      mb: 0.5,
                    }}
                  >
                    <LocalOffer sx={{ fontSize: 12, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">
                      Entities
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {item.entities!.slice(0, 5).map((e, i) => (
                      <Chip
                        key={i}
                        label={e}
                        size="small"
                        sx={{ height: 18, fontSize: '0.65rem' }}
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Scene Location */}
              {hasScene && (
                <Box
                  sx={{ mb: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}
                >
                  <Room sx={{ fontSize: 12, color: 'text.secondary' }} />
                  <Typography variant="caption">
                    {item.scene_location}
                  </Typography>
                </Box>
              )}

              {/* Visible Text (OCR) */}
              {hasVisibleText && (
                <Box sx={{ mb: 1 }}>
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      mb: 0.5,
                    }}
                  >
                    <Visibility sx={{ fontSize: 12, color: 'text.secondary' }} />
                    <Typography variant="caption" color="text.secondary">
                      Visible Text
                    </Typography>
                  </Box>
                  <Typography
                    variant="caption"
                    sx={{
                      fontFamily: 'monospace',
                      bgcolor: 'action.hover',
                      px: 0.5,
                      borderRadius: 0.5,
                    }}
                  >
                    {item.visible_text!.join(' | ')}
                  </Typography>
                </Box>
              )}

              {/* Score Breakdown */}
              {item.base_score != null && (
                <Box
                  sx={{
                    display: 'flex',
                    gap: 1,
                    fontSize: '0.65rem',
                    color: 'text.secondary',
                  }}
                >
                  <span>Base: {((item.base_score ?? 0) * 100).toFixed(0)}%</span>
                  <span>
                    Boost: +{((item.keyword_boost ?? 0) * 100).toFixed(0)}%
                  </span>
                </Box>
              )}
            </Box>
          </Collapse>

          <Typography
            variant="caption"
            color="text.disabled"
            sx={{ mt: 1, display: 'block', fontFamily: 'monospace' }}
            noWrap
          >
            {baseName}
          </Typography>

          {/* Expand indicator */}
          {hasDetails && (
            <Typography
              variant="caption"
              color="primary"
              sx={{
                cursor: 'pointer',
                display: 'block',
                textAlign: 'center',
                mt: 0.5,
              }}
            >
              {expanded ? '▲ Less' : '▼ More details'}
            </Typography>
          )}
        </CardContent>
      </Card>
      <VideoPlayer
        videoPath={videoPath}
        startTime={item.start_time ?? item.start ?? item.timestamp}
        endTime={item.end_time ?? item.end}
        open={playerOpen}
        onClose={() => setPlayerOpen(false)}
        overlayToggles={overlayToggles}
      />
      <Dialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Edit Description (HITL Correction)</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            multiline
            rows={4}
            value={editDescription}
            onChange={e => setEditDescription(e.target.value)}
            placeholder="Describe what's actually in this frame..."
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveDescription}
            disabled={saving || !editDescription.trim()}
          >
            {saving ? <CircularProgress size={20} /> : 'Save & Re-embed'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* HITL Feedback Snackbar */}
      <Snackbar
        open={snackOpen}
        autoHideDuration={3000}
        onClose={() => setSnackOpen(false)}
        message={feedbackGiven === 'up' ? '👍 Thanks! Marked as relevant.' : '👎 Thanks! Marked as not relevant.'}
      />
    </>
  );
});
