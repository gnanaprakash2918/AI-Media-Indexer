import { useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Collapse,
    IconButton,
    Chip,
    Divider,
    alpha,
    useTheme,
    LinearProgress,
} from '@mui/material';
import {
    ExpandMore,
    ExpandLess,
    CheckCircle,
    RadioButtonUnchecked,
    SkipNext,
    Speed,
    Psychology,
    Search,
    Person,
    TextFields,
    Storage,
    AutoFixHigh,
} from '@mui/icons-material';

interface PipelineStep {
    step: string;
    status: 'completed' | 'running' | 'pending' | 'skipped';
    detail: string;
    data?: Record<string, any>;
}

interface SearchDebugPanelProps {
    pipelineSteps: PipelineStep[];
    reasoningChain?: Record<string, string>;
    searchText?: string;
    fallbackUsed?: string | null;
    durationMs?: number;
    isLoading?: boolean;
}

const stepIcons: Record<string, React.ReactNode> = {
    'Query Parsing': <Psychology fontSize="small" />,
    'Identity Resolution': <Person fontSize="small" />,
    'Query Expansion': <TextFields fontSize="small" />,
    'Vector Search': <Storage fontSize="small" />,
    'LLM Reranking': <AutoFixHigh fontSize="small" />,
};

const StatusIcon = ({ status }: { status: string }) => {
    switch (status) {
        case 'completed':
            return <CheckCircle sx={{ fontSize: 16, color: 'success.main' }} />;
        case 'running':
            return <RadioButtonUnchecked sx={{ fontSize: 16, color: 'info.main' }} />;
        case 'skipped':
            return <SkipNext sx={{ fontSize: 16, color: 'text.disabled' }} />;
        default:
            return <RadioButtonUnchecked sx={{ fontSize: 16, color: 'text.disabled' }} />;
    }
};

export function SearchDebugPanel({
    pipelineSteps,
    reasoningChain,
    searchText,
    fallbackUsed,
    durationMs,
    isLoading,
}: SearchDebugPanelProps) {
    const [expanded, setExpanded] = useState(true);
    const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set());
    const theme = useTheme();

    const toggleStep = (index: number) => {
        setExpandedSteps(prev => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    if (!pipelineSteps || pipelineSteps.length === 0) {
        return null;
    }

    return (
        <Paper
            sx={{
                mb: 3,
                overflow: 'hidden',
                border: '1px solid',
                borderColor: alpha(theme.palette.primary.main, 0.2),
                bgcolor: alpha(theme.palette.background.paper, 0.8),
            }}
        >
            {/* Header */}
            <Box
                sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1.5,
                    cursor: 'pointer',
                    bgcolor: alpha(theme.palette.primary.main, 0.05),
                    '&:hover': { bgcolor: alpha(theme.palette.primary.main, 0.1) },
                }}
                onClick={() => setExpanded(!expanded)}
            >
                <Search sx={{ color: 'primary.main', fontSize: 20 }} />
                <Typography variant="subtitle2" fontWeight={600} sx={{ flex: 1 }}>
                    Search Pipeline
                </Typography>

                {fallbackUsed && (
                    <Chip
                        label={`Fallback: ${fallbackUsed}`}
                        size="small"
                        color="warning"
                        sx={{ height: 22, fontSize: '0.7rem' }}
                    />
                )}

                {durationMs && (
                    <Chip
                        icon={<Speed sx={{ fontSize: 14 }} />}
                        label={`${(durationMs / 1000).toFixed(2)}s`}
                        size="small"
                        sx={{ height: 22, fontSize: '0.7rem' }}
                    />
                )}

                <IconButton size="small">
                    {expanded ? <ExpandLess /> : <ExpandMore />}
                </IconButton>
            </Box>

            {isLoading && <LinearProgress />}

            <Collapse in={expanded}>
                <Divider />
                <Box sx={{ p: 1.5 }}>
                    {/* Pipeline Steps */}
                    {pipelineSteps.map((step, index) => (
                        <Box key={index} sx={{ mb: 1 }}>
                            <Box
                                sx={{
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: 1,
                                    cursor: step.data ? 'pointer' : 'default',
                                    p: 0.5,
                                    borderRadius: 1,
                                    '&:hover': step.data ? { bgcolor: 'action.hover' } : {},
                                }}
                                onClick={() => step.data && toggleStep(index)}
                            >
                                <StatusIcon status={step.status} />

                                <Box sx={{ color: 'text.secondary', display: 'flex' }}>
                                    {stepIcons[step.step] || <RadioButtonUnchecked fontSize="small" />}
                                </Box>

                                <Typography
                                    variant="body2"
                                    fontWeight={500}
                                    sx={{
                                        color: step.status === 'skipped' ? 'text.disabled' : 'text.primary',
                                        minWidth: 140,
                                    }}
                                >
                                    {step.step}
                                </Typography>

                                <Typography
                                    variant="caption"
                                    sx={{
                                        color: step.status === 'skipped' ? 'text.disabled' : 'text.secondary',
                                        flex: 1,
                                    }}
                                >
                                    {step.detail}
                                </Typography>

                                {step.data && (
                                    <ExpandMore
                                        sx={{
                                            fontSize: 16,
                                            transform: expandedSteps.has(index) ? 'rotate(180deg)' : 'rotate(0)',
                                            transition: 'transform 0.2s',
                                            color: 'text.disabled',
                                        }}
                                    />
                                )}
                            </Box>

                            {/* Expanded Step Data */}
                            <Collapse in={expandedSteps.has(index)}>
                                {step.data && (
                                    <Box
                                        sx={{
                                            ml: 4,
                                            mt: 0.5,
                                            p: 1,
                                            bgcolor: alpha(theme.palette.background.default, 0.5),
                                            borderRadius: 1,
                                            border: '1px dashed',
                                            borderColor: 'divider',
                                        }}
                                    >
                                        {Object.entries(step.data).map(([key, value]) => (
                                            <Box
                                                key={key}
                                                sx={{
                                                    display: 'flex',
                                                    gap: 1,
                                                    mb: 0.5,
                                                    '&:last-child': { mb: 0 },
                                                }}
                                            >
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: 'text.secondary',
                                                        fontWeight: 600,
                                                        minWidth: 120,
                                                        textTransform: 'capitalize',
                                                    }}
                                                >
                                                    {key.replace(/_/g, ' ')}:
                                                </Typography>
                                                <Typography
                                                    variant="caption"
                                                    sx={{
                                                        color: 'text.primary',
                                                        fontFamily: typeof value === 'string' && value.length > 30 ? 'inherit' : 'monospace',
                                                        wordBreak: 'break-word',
                                                    }}
                                                >
                                                    {Array.isArray(value)
                                                        ? value.length > 0
                                                            ? value.join(', ')
                                                            : '(none)'
                                                        : typeof value === 'boolean'
                                                            ? value ? 'Yes' : 'No'
                                                            : String(value) || '(empty)'}
                                                </Typography>
                                            </Box>
                                        ))}
                                    </Box>
                                )}
                            </Collapse>
                        </Box>
                    ))}

                    {/* Expanded Search Text */}
                    {searchText && (
                        <>
                            <Divider sx={{ my: 1 }} />
                            <Box>
                                <Typography
                                    variant="caption"
                                    fontWeight={600}
                                    color="text.secondary"
                                    sx={{ display: 'block', mb: 0.5 }}
                                >
                                    Expanded Query:
                                </Typography>
                                <Typography
                                    variant="caption"
                                    sx={{
                                        display: 'block',
                                        p: 1,
                                        bgcolor: 'action.hover',
                                        borderRadius: 1,
                                        fontStyle: 'italic',
                                        lineHeight: 1.5,
                                    }}
                                >
                                    "{searchText}"
                                </Typography>
                            </Box>
                        </>
                    )}
                </Box>
            </Collapse>
        </Paper>
    );
}

export default SearchDebugPanel;
