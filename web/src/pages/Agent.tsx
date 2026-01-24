import { useState, useRef, useEffect } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import {
    Box,
    Typography,
    TextField,
    Paper,
    IconButton,
    List,
    ListItem,
    ListItemAvatar,
    Avatar,
    ListItemText,
    CircularProgress,
    Chip,
    Divider,
} from '@mui/material';
import { Send, SmartToy, Person, RestartAlt } from '@mui/icons-material';
import { apiClient } from '../api/client';

interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
}

interface AgentStatus {
    status: string;
    model: string;
    loaded: boolean;
}

const getAgentStatus = async () => {
    const res = await apiClient.get<AgentStatus>('/agent/status');
    return res.data;
};

const sendChat = async (message: string) => {
    const res = await apiClient.post('/agent/chat', {
        message,
        session_id: 'default-session' // Simple session for now
    });
    return res.data; // { response: string }
};

export default function AgentPage() {
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    const { data: status } = useQuery({
        queryKey: ['agent', 'status'],
        queryFn: getAgentStatus,
        refetchInterval: 5000,
    });

    const chatMutation = useMutation({
        mutationFn: sendChat,
        onSuccess: (data) => {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: data.response, timestamp: Date.now() }
            ]);
        },
        onError: (error) => {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: `Error: ${error.message}`, timestamp: Date.now() }
            ]);
        }
    });

    const handleSend = () => {
        if (!input.trim() || chatMutation.isPending) return;

        const userMsg: ChatMessage = { role: 'user', content: input, timestamp: Date.now() };
        setMessages(prev => [...prev, userMsg]);
        chatMutation.mutate(input);
        setInput('');
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, chatMutation.isPending]);

    return (
        <Box sx={{ height: 'calc(100vh - 100px)', display: 'flex', flexDirection: 'column' }}>
            {/* Header */}
            <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Box>
                    <Typography variant="h5" fontWeight={700}>Agent Chat</Typography>
                    <Typography variant="body2" color="text.secondary">
                        Chat with your video library using the Context-Aware Agent.
                    </Typography>
                </Box>
                <Box sx={{ textAlign: 'right' }}>
                    <Chip
                        icon={<SmartToy />}
                        label={status?.status || 'Unknown'}
                        color={status?.status === 'ready' ? 'success' : 'warning'}
                        variant="outlined"
                    />
                    {status?.model && (
                        <Typography variant="caption" display="block" sx={{ mt: 0.5, fontFamily: 'monospace' }}>
                            Model: {status.model}
                        </Typography>
                    )}
                </Box>
            </Box>

            {/* Chat Area */}
            <Paper sx={{ flex: 1, mb: 2, overflow: 'hidden', display: 'flex', flexDirection: 'column', p: 0 }}>
                <Box
                    ref={scrollRef}
                    sx={{
                        flex: 1,
                        overflowY: 'auto',
                        p: 3,
                        bgcolor: 'action.hover',
                        display: 'flex',
                        flexDirection: 'column',
                        gap: 2
                    }}
                >
                    {messages.length === 0 && (
                        <Box sx={{ textAlign: 'center', mt: 10, opacity: 0.6 }}>
                            <SmartToy sx={{ fontSize: 60, mb: 2, color: 'primary.main' }} />
                            <Typography variant="h6">How can I help you explore your media?</Typography>
                            <Typography variant="body2">
                                Try asking: "Find videos of Prakash bowling", "Summarize the last vacation video", or "Who is the main speaker in the interview?"
                            </Typography>
                        </Box>
                    )}

                    {messages.map((msg, idx) => (
                        <Box
                            key={idx}
                            sx={{
                                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                                maxWidth: '80%',
                                display: 'flex',
                                gap: 2,
                                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
                            }}
                        >
                            <Avatar
                                sx={{
                                    bgcolor: msg.role === 'user' ? 'secondary.main' : 'primary.main',
                                    width: 32, height: 32
                                }}
                            >
                                {msg.role === 'user' ? <Person fontSize="small" /> : <SmartToy fontSize="small" />}
                            </Avatar>
                            <Paper
                                sx={{
                                    p: 2,
                                    borderRadius: 3,
                                    bgcolor: msg.role === 'user' ? 'primary.dark' : 'background.paper',
                                    color: msg.role === 'user' ? 'white' : 'text.primary',
                                    boxShadow: 1
                                }}
                            >
                                <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                                    {msg.content}
                                </Typography>
                            </Paper>
                        </Box>
                    ))}

                    {chatMutation.isPending && (
                        <Box sx={{ display: 'flex', gap: 2 }}>
                            <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                                <SmartToy fontSize="small" />
                            </Avatar>
                            <Paper sx={{ p: 2, borderRadius: 3 }}>
                                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                    <CircularProgress size={16} />
                                    <Typography variant="body2" color="text.secondary">Thinking...</Typography>
                                </Box>
                            </Paper>
                        </Box>
                    )}
                </Box>

                <Divider />

                {/* Input Area */}
                <Box sx={{ p: 2, bgcolor: 'background.paper' }}>
                    <TextField
                        fullWidth
                        placeholder="Ask a question about your videos..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={chatMutation.isPending}
                        multiline
                        maxRows={4}
                        slotProps={{
                            input: {
                                endAdornment: (
                                    <IconButton
                                        color="primary"
                                        onClick={handleSend}
                                        disabled={!input.trim() || chatMutation.isPending}
                                    >
                                        <Send />
                                    </IconButton>
                                )
                            }
                        }}
                    />
                </Box>
            </Paper>
        </Box>
    );
}
