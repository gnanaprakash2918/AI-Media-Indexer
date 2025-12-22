import { Search } from '@mui/icons-material';
import { Paper, InputBase, IconButton, alpha, useTheme, CircularProgress } from '@mui/material';
import { useState } from 'react';

interface SearchBarProps {
    onSearch: (query: string) => void;
    isLoading?: boolean;
}

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
    const [value, setValue] = useState("");
    const theme = useTheme();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (value.trim()) onSearch(value);
    };

    return (
        <Paper
            component="form"
            onSubmit={handleSubmit}
            elevation={0}
            sx={{
                p: '2px 4px',
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                maxWidth: 600,
                mx: 'auto',
                borderRadius: 3,
                border: `1px solid ${theme.palette.divider}`,
                transition: 'all 0.3s ease',
                '&:hover, &:focus-within': {
                    borderColor: theme.palette.primary.main,
                    boxShadow: `0 0 0 4px ${alpha(theme.palette.primary.main, 0.1)}`,
                }
            }}
        >
            <IconButton sx={{ p: '10px' }} aria-label="search" disabled>
                <Search sx={{ color: 'text.secondary' }} />
            </IconButton>
            <InputBase
                sx={{ ml: 1, flex: 1, fontWeight: 500 }}
                placeholder="Search for 'action scenes', 'face of Brad Pitt', 'angry voice'..."
                value={value}
                onChange={(e) => setValue(e.target.value)}
                disabled={isLoading}
            />
            <IconButton
                type="button"
                sx={{
                    p: '10px',
                    mr: 0.5,
                    color: 'primary.main',
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    '&:hover': {
                        bgcolor: alpha(theme.palette.primary.main, 0.2),
                    }
                }}
                aria-label="search"
                onClick={handleSubmit}
                disabled={isLoading}
            >
                {isLoading ? (
                    <CircularProgress size={24} color="inherit" />
                ) : (
                    <Search />
                )}
            </IconButton>
        </Paper>
    );
}
