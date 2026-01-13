import {createTheme, type ThemeOptions, alpha} from '@mui/material/styles';

const baseTheme: ThemeOptions = {
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {fontWeight: 800, letterSpacing: '-0.02em'},
    h2: {fontWeight: 800, letterSpacing: '-0.02em'},
    h3: {fontWeight: 700, letterSpacing: '-0.01em'},
    h4: {fontWeight: 700},
    h5: {fontWeight: 600},
    h6: {fontWeight: 600},
    button: {fontWeight: 600, textTransform: 'none'},
  },
  shape: {
    borderRadius: 12,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 10,
          padding: '10px 20px',
          boxShadow: 'none',
          '&:hover': {
            boxShadow: 'none',
          },
        },
        contained: {
          '&:hover': {
            transform: 'translateY(-1px)',
            transition: 'transform 0.2s ease',
          },
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            borderRadius: 10,
          },
        },
      },
    },
    MuiLinearProgress: {
      styleOverrides: {
        root: {
          borderRadius: 4,
        },
      },
    },
  },
};

const lightPalette = {
  mode: 'light' as const,
  primary: {
    main: '#6366f1',
    light: '#818cf8',
    dark: '#4f46e5',
    contrastText: '#ffffff',
  },
  secondary: {
    main: '#ec4899',
    light: '#f472b6',
    dark: '#db2777',
    contrastText: '#ffffff',
  },
  success: {
    main: '#10b981',
    light: '#34d399',
    dark: '#059669',
  },
  error: {
    main: '#ef4444',
    light: '#f87171',
    dark: '#dc2626',
  },
  warning: {
    main: '#f59e0b',
    light: '#fbbf24',
    dark: '#d97706',
  },
  background: {
    default: '#f8fafc',
    paper: '#ffffff',
  },
  text: {
    primary: '#0f172a',
    secondary: '#64748b',
  },
  divider: '#e2e8f0',
};

const darkPalette = {
  mode: 'dark' as const,
  primary: {
    main: '#818cf8',
    light: '#a5b4fc',
    dark: '#6366f1',
    contrastText: '#0f0f23',
  },
  secondary: {
    main: '#f472b6',
    light: '#f9a8d4',
    dark: '#ec4899',
    contrastText: '#0f0f23',
  },
  success: {
    main: '#34d399',
    light: '#6ee7b7',
    dark: '#10b981',
  },
  error: {
    main: '#f87171',
    light: '#fca5a5',
    dark: '#ef4444',
  },
  warning: {
    main: '#fbbf24',
    light: '#fcd34d',
    dark: '#f59e0b',
  },
  background: {
    default: '#0a0a0f',
    paper: '#111118',
  },
  text: {
    primary: '#f1f5f9',
    secondary: '#94a3b8',
  },
  divider: '#1e293b',
};

export function getTheme(mode: 'light' | 'dark') {
  const palette = mode === 'dark' ? darkPalette : lightPalette;

  return createTheme({
    ...baseTheme,
    palette,
    components: {
      ...baseTheme.components,
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarColor:
              mode === 'dark' ? '#374151 #111118' : '#cbd5e1 #f8fafc',
            '&::-webkit-scrollbar': {
              width: 8,
              height: 8,
            },
            '&::-webkit-scrollbar-track': {
              background: mode === 'dark' ? '#111118' : '#f8fafc',
            },
            '&::-webkit-scrollbar-thumb': {
              background: mode === 'dark' ? '#374151' : '#cbd5e1',
              borderRadius: 4,
            },
            '&::-webkit-scrollbar-thumb:hover': {
              background: mode === 'dark' ? '#4b5563' : '#94a3b8',
            },
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            backgroundColor: palette.background.paper,
            borderRight: `1px solid ${palette.divider}`,
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundColor: alpha(palette.background.paper, 0.8),
            backdropFilter: 'blur(12px)',
            color: palette.text.primary,
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            '&.Mui-selected': {
              backgroundColor: palette.primary.main,
              color: palette.primary.contrastText,
              '&:hover': {
                backgroundColor: palette.primary.dark,
              },
            },
          },
        },
      },
    },
  });
}

export default getTheme;
