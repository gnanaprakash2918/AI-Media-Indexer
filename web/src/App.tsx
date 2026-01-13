import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
  NavLink,
} from 'react-router-dom';
import {useState, useMemo, lazy, Suspense} from 'react';
import {
  ThemeProvider,
  CssBaseline,
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  useMediaQuery,
  Divider,
  Chip,
  CircularProgress,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Search,
  CloudUpload,
  Face,
  Settings,
  LightMode,
  DarkMode,
  Dashboard,
  RecordVoiceOver,
} from '@mui/icons-material';
import {
  QueryClient,
  QueryClientProvider,
  useQuery,
} from '@tanstack/react-query';
import {AnimatePresence, motion} from 'framer-motion';

import {getTheme} from './theme';
import {healthCheck} from './api/client';

// Lazy load pages for faster initial load
const DashboardPage = lazy(() => import('./pages/Dashboard'));
const SearchPage = lazy(() => import('./pages/Search'));
const IngestPage = lazy(() => import('./pages/Ingest'));
const FacesPage = lazy(() => import('./pages/Faces'));
const VoicesPage = lazy(() => import('./pages/Voices'));
const SettingsPage = lazy(() => import('./pages/Settings'));

const PageLoader = () => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: 200,
    }}
  >
    <CircularProgress />
  </Box>
);

const DRAWER_WIDTH = 240;
const DRAWER_WIDTH_COLLAPSED = 72;

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60,
      retry: 1,
    },
  },
});

const navItems = [
  {path: '/', label: 'Dashboard', icon: <Dashboard />},
  {path: '/search', label: 'Search', icon: <Search />},
  {path: '/ingest', label: 'Ingest', icon: <CloudUpload />},
  {path: '/faces', label: 'Faces', icon: <Face />},
  {path: '/voices', label: 'Voices', icon: <RecordVoiceOver />},
  {path: '/settings', label: 'Settings', icon: <Settings />},
];

function Layout() {
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');
  const isMobile = useMediaQuery('(max-width: 768px)');

  const [mode, setMode] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('theme-mode');
    if (saved === 'light' || saved === 'dark') return saved;
    return prefersDarkMode ? 'dark' : 'light';
  });

  const [drawerOpen, setDrawerOpen] = useState(!isMobile);

  const theme = useMemo(() => getTheme(mode), [mode]);

  const toggleColorMode = () => {
    const newMode = mode === 'light' ? 'dark' : 'light';
    setMode(newMode);
    localStorage.setItem('theme-mode', newMode);
  };

  const health = useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    refetchInterval: 30000,
  });

  const drawerWidth = isMobile
    ? DRAWER_WIDTH
    : drawerOpen
      ? DRAWER_WIDTH
      : DRAWER_WIDTH_COLLAPSED;

  const drawerContent = (
    <Box sx={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <Toolbar
        sx={{
          px: 2,
          justifyContent: drawerOpen || isMobile ? 'flex-start' : 'center',
        }}
      >
        {(drawerOpen || isMobile) && (
          <Typography
            variant="h6"
            sx={{
              fontWeight: 800,
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              whiteSpace: 'nowrap',
            }}
          >
            AI Media Indexer
          </Typography>
        )}
        {!drawerOpen && !isMobile && (
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: '8px',
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 900,
              color: '#fff',
              fontSize: 14,
            }}
          >
            AI
          </Box>
        )}
      </Toolbar>
      <Divider />
      <List sx={{flex: 1, px: 1, py: 2}}>
        {navItems.map(item => (
          <ListItem key={item.path} disablePadding sx={{mb: 0.5}}>
            <ListItemButton
              component={NavLink}
              to={item.path}
              onClick={() => isMobile && setDrawerOpen(false)}
              sx={{
                borderRadius: 2,
                minHeight: 48,
                justifyContent: drawerOpen || isMobile ? 'initial' : 'center',
                px: 2.5,
                '&.active': {
                  bgcolor: 'primary.main',
                  color: 'primary.contrastText',
                  '& .MuiListItemIcon-root': {
                    color: 'primary.contrastText',
                  },
                },
                '&:hover:not(.active)': {
                  bgcolor: 'action.hover',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 0,
                  mr: drawerOpen || isMobile ? 2 : 0,
                  justifyContent: 'center',
                }}
              >
                {item.icon}
              </ListItemIcon>
              {(drawerOpen || isMobile) && (
                <ListItemText primary={item.label} />
              )}
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <Box sx={{p: 2, display: 'flex', justifyContent: 'center'}}>
        <Chip
          size="small"
          label={health.isSuccess ? 'Online' : 'Offline'}
          color={health.isSuccess ? 'success' : 'error'}
          variant="outlined"
          sx={{
            fontWeight: 600,
            display: drawerOpen || isMobile ? 'flex' : 'none',
          }}
        />
      </Box>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          display: 'flex',
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        {/* Mobile AppBar */}
        {isMobile && (
          <AppBar
            position="fixed"
            elevation={0}
            sx={{
              bgcolor: 'background.paper',
              borderBottom: 1,
              borderColor: 'divider',
            }}
          >
            <Toolbar>
              <IconButton
                edge="start"
                color="inherit"
                onClick={() => setDrawerOpen(true)}
                sx={{mr: 2}}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" sx={{flexGrow: 1, fontWeight: 700}}>
                AI Media Indexer
              </Typography>
              <IconButton onClick={toggleColorMode} color="inherit">
                {mode === 'dark' ? <LightMode /> : <DarkMode />}
              </IconButton>
            </Toolbar>
          </AppBar>
        )}

        {/* Sidebar Drawer */}
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={isMobile ? drawerOpen : true}
          onClose={() => setDrawerOpen(false)}
          sx={{
            width: drawerWidth,
            flexShrink: 0,
            '& .MuiDrawer-paper': {
              width: drawerWidth,
              boxSizing: 'border-box',
              borderRight: 1,
              borderColor: 'divider',
              bgcolor: 'background.paper',
              transition: 'width 0.2s ease-in-out',
            },
          }}
        >
          {drawerContent}
        </Drawer>

        {/* Main Content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            p: 3,
            mt: isMobile ? 8 : 0,
            minHeight: '100vh',
            overflow: 'auto',
          }}
        >
          {/* Desktop Header */}
          {!isMobile && (
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                mb: 3,
                pb: 2,
                borderBottom: 1,
                borderColor: 'divider',
              }}
            >
              <IconButton
                onClick={() => setDrawerOpen(!drawerOpen)}
                sx={{mr: 2}}
              >
                <MenuIcon />
              </IconButton>
              <Box sx={{flexGrow: 1}} />
              <IconButton onClick={toggleColorMode}>
                {mode === 'dark' ? <LightMode /> : <DarkMode />}
              </IconButton>
            </Box>
          )}

          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{opacity: 0, y: 10}}
              animate={{opacity: 1, y: 0}}
              exit={{opacity: 0, y: -10}}
              transition={{duration: 0.15}}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<PageLoader />}>
            <DashboardPage />
          </Suspense>
        ),
      },
      {
        path: 'search',
        element: (
          <Suspense fallback={<PageLoader />}>
            <SearchPage />
          </Suspense>
        ),
      },
      {
        path: 'ingest',
        element: (
          <Suspense fallback={<PageLoader />}>
            <IngestPage />
          </Suspense>
        ),
      },
      {
        path: 'faces',
        element: (
          <Suspense fallback={<PageLoader />}>
            <FacesPage />
          </Suspense>
        ),
      },
      {
        path: 'voices',
        element: (
          <Suspense fallback={<PageLoader />}>
            <VoicesPage />
          </Suspense>
        ),
      },
      {
        path: 'settings',
        element: (
          <Suspense fallback={<PageLoader />}>
            <SettingsPage />
          </Suspense>
        ),
      },
    ],
  },
]);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}

export default App;
