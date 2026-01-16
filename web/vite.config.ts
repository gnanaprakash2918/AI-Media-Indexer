import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Use function-based chunk splitting to avoid breaking React context
        manualChunks(id) {
          if (id.includes('node_modules')) {
            // Keep React together to avoid context issues
            if (id.includes('react') || id.includes('scheduler')) {
              return 'vendor-react';
            }
            // MUI icons are huge and rarely change
            if (id.includes('@mui/icons-material')) {
              return 'vendor-mui-icons';
            }
            // MUI core
            if (id.includes('@mui/material') || id.includes('@emotion')) {
              return 'vendor-mui';
            }
            // Data fetching libraries
            if (id.includes('@tanstack') || id.includes('axios')) {
              return 'vendor-query';
            }
          }
        },
      },
    },
  },
})
