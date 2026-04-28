import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
// base: './' 確保部署到 GitHub Pages 子路徑時資源路徑正確
export default defineConfig({
  plugins: [react()],
  base: './',
})
