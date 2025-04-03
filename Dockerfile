# Multi-stage build for NeuraShield components

# Backend build
FROM node:18-alpine AS backend-build
WORKDIR /app/backend
COPY backend/package*.json ./
RUN npm ci
COPY backend/ .

# Frontend build
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# AI Service build
FROM python:3.9-slim AS ai-build
WORKDIR /app/ai_models
COPY ai_models/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ai_models/ .

# Backend runtime
FROM node:18-alpine AS backend
WORKDIR /app
COPY --from=backend-build /app/backend ./backend
WORKDIR /app/backend
ENV NODE_ENV=production
EXPOSE 3001
CMD ["node", "src/server.js"]

# Frontend runtime
FROM node:18-alpine AS frontend
WORKDIR /app
COPY --from=frontend-build /app/frontend ./frontend
COPY --from=frontend-build /app/frontend/.next ./.next
WORKDIR /app/frontend
ENV NODE_ENV=production
EXPOSE 3000
CMD ["npm", "start"]

# AI service runtime
FROM python:3.9-slim AS ai-service
WORKDIR /app
COPY --from=ai-build /app/ai_models ./ai_models
WORKDIR /app/ai_models
ENV PYTHONPATH=/app
EXPOSE 5000
EXPOSE 8000
CMD ["python", "-m", "ai_models.main"] 