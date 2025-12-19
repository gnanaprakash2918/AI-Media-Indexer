FROM qdrant/qdrant:v1.9.3
FROM postgres:16
FROM clickhouse/clickhouse-server:24.3
FROM redis:7.2.4
FROM minio/minio
FROM minio/mc
FROM langfuse/langfuse:3