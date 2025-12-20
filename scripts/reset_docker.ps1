# Reset Docker Environment

Write-Host "Stopping all containers..."
docker compose down -v

Write-Host "Pruning unused images/containers (optional, skipping aggressive prune to save time)..."
# docker system prune -f 

Write-Host "Pulling latest images..."
docker compose pull

Write-Host "Starting services..."
docker compose up -d

Write-Host "Waiting for services to stabilize..."
Start-Sleep -Seconds 10

Write-Host "Docker Status:"
docker ps

Write-Host "Checking Langfuse Logs..."
docker logs langfuse-server --tail 20

Write-Host "Checking Qdrant..."
curl http://localhost:6333/dashboard
