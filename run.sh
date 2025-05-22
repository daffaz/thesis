  #!/bin/bash

# Stop any running containers
docker-compose down

# Build and start the services
DOCKER_BUILDKIT=1 docker-compose up --build