#!/bin/bash

# Stop any running containers
docker-compose -f docker-compose.debug.yml down

# Build and start the services in debug mode
DOCKER_BUILDKIT=1 docker-compose -f docker-compose.debug.yml up --build 