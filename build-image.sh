#!/bin/sh

# Create docker image containing Active Learning tools 

# @author  Fabrice Jammes

set -e
#set -x

DIR=$(cd "$(dirname "$0")"; pwd -P)

GIT_HASH="$(git describe --dirty --always)"
TAG=${DEPLOY_VERSION:-${GIT_HASH}}


PRODUCT="activelearning"
IMAGE="$PRODUCT:$TAG"

echo "Building image $IMAGE"

docker build --tag "$IMAGE" "$DIR"
docker tag "$IMAGE" "$PRODUCT:latest"
#docker push "$IMAGE"
#echo "$IMAGE pushed to Docker Hub"
