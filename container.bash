#!/usr/bin/env bash
# inputs
MODE=$1
VERSION=$2
GPU=$3
CMD=$4

# default paths
DATASETS_PATH="$(pwd)/../../datasets"
CURRENT_FOLDER="$(pwd)"
WANDB_KEY=06de2b089b5d98ee67dcf4fdffce3368e8bac2e4
USER=dkm
USER_ID=1003
USER_GROUP=dkm
USER_GROUP_ID=1003

# variables
DOCKER_FOLDER="/home/drigoni/repository/ban-vqa"


if [[ $MODE == "build" ]]; then
  # build container
  docker build ./ -t $VERSION
elif [[ $MODE == "exec" ]]; then
  echo "Remove previous container: "
  docker container rm ${VERSION}-${GPU//,}
  # execute container
  echo "Execute container:"
  docker run \
    -u ${USER}:${USER_GROUP} \
    --env CUDA_VISIBLE_DEVICES=${GPU} \
    --env WANDB_API_KEY=${WANDB_KEY}\
    --name ${VERSION}-${GPU//,} \
    --runtime=nvidia \
    --ipc=host \
    -it  \
    -v ${CURRENT_FOLDER}/:${DOCKER_FOLDER}/ \
    -v ${CURRENT_FOLDER}/data:${DOCKER_FOLDER}/data \
    -v ${DATASETS_PATH}/flickr30k/flickr30k_entities/:${DOCKER_FOLDER}/data/flickr30k/flickr30k_entities \
    -v ${DATASETS_PATH}/flickr30k/flickr30k_images/:${DOCKER_FOLDER}/data/flickr30k/flickr30k_images \
    $VERSION \
    $CMD
elif [[ $MODE == "interactive" ]]; then
  docker run \
    -v $CURRENT_FOLDER/:${DOCKER_FOLDER}/ \
    -u ${USER}:${USER_GROUP}\
    --runtime=nvidia \
    -it \
    $VERSION \
    '/bin/bash'
else
  echo "To be implemented."
fi