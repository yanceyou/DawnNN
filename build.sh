#!/bin/bash

SCRIPT_PATH="$(cd "$(dirname "$0")"; pwd -P)"

cd ${SCRIPT_PATH}

cmake --preset default
if [ $? -ne 0 ]; then
  printf "[build] preset >>> default <<< error"
  exit 1
fi

cmake --build --preset default
if [ $? -ne 0 ]; then
  printf "[build] preset >>> default <<< error"
  exit 1
fi
