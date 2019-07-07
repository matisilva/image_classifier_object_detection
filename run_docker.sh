#!/bin/bash

docker run -it \
	--name no_molestar \
    --privileged \
    --rm \
    --runtime=nvidia \
    --log-driver none \
    -v $(pwd):/src/ \
    -w /src \
    despegar:mariano \
    bash
