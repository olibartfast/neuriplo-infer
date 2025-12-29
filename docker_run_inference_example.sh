#!/bin/bash
docker run --gpus=all --rm \
    -v $(pwd)/data:/app/data \
    -v $HOME/repos/D-FINE/exports:/weights \
    -v $(pwd)/labels:/labels \
    vision-inference:tensorrt \
    --type=dfine \
    --weights=/weights/model.engine \
    --source=/app/data/dog.jpg \
    --labels=/labels/coco.names \
    --input_sizes='3,640,640;2' 


# docker run --rm \
#     -v data:/app/data \
#     -v $HOME/repos/D-FINE:/weights \
#     -v labels:/labels \
#     -w /app/data \
#     --entrypoint ls \
#     vision-inference:tensorrt  \
#     -alt /app/app
