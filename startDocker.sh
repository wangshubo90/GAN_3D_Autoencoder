#!/bin/bash

DOCKER_IMAGE=horovod

DOCKER_Run_Name=uctgan
WORK_SPACE=$HOME/repos/GAN_3D_Autoencoder
DATA_DIR=$HOME/data
jnotebookPort=8888
GPU_IDs=all

#################################### check if name is used then exit
docker ps -a|grep ${DOCKER_Run_Name}
dockerNameExist=$?
if ((${dockerNameExist}==0)) ;then
  echo --- dockerName ${DOCKER_Run_Name} already exist
  echo ----------- attaching into the docker
  docker exec -it ${DOCKER_Run_Name} /bin/bash
  exit
fi

echo -----------------------------------
echo starting docker for ${DOCKER_IMAGE} using GPUS ${GPU_IDs} jnotebookPort ${jnotebookPort} 
echo -----------------------------------

extraFlag="-it "
cmd2run="/bin/bash"

extraFlag=${extraFlag}" -p "${jnotebookPort}":8888"

docker run --rm ${extraFlag} \
  --name=${DOCKER_Run_Name} \
  --gpus ${GPU_IDs} \
  -v ${WORK_SPACE}:/uCTGan \
  -v ${DATA_DIR}:/uCTGan/data \
  -w /uCTGan \
  --runtime=nvidia \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}

echo -- exited from docker image