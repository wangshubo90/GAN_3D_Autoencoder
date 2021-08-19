#!/bin/bash
source setENV.sh
DOCKER_IMAGE=horovod

DOCKER_Run_Name=uctgan
WORK_SPACE=$WORK_SPACE
DATA_DIR=$DATA_DIR
jnotebookPort=8888
tensorboard=6006
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

extraFlag="-it -d"
cmd2run="/bin/bash"

extraFlag=${extraFlag}" -p "${jnotebookPort}":8888 -p "${tensorboard}":6006"
echo ${extraFlag}
echo -----------------------------------
docker run --rm ${extraFlag} \
  --name=${DOCKER_Run_Name} \
  --gpus ${GPU_IDs} \
  -v ${WORK_SPACE}:/uCTGan \
  -v ${DATA_DIR}:/uCTGan/data \
  -w /uCTGan \
  --runtime=nvidia \
  --shm-size=9g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}

echo -- exited from docker image