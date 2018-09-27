#!/bin/sh

# Wrapper for the Qserv deploy container
# Check for needed variables
# @author Benjamin Roziere <benjamin.roziere@clermont.in2p3.fr>

set -e

IMAGE="activelearning"

DIR=$(cd "$(dirname "$0")"; pwd -P)

usage() {
    cat << EOD

Usage: `basename $0` [options] [cmd]

  Available options:
    -h          this message

  Run a docker container with all the Active Learning tools inside.

EOD
}

# get the options
while getopts h c ; do
    case $c in
        h) usage ; exit 0 ;;
        \?) usage ; exit 2 ;;
    esac
done
shift $(($OPTIND - 1))

if [ $# -ge 2 ] ; then
    usage
    exit 2
elif [ $# -eq 1 ]; then
    CMD=$1
elif [ $# -eq 0 ]; then
    CMD="bash"
fi


MOUNTS="-v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro"
MOUNTS="$MOUNTS -v ${DIR}/data:/data"

echo "Starting Active Learning"

if [ "$AL_DEV" = true ]; then
    echo "Running in development mode"
    MOUNTS="$MOUNTS -v $DIR/rootfs/app:/app"
fi

docker run -it --net=host --rm -l \
    --user=$(id -u):$(id -g $USER) $MOUNTS \
    "$IMAGE" $CMD
