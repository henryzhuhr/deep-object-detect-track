export OpenVINO_HOME="$HOME/program/openvino-2023_release"
source $OpenVINO_HOME/setupvars.sh

BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"
rm CMakeCache.txt
cmake .. -G Ninja

if [ "$(uname)" = "Darwin" ]; then
    NUM_CORES=`sysctl -n hw.ncpu`
elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
    NUM_CORES=`nproc --all`
else
    NUM_CORES=4
fi

ninja -j $NUM_CORES

./infer