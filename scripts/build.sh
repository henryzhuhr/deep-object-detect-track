export OpenVINO_HOME="$HOME/program/openvino-2023_release"
source $OpenVINO_HOME/setupvars.sh




BUILD_DIR="build"

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"
rm CMakeCache.txt
cmake ..
make

./infer