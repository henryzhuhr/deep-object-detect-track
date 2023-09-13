export ENV_DIR=$PWD/.env/deep-obj
if [ ! -d $ENV_DIR ];then
    python3 -m venv $ENV_DIR
fi
source $ENV_DIR/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
