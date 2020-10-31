# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"

# Tools config for CUDA, Anaconda installed in the common /tools directory
source /scratch/scratch2/dhruvramani/config.sh
# Activate your environment
source /scratch/scratch2/dhruvramani/miniconda3/envs/py3.6/bin/activate 

xvfb-run -a -s "-screen 0 1400x900x24" bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/storage/home/dhruvramani/.mujoco/mujoco200/bin
export MUJOCO_PY_MUJOCO_PATH=/storage/home/dhruvramani/.mujoco/mujoco200
export MUJOCO_PY_MJKEY_PATH=/storage/home/dhruvramani/.mujoco/mjkey.txt

# Change to the directory in which your code is present
cd /path/to/run.sh
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.
# Here, the code is the MNIST Tensorflow example.
python3 file_name.py &> output.out
