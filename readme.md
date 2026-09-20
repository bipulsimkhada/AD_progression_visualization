# registering gpu to tf

1. follow tf installation: https://www.tensorflow.org/install/pip#linux_1
2. add following path to .venv/bin/activate

export LD_LIBRARY_PATH=$(python3 -c "import site, os; p = site.getsitepackages()[0] + '/nvidia'; print(':'.join([os.path.join(p, d, 'lib') for d in os.listdir(p) if os.path.isdir(os.path.join(p, d, 'lib'))]))"):$LD_LIBRARY_PATH


# to test
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"