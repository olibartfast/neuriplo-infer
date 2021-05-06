# Build Tensorflow 2.4.0 with Bazel 3.1.0 libs

### Install python virtual env
```
sudo apt-get install python3-venv
cd
python3 -m venv --system-site-packages ./tensorflow_venv
```
* then to activate virtual env:
```
source tensorflow_venv/bin/activate
```

### Install Bazel

### Build Tensorflow

```
git clone https://github.com/tensorflow/tensorflow.git
git checkout v2.4.0

```