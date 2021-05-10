# Build Tensorflow 2.4.0 with Bazel 3.1.0 libs

### Install Bazel
* pay attention! you must install a bazel version corrisponding to your installed tensorflow release,  check [tested build configurations](https://www.tensorflow.org/install/source#tested_build_configurations)

* Add Bazel distribution URI as a package source as specified in https://docs.bazel.build/versions/master/install-ubuntu.html then
```
sudo apt update && sudo apt install bazel-<needed_version>
sudo ln -s /usr/bin/bazel--<needed_version> /usr/bin/bazel
bazel --version
```

### Build Tensorflow

```
git clone https://github.com/tensorflow/tensorflow.git
git checkout v2.4.0
./configure
bazel build [--config=option] //tensorflow:libtensorflow_cc.so //tensorflow:install_headers
```
