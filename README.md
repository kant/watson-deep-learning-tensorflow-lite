# Deploying Watson Deep Learning Models to Edge Devices

This project includes sample code how to train a model with [TensorFlow](https://www.tensorflow.org/) and the [Deep Learning service](https://www.ibm.com/blogs/watson/2018/03/deep-learning-service-ibm-makes-advanced-ai-accessible-users-everywhere/) within Watson Studio and how to deploy and access the model on iOS devices.

This is a screenshot from the app running on an iPhone where currently a truck is recognized:

![alt text](documentation/ios-camera-app-small.JPEG "Screenshot")

Check out the [video](https://youtu.be/4WTpMmqraXI) for a quick demo.

In order to train the model I've taken pictures from seven items: plug, soccer ball, mouse, hat, truck, banana and headphones. You can find the images in the [data](data/images) directory.

![alt text](documentation/items-small.JPG "Photo")


## Prerequisites 

Get a free [IBM Cloud](https://ibm.biz/nheidloff) lite account (no time restriction, no credit card required).

Create an instance of the [Machine Learning](https://console.bluemix.net/catalog/services/machine-learning) service. From the credentials get the user name, password and the instance id.

Install the IBM Cloud CLI with the machine learning plugin and set environment variables by following these [instructions](https://datascience.ibm.com/docs/content/analyze-data/ml_dlaas_environment.html).

Create an instance of the [Cloud Object Storage
](https://console.bluemix.net/catalog/services/cloud-object-storage) service and create HMAC credentials by following these [instructions](https://datascience.ibm.com/docs/content/analyze-data/ml_dlaas_object_store.html). Make sure to use 'Writer' or 'Manager' access and note the aws_access_key_id and aws_secret_access_key for a later step.

Install and configure the AWS CLI by following these [instructions](https://console.bluemix.net/docs/services/cloud-object-storage/cli/aws-cli.html#use-the-aws-cli).


## Training of the Model

Models can be trained either locally or with IBM Watson in the cloud.

In both cases clone this repo, download MobileNet and set up the environment:

```bash
$ git clone https://github.com/nheidloff/watson-deep-learning-tensorflow-lite
$ cd watson-deep-learning-tensorflow-lite
$ my_project_dir=$(pwd)
$ export PROJECT_DIR=$my_project_dir
$ cd data
$ wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_224.tgz
$ tar xvzf mobilenet_v1_0.25_224.tgz 
$ cp -R ${PROJECT_DIR}/data ${PROJECT_DIR}/volume/data
```

### Training with IBM Watson

Create two buckets (use unique names):

```bash
$ aws --endpoint-url=http://s3-api.dal-us-geo.objectstorage.softlayer.net --profile ibm_cos s3 mb s3://nh-recognition-input
$ aws --endpoint-url=http://s3-api.dal-us-geo.objectstorage.softlayer.net --profile ibm_cos s3 mb s3://nh-recognition-output
```

Upload bucket with MobileNet and data (use your unique bucket name):

```bash
$ cd ${PROJECT_DIR}/data
$ aws --endpoint-url=http://s3-api.dal-us-geo.objectstorage.softlayer.net --profile ibm_cos s3 cp . s3://nh-recognition-input/ --recursive 
```

Prepare the training:
* Define your object storage credentials and your bucket names in [tf-train.yaml](model/tf-train.yaml).
* Compress [retrain.py](model/retrain.py) into [tf-model.zip](model/tf-model.zip) (only necessary if you change this file).

Invoke the training and check for status (change the generated training name):

```bash
$ cd ${PROJECT_DIR}/model
$ bx ml train tf-model.zip tf-train.yaml
$ bx ml list training-runs
$ bx ml monitor training-runs training-CaXai_DmR
$ bx ml show training-runs training-CaXai_DmR
```

Download the saved model:

```bash
$ cd ${PROJECT_DIR}/saved-model
$ aws --endpoint-url=http://s3-api.dal-us-geo.objectstorage.softlayer.net --profile ibm_cos s3 sync s3://nh-recognition-output .
```

Run these commands (replace the training sub directory name):

```bash
$ cp ${PROJECT_DIR}/saved_model/training-CaXai_DmR/graph.pb ${PROJECT_DIR}/volume/training/graph.pb
$ cp ${PROJECT_DIR}/saved_model/training-CaXai_DmR/labels.txt ${PROJECT_DIR}/ios-photos/data/labels.txt
$ cp ${PROJECT_DIR}/saved_model/training-CaXai_DmR/labels.txt ${PROJECT_DIR}/ios-camera/data/labels.txt
```


### Local Training

Run the Docker image:

```bash
$ docker run -v ${PROJECT_DIR}/volume:/volume -it tensorflow/tensorflow:1.7.1-devel bash
```

In the Docker container invoke these commands:

```bash
$ python /volume/retrain.py \
  --bottleneck_dir /volume/training/bottlenecks \
  --image_dir /volume/data/images \
  --how_many_training_steps=1000 \
  --architecture mobilenet_0.25_224 \
  --output_labels /volume/training/labels.txt \
  --output_graph /volume/training/graph.pb \
  --model_dir /volume/data \
  --learning_rate 0.01 \
  --summaries_dir /volume/training/retrain_logs
$ exit
```

Run these commands:

```bash
$ cp ${PROJECT_DIR}/volume/labels.txt ${PROJECT_DIR}/ios-photos/data/labels.txt
$ cp ${PROJECT_DIR}/volume/labels.txt ${PROJECT_DIR}/ios-camera/data/labels.txt
```


### Local Training and Watson Traning

Run the Docker image:

```bash
$ docker run -v ${PROJECT_DIR}/volume:/volume -it tensorflow/tensorflow:1.7.1-devel bash
```

In the Docker container invoke these commands:

```bash 
$ toco \
  --input_file=/volume/training/graph.pb \
  --output_file=/volume/graph.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT
$ exit
```

After exiting the container, run these commands:

```bash
$ cp ${PROJECT_DIR}/volume/graph.lite ${PROJECT_DIR}/ios-photos/data/graph.lite
$ cp ${PROJECT_DIR}/volume/graph.lite ${PROJECT_DIR}/ios-camera/data/mobilenet_quant_v1_224.tflite
```


## Deployment of the iOS Apps

tbd

https://www.tensorflow.org/mobile/tflite/demo_ios

cd ${PROJECT_DIR}/ios-camera
pod install
open tflite_camera_example.xcworkspace

cd ${PROJECT_DIR}/ios-photos
pod install
open tflite_photos_example.xcworkspace