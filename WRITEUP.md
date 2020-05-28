# Depoy a People Counter at Edge 

This project is to detect and count person in the video, We will use person detection model and convert it to an Intermediate Representation for use with the Model Optimizer. 
We will extract useful data concerning the count of people in frame and how long they stay in frame. We will also send data data over MQTT, as well as sending the output frame, in order to view it froma separate UI server over a network.

## Explaining Custom Layers in OpenVINO™:
Layers which are [Supported Layer List](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) are Unsupported Layers, This layers are classified as Custom layer by Model Optimizer.

### Process to handle Custom layers:
We need to handle it according to the framework,

TensorFlow Framework Model we have three ways to handle custom layer,
1. Register the custom layers as extensions to the Model Optimizer
2. Replace the unsupported subgraph with a different subgraph
3. offload the computation of the subgraph back to TensorFlow during inference

Caffe Model we can handle custom layer by Two ways,
1. Register the custom layers as extensions to the Model Optimizer.
2. Register the layers as Custom, then use Caffe to calculate the output shape of the layer

#### Custom Layer Implementation Workflow:

The Model Optimizer starts with a library of known extractors and operations for each supported model framework which must be extended to use each unknown custom layer. The custom layer extensions needed by the Model Optimizer are:

* Custom Layer Extractor: Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.

* Custom Layer Operation: Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters. The --mo-op command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.
    ```
    The script for this is available here-
  /opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py
  
  usage: You can use any combination of the following arguments:

    Arguments to configure extension generation in the interactive mode:
    
    optional arguments:
      -h, --help            show this help message and exit
      --mo-caffe-ext        generate a Model Optimizer Caffe* extractor
      --mo-mxnet-ext        generate a Model Optimizer MXNet* extractor
      --mo-tf-ext           generate a Model Optimizer TensorFlow* extractor
      --mo-op               generate a Model Optimizer operation
      --ie-cpu-ext          generate an Inference Engine CPU extension
      --ie-gpu-ext          generate an Inference Engine GPU extension
      --output_dir OUTPUT_DIR
                            set an output directory. If not specified, the current
                            directory is used by default.
    ```

Detailed Process of handling custom layer can be found in this [documentation](https://docs.openvinotoolkit.org/2019_R3.1/_docs_HOWTO_Custom_Layers_Guide.html)

### Potential reasons for handling custom layer:

Sometimes we are working on some research based project in which we uses models with the layers which are not supported layers of Model Optimizer, For running our application perfectly smooth on Edge we need to know how we can handle such layers.

## Comparing Model Performance

I used model from Intel OpenVino Model Zoo due to poor performance of converted models.  
I have explained the details of models I experimented with.

### Model size

| |SSD MobileNet V2|YOLO V3 Tiny|Faster RCNN Inception V2 COCO|
|-|-|-|-|
|Before Conversion|67 MB|34 MB|55 MB|
|After Conversion|65 MB|32 MB|52 MB|

### Inference Time

| |SSD MobileNet V2|YOLO V3 Tiny|Faster RCNN Inception V2 COCO|
|-|-|-|-|
|Before Conversion|50 ms|45 ms| 336 ms|
|After Conversion|60 ms|50 ms| 345 ms|

## Assess Model Use Cases

### Case 1. **Corona Smart Surveillance integrated with Drone**: 
Countries like india in this pandemic time, movement of people are analysed by drones manually,
If we integrate our People Counter App with it. 
* If more people are gathered this app will generate alert with location indices,
* Police can go to the location and restrict the movement
* This will reduce manual effort of surveillance and save cost 
* This will also reduce human error
* This same application can be used in **curfew time** to restrict movement

### 2. **Smart Queue Management System**: 
Our project can be used in smart queue system,
* It can help to Optimize Products and Employees based on length of queue,
* Products and Employee optimization can lead to higher revenue,
* Management person can easily manage remotely if such system is used

### 3. **Security Application in Industry for safety**: 
In many industries human movements are restricted in some toxic area,
* Our application can be deployed in such area. if any human detected siren will be ON.
* This is the AI that can save life.

### 4. **Road Safety Surveillance**: 
This application can be deployed to computer with traffic surveillance camera,
*  We can count number of people in the vehicles by training model on this real dataset, If count of people is more than safety limit of vehicle, fine will be sent to the owner of vehicle. 
*  This will help us to force people follow rules of traffic

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model:

Lighting: Different lighting condition will surely impact our application. It also can lead to zero detection. To tackle this problem, we can do some image pre-processing steps before feeding images to model for inference

Model Accuracy: We will not get desired accuracy for all the times. We might need need to train model according to application. This custom model tested for custom application will surely works better because it is trained on real life dataset.

Camera focal length/image size: Different image size can be handle using resizing the image step while preprocessing images before inference but again this might lead to lead to less accuracy. We can use model trained for that image size  or use custom model trained for specific application
 

## Model Research

In investigating potential people counter models, I tried each of the following three models:

- ### Model 1.  [SSD MobileNet V2](https://github.com/opencv/open_model_zoo/blob/master/models/public/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.md):
  - I have downloaded this model from Tensorflow site with following command in the terminal,
      ```
      #download model
      wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
      #extracted using below commanad
      tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
      ```
  - I converted the model to an Intermediate Representation with the following command,
      ```
      python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
      ```
  - The model was insufficient for the app because it was giving me around 180 missing frames for one person, detection accuracy was very acceptable.
  - I tried to improve count detection using techniques like detecting person only when they enter in the frame from one side but due to low accuracy detection was not proper.  
  
- ### Model 2. [YOLO V3 Tinyll](https://github.com/opencv/open_model_zoo/blob/master/models/public/yolo-v3-tf/yolo-v3-tf.md):
  I have followed the [official documentation](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) for using YOLOV3 by openvinotoolkit.org
  Here are the steps,
  1. Clone the Repo: 
        ```
        git clone https://github.com/mystic123/tensorflow-yolo-v3.git
        cd tensorflow-yolo-v3
        ```
  2. checkout tested commit:
      ` git checkout ed60b90`

  3. Download weights and labels:
      ```
      wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
      wget https://pjreddie.com/media/files/yolov3-tiny.weights 
      ```
  4. Run Converter: to get weights in to pb file
      ```
      python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
      ```
  5. Convert YOLOv3 TensorFlow Model to the IR:
      ```
      python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py  --input_model yolov3.pb --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/yolov3.json --batch 1
      ```
    
  - Model Accuracy was not enough, i was getting around almost same detection with bit higher speed
  - I tried to get proper count by skipping 160 missing frame when that person in black tshirt was not detected, i was getting proper count but that will not be acceptable because this might not be general solution  

- ### Model 3. [Faster RCNN Inception V2 COCO](https://github.com/opencv/open_model_zoo/blob/master/models/public/faster_rcnn_inception_v2_coco/faster_rcnn_inception_v2_coco.md):
  - I have downloaded this model from Tensorflow site with following command in the terminal,
      ```
      #download model
     wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
      #extracted using below commanad
      tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
      ```
  - I converted the model to an Intermediate Representation with the following command,
      ```
      python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
      ```
  - The model was insufficient for the app because of very low inference speed and less accuracy
  - This model was really slow, i tried to skip inference for few frames and tried to improve performance but again accuracy beat me

## Use of Intel Pretrained model:
As we were not getting proper results using other models, I tried following two model from OpenVino Model Zoo,
- [person-detection-retail-0002](https://docs.openvinotoolkit.org/latest/person-detection-retail-0002.html)
- [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)

For second one, i was getting higher accuracy,

To download the model i used following code in the terminal:
```
sudo /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 --precisions FP16  -o /home/workspace
``` 
 
To Run the Inference on Video i use following command in the terminal:
```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```