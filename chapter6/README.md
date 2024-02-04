# 6. Deep Learning in Object Detection
## Training Object Detection Model Using TensorFlow
We are now prepared to write code to build and train our own object detection models. We will use the TensorFlow API and write code in Python. Object detection models are very compute-intensive and require a lot of memory and a powerful processor. Most general-purpose laptops or computers may not be able to handle the computations necessary to build and train an object detection model. **For example, a MacBook Air with 32GB RAM and an eight-core CPU is not able to train a detection model involving about 7,000 images**. Thankfully, Google provides a limited amount of GPU-based computing for free. It has been proven that these models run many folds faster on a GPU than on a CPU. Therefore, it is important to learn how to train a model on a GPU. For the purposes of demonstration and learning, we will use the free version of Google GPU. Let’s first define what our learning objective is and how we want to achieve it.

* Objective: Learn how to train an object detection model using Keras and TensorFlow.

* Dataset: The Oxford-IIIT Pet dataset, which is freely available at https://robots.ox.ac.uk/~vgg/data/pets/. The dataset consists of 37 categories of pets with roughly 200 images for each class. The images have large variations in scale, pose, and lighting. They are already annotated with bounding boxes and labeled.

* Execution environment: We will use Google Colaboratory (https://colab.research.google.com), or Colab for short. We will utilize the GPU hardware accelerator that comes free with Colab. Google Colab is a free Jupyter notebook environment that requires no setup and runs entirely in the cloud. Jupyter notebook is an open source web-based application to write and execute Python programs. To learn more about how to use a Jupyter notebook, visit https://jupyter.org. The documentation is available at https://jupyter-notebook.readthedocs.io/en/stable/. We will learn the Colab notebook environment as we work through the code.

We will train the detection model with TensorFlow 2.12.0 on Google Colab, and after the model is trained, we will download and use it with a local environment running on our laptop.

## Selecting a GPU Hardware Accelerator
Select GPU as the hardware accelerator. Make sure you have Python 3 selected for the runtime type.
* CUDA 11.x
* Tensoflow 2.13.1
* https://github.com/googlecolab/colabtools/issues/4227

## Downloading a Pretrained Model for Transfer Learning
Instead of training a model from scratch, transfer learning allows us to transfer the knowledge learned by the pretrained model to the new problem domain.

Transfer learning in object detection using TensorFlow involves leveraging pretrained models, such as those available in the TensorFlow Model Zoo, to accelerate and improve the training process for object detection tasks. TensorFlow provides several pretrained object detection models, such as the Single Shot MultiBox Detector (SSD) and Faster R-CNN, which are trained on large-scale datasets like Common Objects in Context (COCO) or Open Images.

A collection of SSD-based models pretrained on the COCO 2017 dataset is available at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md. 

To demonstrate our training approach, we’ll utilize transfer learning from the SSD ResNet50 V1 FPN 640x640 (RetinaNet50) model from the following URL:
http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

## Configuring the Object Detection Pipeline
The training pipeline is a configuration file that specifies various settings and hyperparameters for training an object detection model. It contains information about the model architecture, dataset paths, batch size, learning rate, data augmentation, evaluation settings, and more. It is usually written in the Protocol Buffers format (.config file).

The schema for the training pipeline is available in the location ``object_detection/protos/pipeline.proto`` in the ``research`` directory under the TensorFlow models project.

The JSON-formatted training pipeline is broadly divided into five sections, as shown here:
```config
model: {
        (... Add model config here...)
}
train_config : {
        (... Add train_config here...)
}
train_input_reader: {
        (... Add train_input configuration here...)
}
eval_config: {
        (... Add eval_configuration here...)
}
eval_input_reader: {
        (... Add eval_input configuration here...)
}
```
The sections of the training pipeline is explained below:
* model: This section defines the architecture and configuration of the object detection model.
    * ssd or faster_rcnn: The type of model (Single Shot MultiBox Detector or Faster R-CNN)
* train_config: This section includes various settings related to the training process.
    * batch_size: The number of images used in each training batch
    * fine_tune_checkpoint: Path to the pretrained model checkpoint to initialize the model before training
    * num_steps: The number of training steps to be performed during the training process
    * data_augmentation_options: Options for data augmentation during training (e.g., random brightness, rotation, etc.)
* train_input_reader: This section includes the configuration for the training data input.
    * tf_record_input_reader: Path to the training TFRecord file (train.tfrecord)
    * label_map_path: Path to the label_map.pbtxt file
* eval_config: This section includes the settings for the evaluation process.
    * num_examples: The number of examples to be used for evaluation
    * max_evals: Maximum number of evaluations to perform
* eval_input_reader: This section includes the configuration for the evaluation data input.
    * tf_record_input_reader: Path to the evaluation TFRecord file (eval.tfrecord)
    * label_map_path: Path to the label_map.pbtxt file

Notice the file ``pipeline.config`` in the pretrained model’s directory, ``ssd_resnet50_v1_fpn_640x640_coco17_tpu-8``. We need to edit the ``pipeline.config`` file to align the training settings accordingly. Download the ``pipeline.config`` file (right-click and Download) from the Colab, save it in the local computer drive, and edit it to configure the training pipeline for the model.

Let’s examine the modifications that need to be made to the ``pipeline.config`` file, indicated by highlighting them in bold. The comments next to the lines that require editing offer further guidance on the specific values to be configured based on the given situation.

```config
model {
  ssd {
    num_classes: 37 # Set this to the number of different label classes
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "ssd_resnet50_v1_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00039999998989515007
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }
      }
      override_base_feature_extractor_hyperparams: true
      fpn {
        min_level: 3
        max_level: 7
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5  # Increase this for a better match.
        unmatched_threshold: 0.5 # Sum of matched and unmatched threshold must be equal to 1.
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00039999998989515007
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.009999999776482582
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }
        }
        depth: 256
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          gamma: 2.0
          alpha: 0.25
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }
}
train_config {
  batch_size: 64  # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03999999910593033
          total_steps: 25000
          warmup_learning_rate: 0.013333000242710114
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "/content/pre-trained-model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pretrained model
  num_steps: 25000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
  use_bfloat16: false # Set this to false if you are not training on a TPU
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "/content/models/research/object_detection/data/pet_label_map.pbtxt"  # Path to label map file
  tf_record_input_reader {
    input_path: "/content/computer_vision/petdata/tfrecords/pet_faces_train.record-?????-of-00010"  # Path to training TFRecord file
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "/content/models/research/object_detection/data/pet_label_map.pbtxt" # Path to label map file
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "/content/computer_vision/petdata/tfrecords/pet_faces_val.record-?????-of-00010"  # Path to testing TFRecord 
  }
}

```
* As the ``pipeline.config`` file was preserved during the training of the pretrained model we downloaded for transfer learning, we will retain most of its contents intact, making alterations only to the specific highlighted parts. The parameters that require adjustment, based on the settings in your Colab environment, are as follows:

* ``num_classes: 37``: Represents the 37 categories of pets in our dataset.

* ``fine_tune_checkpoint``: "/content/pre-trained-model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0": This is the path where we stored the pretrained model checkpoint. Notice in Figure 6-27 that the file name of the model checkpoint is model.ckpt.data-00000-of-00001, but in the fine_tune_checkpoint configuration we provide only up to model.ckpt (you must not include the full name of the checkpoint file). To get the path of this checkpoint file, in the Colab file browser, right-click the file name and click Copy Path.

* ``num_steps: 25000``: This is the number of steps the algorithm should execute. You may need to tune this number to get a desirable accuracy level.

* ``fine_tune_checkpoint_type: "detection"``: By default, this is set to "classification" but needs to be changed to "detection" for object detection training.

* ``use_bfloat16: false``: This is set to true by default for training the model on TPU-based hardware.

* ``train_input_reader → label_map_path: /content/computer_vision/models/research/object_detection/data/pet_label_map.pbtxt``: This is the path of the file that contains the mapping of ID and class name. For the pet dataset, this is available in the research directory.

* ``train_input_reader → input_path: /content/computer_vision/petdata/pet_faces_train.record-?????-of-00010``: This is the path of the TFRecord file for the training dataset. Notice that we used a regular expression (?????) in the training set path. This is important to include all training TFRecord files.

* ``eval_input_reader → label_map_path: /content/computer_vision/models/research/object_detection/data/pet_label_map.pbtxt``: This is the same as the training label map.

* ``eval_input_reader → input_path: /content/computer_vision/petdata/pet_faces_eval.record-?????-of-00010``: This is the path of the TFRecord file for the evaluation dataset. Notice that we used a regular expression (?????) in the evaluation set path. This is important to include all evaluation TFRecord files.

It is important to note that ``pipeline.config`` has the parameter ``override_base_feature_extractor_hyperparams`` set to true; do not change this setting or else the model will not run.

After editing the ``pipeline.config`` file, we need to upload it to Colab. We can upload it to any directory location, but in this case, we are uploading it to its original location from where we downloaded it. We will first remove the old ``pipeline.config`` file and then upload the updated one.

To delete the old ``pipeline.config`` file from the Colab directory location, right-click it and then click Delete. To upload the updated ``pipeline.config`` file from your local computer, right-click the Colab directory (ssd_resnet50_v1_fpn_640x640_coco17_tpu-8), click Upload, and browse and upload the edited version of the ``pipeline.config`` file from the local computer.

## Detecting Objects Using Trained Models
The following are the general steps for crafting the object detection program, covered in depth in the next two sections:
* Download and install the TensorFlow models project from the GitHub repository.
* Write the Python code that will utilize the exported TensorFlow graph (exported model) to predict objects within new images that were not included in the training or test sets.

### Installing TensorFlow’s models 
* Install the libraries that are needed to build and install the models project. 
```
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
```
* Install Google’s Protobuf compiler
```bash
sudo apt install protobuf-compiler
```
* Clone the TensorFlow models project from GitHub:
```bash
git clone --depth 1 https://github.com/tensorflow/models.git
```
* Compile the models project using the Protobuf compiler. Run the following set of commands from the models/research directory:
```bash
$ cd models/research
$ protoc object_detection/protos/*.proto --python_out=.
```

* If you installed Protobuf manually and unzipped it in a directory, provide the full path up to bin/protoc in the previous command.

* Set the following environment variables. It’s a standard practice to set these environment variables in ~/.bash_profile. Here are the instructions to do that:
  * Open your command prompt or terminal and type vi ~/.bash_profile. You can use any other editor, such as nano, to edit the .bash_profile file.
Add the following three lines at the end of .bash_profile. Make sure the paths match with the directory paths you have in your computer.
    ```bash
    export PYTHONPATH=$PYTHONPATH:/vagrant/chapter6/models/research/object_detection
    export PYTHONPATH=$PYTHONPATH:/vagrant/chapter6/models/research
    export PYTHONPATH=$PYTHONPATH:/vagrant/chapter6/models/research/slim
    ```
  * Save the file ~/.bash_profile after adding the previous line.
  * Close your terminal and relaunch it to effect the change. 
     You will need to close your PyCharm IDE to have the environment variables update in your IDE.
     To test the setting, type the command echo $PYTHONPATH in your PyCharm terminal window. 
     It should print the paths you just set up.
* Build and install the research project that we just built using Protobuf. Execute the following commands from the models/research directory:
    ```bash
    cd /vagrant/chapter6/models/research/
    cp /vagrant/chapter6/models/research/object_detection/packages/tf2/setup.py .
    python setup.py build
    python setup.py install
    ```
    or new command
    ```bash
    cd /vagrant/chapter6/models/research/
    cp /vagrant/chapter6/models/research/object_detection/packages/tf2/setup.py .
    pip wheel --no-deps -w dist .
    pip install object-detection
    ```
    or another new command
    ```bash
    cd /vagrant/chapter6/models/research/
    cp /vagrant/chapter6/models/research/object_detection/packages/tf2/setup.py .
    python -m pip install .
    ```
    If these commands successfully run, you should output something like this:
    ```bash
    Finished processing dependencies for object-detection==0.1
    ```
* Our environment is now prepared to begin coding for image object detection. The model we’ll be working with is the one we exported and downloaded from Colab. If you haven’t completed this step yet, download the final model either from Google Colab or from Google Drive, assuming you stored your models there.

## Configuration and Initialization
In this section of the code, we initialize the model path, image input, and output directories. **Listing_6-15.py** shows the first part of the code that includes the library imports and path setup.

## Training a YOLOv7 Model for Object Detection
* The official YOLOv7 source code: https://github.com/WongKinYiu/yolov7
### Dataset
We utilize an already labeled dataset consisting of images of individuals wearing safety helmets and reflective jackets. The dataset is available freely at the following URL:
``https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket/download?datasetVersionNumber=1``

The dataset is also available at the following Google Drive link:
``https://drive.google.com/file/d/1MdlK9lvwWXHQgRcsjZFcteA5oiFb0jBy/view?usp=sharing``

1. Within a parent directory, there are three subdirectories: **train**, **test**, and **valid**.
2. Each of these subdirectories contains two additional subdirectories inside them: **images** and **labels**.
3. The **images** subdirectory contains image files (e.g., .png, .jpg, and .jpeg).
4. The **labels** subdirectory contains annotations in text files, one annotation text file per image. The file names in the labels directory must be the same as the image file names, except that it has the extension .txt. For example, if the image file name is **helmet_jacket_07350.jpg**, the annotation file name in the **labels** directory must be **helmet_jacket_07350.txt**.
5. The text file contains one or more lines of entries. Each line in the text file represents an object annotation. Multiple lines in the file means that the image contains multiple objects. Each line in the annotation text file must contain the annotated bounding box and object class in one single line in the following format:

``<object-class> <x_center> <y_center> <width> <height>``

where

``<object-class>`` is the integer class index of the object, from 0 to ``(num_class-1)``.

``<x_center>`` and ``<y_center>`` are float values representing the center of the bounding boxes relative to the image height and width.

``<width>`` and ``<height> ``are the width and height of bounding boxes relative to the image height and width.

Note that the line entries in this file are separated by blank spaces and not by commas or any other delimiters.

The parent directory ``safety-Helmet-Reflective-Jacket`` contains subdirectories: **train**, **test**, and **valid**. Within each, the **images** subdirectories contain image files used in training, testing, and validation and the **labels** subdirectories contain annotation text files. Figure 6-35 shows an example annotation with two objects in class 0 and class 1 and corresponding bounding boxes. Note that the coordinates of the bounding boxes are relative to the image size: the x-coordinate is divided by the image width, and the y-coordinate is divided by the image height. This makes the values of the bounding box coordinates range between 0 and 1.

### Preparing Colab Environment

#### Training on a Single GPU
Listing 6-21 shows how to train a YOLOv7 model from scratch on a single GPU.
```
%%shell
cd /content/yolov7
python train.py \
--epochs 10  \
--workers 8 \
--device 0 \
--batch-size 16 \
--data /content/safety-Helmet-Reflective-Jacket/data.yaml \
--img 640 640 \
--cfg cfg/training/yolov7.yaml \
--weights '' \
--name yolov7-ppe \
--hyp data/hyp.scratch.p5.yaml
```
**--epochs 10**: This indicates the training should run 10 epochs.

**--workers 8**: This sets the number of data-loading worker processes to 8. These processes help load and preprocess training data efficiently.

**--device 0**: This flag specifies which GPUs to use for training. In this case, a single GPU 0 is selected.

**--batch-size 16**: This sets the batch size to 16 images per iteration.

**--data /content/safety-Helmet-Reflective-Jacket/data.yaml**: This flag points to the YAML file containing dataset configuration details.

**--img 640 640**: This defines the input image size as 640×640 pixels.

**--cfg cfg/training/yolov7.yaml**: This specifies the model configuration file (YAML) for YOLOv7 architecture.

**--weights ''**: This indicates that no pretrained weights are used for the model. Training will start from scratch. If the weights parameter is left blank, as in this example, it indicates training from scratch. Providing a file path to pr-existing weights indicates transfer learning.

**--name yolov7-ppe**: This provides a name for the training session, which can be used to save checkpoints and logs. By default, the model is stored within the runs/train subdirectory of the project directory (yolov7 in our example). In our example, the model will be stored in /content/drive/MyDrive/PPE/yolov7/runs/train/yolov7-ppe directory. If we run the training the second time, the model will get created in the directory /content/drive/MyDrive/PPE/yolov7/runs/train/yolov7-ppe2, and so on.

**--hyp data/hyp.scratch.p5.yaml**: This points to the hyperparameters file in YAML format.


we have to modify  ``/content/safety-Helmet-Reflective-Jacket/data.yaml``
```yaml
names:
  - 'Safety-Helmet'
  - 'Reflective-Jacket'
nc: 2
test: ../safety-Helmet-Reflective-Jacket/test/images
train: ../safety-Helmet-Reflective-Jacket/train/images
val: ../safety-Helmet-Reflective-Jacket/valid/images

```
#### Training on Multiple GPUs
Training an object detection model demands substantial computational resources. When dealing with a large training dataset containing high-resolution images, it might be necessary to leverage multiple GPUs to accelerate the learning procedure.

By passing a few additional parameters into Listing 6-21, we can enhance the training process to utilize multiple GPUs. Listing 6-22 highlights these supplementary parameters in bold for clarity.
```
%%shell
cd /content/yolov7
 python \
-m torch.distributed.launch \
--nproc_per_node 4 \
--master_port 9527 \
train.py \
--epochs 100  \
--workers 8 \
--device 0,1,2,3 \
--sync-bn \
--batch-size 16 \
--data /content/safety-Helmet-Reflective-Jacket/data.yaml \
--img 640 640 \
--cfg cfg/training/yolov7.yaml \
--weights '' \
--name yolov7-ppe \
--hyp data/hyp.scratch.p5.yaml
```
**python -m torch.distributed.launch**: This part of the command is invoking the Python interpreter with the torch.distributed.launch module, which facilitates distributed training across multiple nodes or GPUs.

**--nproc_per_node 4**: This specifies the number of processes (GPUs) per node to be utilized for training. In this case, it’s set to four GPUs.

**--master_port 9527**: This parameter defines the communication port used by the master process for synchronization during distributed training.

**train.py**: This is the name of the Python script responsible for initiating the training process.

**--workers 8**: This sets the number of data-loading worker processes to 8. These processes help load and preprocess training data efficiently.

**--device 0,1,2,3**: This flag specifies which GPUs to use for training. In this case, GPUs 0, 1, 2, and 3 are selected.

**--sync-bn**: This indicates the usage of synchronized batch normalization, which can improve convergence during distributed training.

**--batch-size 16**: This sets the batch size to 16 images per iteration.

**--data /content/safety-Helmet-Reflective-Jacket/data.yaml**: This flag points to the YAML file containing dataset configuration details.

**--img 640 640**: This defines the input image size as 640×640 pixels.

**--cfg cfg/training/yolov7.yaml**: This specifies the model configuration file (YAML) for YOLOv7 architecture.

**--weights ''**: This indicates that no pretrained weights are used for the model. Training will start from scratch.

**--name yolov7-ppe**: This provides a name for the training session, which can be used to save checkpoints and logs.

**--hyp data/hyp.scratch.p5.yaml**: This points to the hyperparameters file in YAML format.


####  Monitoring Training Metrics Using TensorBoard
Listing 6-23 shows the command to launch TensorBoard by pointing it to the log directory. Ideally, you should launch TensorBoard from a different instance of Colab than the instance that is executing the training code. This will allow you to monitor the training progress while the training is running, and not after the training is completed.

```
%load_ext tensorboard
%tensorboard --logdir /content/yolov7/runs/train
```

### Inference or Object Detection Using the Training YOLOv7 Model
The final weights file is stored in the directory ``runs/train/yolov7-ppe/weights/best.pt``. The command in Listing 6-24 shows how to use the trained model to detect objects within images. Listing 6-25 shows how to detect objects in a video file.
```
%%shell
cd /content/drive/MyDrive/PPE/yolov7
python detect.py \
--project /content/detection \
--weights /content/yolov7/runs/train/yolov7-ppe3/weights/best.pt \
--conf 0.25 --img-size 640 \
--source /content/safety-Helmet-Reflective-ConstructionWorkers.png
```
Listing 6-24 Launching TensorBoard to Monitor the Training Metrics of YOLO

```
%%shell
cd /content/yolov7
python detect.py \
--project /content/detection \
--weights /content/yolov7/runs/train/yolov7-ppe3/weights/best.pt \
--conf 0.25 \
--img-size 640 \
--source /content/construction-site.mp4
```
Listing 6-25 Launching TensorBoard to Monitor the Training Metrics

In the preceding two examples, the argument ``--project`` takes a directory path where the prediction output is stored. If this argument is omitted, the output is stored in the project’s ``runs/detect/exp directory``.

#### Exporting YOLOv7 Model to ONNX
ONNX, which stands for Open Neural Network Exchange, is an open and interoperable format for representing and exchanging deep learning models between various frameworks and tools.

The main idea behind ONNX is to provide a common standard for representing machine learning models, regardless of the framework they were trained in. This standardization makes it easier to deploy and use models across different platforms and frameworks, which is particularly valuable in scenarios where different parts of a workflow might involve different tools or environments.

ONNX supports a wide range of deep learning frameworks, including TensorFlow, PyTorch, Caffe, and more. With ONNX, we train a model in one framework and then export it to ONNX format. Once in the ONNX format, we can import the model into another supported framework, such as TensorFlow, for inference or further development without needing to retrain it.

In addition to the model format, ONNX also includes a runtime that allows us to load and run ONNX models on different devices, making it easier to deploy models on edge devices, mobile devices, or cloud servers.

To export the YOLOv7 model to ONNX, we need to install the onnx module from pypi. In Google Colab, open a new code block and simply use !pip install onnx to install the ONNX runtime.

The code block in Listing 6-26 exports the YOLOv7 model to ONNX and stores it in the same location where the best.pt model is located.

```
%%shell
cd /content/yolov7
python export.py \
--weights /content/yolov7/runs/train/yolov7-ppe3/weights/best.pt \
--grid --end2end --simplify \
--topk-all 100 --iou-thres 0.65 \
--conf-thres 0.35 \
--img-size 640 640 --max-wh 640
```
#### Converting the ONNX Model to TensorFlow and TensorFlow Lite Formats
We need to install the onnx-tf and tensorflow-probability modules in order to convert the ONNX model into TF and TFLite formats. Listing 6-27 shows the command to install these modules.
```
!pip install onnx-tf tensorflow-probability
```
Listing 6-27 Installing Packages Needed for Conversion from ONNX to TF and TFLite

Listing 6-28 shows the code segment that actually converts the ONNX model to TensorFlow format.
```
%%shell
onnx-tf convert \
-i /content/drive/MyDrive/PPE/yolov7/runs/train/yolov7-ppe3/weights/best.onnx \
-o yolov7-tf
```
Listing 6-28 Code to Convert ONNX to TensorFlow

The argument -i takes the ONNX file path. The argument -o specifies the output location where the TensorFlow model will be saved. In Listing 6-28, the TF model will be stored in the directory yolov7-tf in the current working directory.

TensorFlow Lite is a lightweight version of the TensorFlow deep learning framework developed by Google. It’s specifically designed for deploying machine learning models on resource-constrained devices, such as smartphones, embedded devices, microcontrollers, and other edge devices.

Listing 6-29 shows a Python code snippet that converts a TensorFlow model into TensorFlow Lite format.
```
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model('/content/yolov7-tf/')
tflite_model = converter.convert()

with open('/content/tflite/yolov7-tiny.tflite', 'wb') as f:
  f.write(tflite_model)
```
Listing 6-29 Code to Convert TensorFlow to TensorFlow Lite


After executing this code, the YOLOv7 model will get converted into TensorFlow Lite format.

In the following section, we will explore how to predict object detection from the TensorFlow Lite model.

#### Predicting Using TensorFlow Lite Model
Listing 6-30 showcases the code used for detecting objects using a TensorFlow Lite model that was exported from the YOLO model.
```python
1    import cv2
2    import random
3    import numpy as np
4    from PIL import Image
5    import tensorflow as tf
6    from google.colab.patches import cv2_imshow
7    import numpy as np
8    # Load the TFLite model
9    interpreter = tf.lite.Interpreter(model_path="/content/tflite/yolov7-tiny.tflite")
10
11   def resize_image(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
12       # Resize and pad image while meeting stride-multiple constraints
13       shape = im.shape[:2]  # current shape [height, width]
14       if isinstance(new_shape, int):
15           new_shape = (new_shape, new_shape)
16
17       # Scale ratio (new / old)
18       r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
19       if not scaleup:  # only scale down, do not scale up (for better val mAP)
20           r = min(r, 1.0)
21
22       # Compute padding
23       new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
24       dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
25
26       if auto:  # minimum rectangle
27           dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
28
29       dw /= 2  # divide padding into 2 sides
30       dh /= 2
31
32       if shape[::-1] != new_unpad:  # resize
33           im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
34       top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
35       left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
36       im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
37       return im, r, (dw, dh)
38
39   #Name of the classes according to class indices.
40   names = ['safety-Helmet','Reflective-Jacket']
41
42   #Creating random colors for bounding box visualization.
43   colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
44
45   #Load and preprocess the image.
46   img = cv2.imread('/content/CosntructionWorkers.png')
47   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
48
49   image = img.copy()
50   image, ratio, dwdh = resize_image(image, auto=False)
51   image = image.transpose((2, 0, 1))
52   image = np.expand_dims(image, 0)
53   image = np.ascontiguousarray(image)
54
55   im = image.astype(np.float32)
56   im /= 255
57
58   #Allocate tensors.
59   interpreter.allocate_tensors()
60   # Get input and output tensors.
61   input_details = interpreter.get_input_details()
62   output_details = interpreter.get_output_details()
63
64   # Test the model on random input data.
65   input_shape = input_details[0]['shape']
66   interpreter.set_tensor(input_details[0]['index'], im)
67
68   interpreter.invoke()
69
70   # The function `get_tensor()` returns a copy of the tensor data.
71   # Use `tensor()` in order to get a pointer to the tensor.
72   output_data = interpreter.get_tensor(output_details[0]['index'])
73
74   ori_images = [img.copy()]
75
76   for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(output_data):
77       image = ori_images[int(batch_id)]
78       box = np.array([x0,y0,x1,y1])
79       box -= np.array(dwdh*2)
80       box /= ratio
81       box = box.round().astype(np.int32).tolist()
82       cls_id = int(cls_id)
83       score = round(float(score),3)
84       name = names[cls_id]
85       color = colors[name]
86       name += ' '+str(score)
87       cv2.rectangle(image,box[:2],box[2:],color,2)
88       cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
89
90   prediction = Image.fromarray(ori_images[0])
91
92   open_cv_image = np.array(prediction)
93   # Convert RGB to BGR
94   open_cv_image = open_cv_image[:, :, ::-1].copy()
95   cv2_imshow(open_cv_image)
```