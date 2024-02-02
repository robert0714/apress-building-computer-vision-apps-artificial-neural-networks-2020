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