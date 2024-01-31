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