import tensorflow as tf
import os
from tensorflow.python.ops.numpy_ops import np_config
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
import json

WORKSPACE_PATH = 'Identification_Bills-Prueba/Tensorflow/workspace'
SCRIPTS_PATH = 'Identification_Bills-Prueba/Tensorflow/scripts'
APIMODEL_PATH = 'Identification_Bills-Prueba/Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet'


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-6')).expect_partial()

tf.config.run_functions_eagerly(True)
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detection = detection_model.postprocess(prediction_dict, shapes)
    #detections.save('exported-models/my_tflite_model/saved_model')
    if isinstance(detection, dict):
        # Manejar detecciones como diccionario
        num_detections = int(detection['num_detections'].numpy())
        detection_classes = detection['detection_classes'].numpy().tolist()
        detection_scores = detection['detection_scores'].numpy().tolist()
        detection_boxes = detection['detection_boxes'].numpy().tolist()


    # Guardar detecciones como JSON
    detections = {
        'num_detections': num_detections,
        'detection_classes': detection_classes,
        'detection_scores': detection_scores,
        'detection_boxes': detection_boxes
    }

    #detections = {
    #    'num_detections': int(detection['num_detections']),
    #    'detection_classes': detection['detection_classes'].numpy().tolist(),
    #    'detection_scores': detection['detection_scores'].numpy().tolist(),
    #    'detection_boxes': detection['detection_boxes'].numpy().tolist()
    #}
    with open('detection.json', 'w') as json_file:
        json.dump(detections, json_file)
    #if not isinstance(detections, dict):
    #    print("Se esperaba un diccionario, pero se obtuvo ", {type(detections)})
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
while True:
    ret, frame = cap.read()

    image_np = np.array(frame)
    image_np = np.expand_dims(image_np, axis=0)
    print(image_np)
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    detections = detect_fn(input_tensor)

    #num_detections = int(detections['num_detections'])
    #num_detections = int(detections.numpy())
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break

detections = detect_fn(input_tensor)