"""
Mask R-CNN
Train on the Potato segmentation dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

Copyright (c) 2019 
Written by Melissande Machefer
------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python samples/potato/potato.py train --dataset=/path/to/potato/  --weights=coco --config_file=/path/to/config/file

    # Train a new model starting from ImageNet weights
    python samples/potato/potato.py train --dataset=/path/to/potato/ --weights=imagenet --config_file=/path/to/config/file

    # Continue training a model that you had trained earlier
    python samples/potato/potato.py train --dataset=/path/to/potato/  --weights=/path/to/weights.h5 --config_file=/path/to/config/file

    # Continue training the last model you trained. This will find
    # the last trained weights in the model directory.
    python samples/potato/potato.py train --dataset=/path/to/potato/  --weights=last --config_file=/path/to/config/file


    # Run POTATO evaluation on the last trained model
    python samples/potato/potato.py evaluate --dataset=/path/to/potato/ --weights=last --config_file=/path/to/config/file

    # Apply color splash to an image
    python potato.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>


"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf
from hummingbirdtech.dic.container import Container

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "model_weights/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs/")
############################################################
#  Configurations
############################################################

image_converter = Container().get('hummingbird.image_converter')

class PotatoConfig(Config):
    """Configuration for training on the potato  dataset.
    # TODO : add our final configuration
    Derives from the base Config class and overrides some values.
    """
    NAME = "potato"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3
    
    
     # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
    
    
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    BACKBONE_STRIDES = [4,8,16,32,64]
    
    ##********** 1)ANCHORS GENERATION - for RPN*********
    
    #     Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 24,32, 48) 
    TOP_DOWN_PYRAMID_SIZE = 256
    
    ##********** 2)PROPOSAL LAYER ********* (no deep learning involved here)
    
     # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128 ##in dataset generation

    ## tf.image.non_max_suppression(boxes,scores,max_output_size,iou_threshold=0.5,...)
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD=0.7
    #A float representing the threshold for deciding whether boxes overlap too much with respect to IOU.    
    ## POST_NMS_ROIS_TRAINING~ POST_NMS_ROIS_INFERENCE ~proposal_count ~ max_output_size
    
    POST_NMS_ROIS_TRAINING=1500
    POST_NMS_ROIS_INFERENCE=800
    ##********** 3a)TRAINING - DETECTION TARGET LAYER *********
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 128

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33
    
    ##********** 3B)INFERENCE - DETECTION  LAYER *********

    
    # Non-maximum suppression threshold for detection in DetectionLater
    DETECTION_NMS_THRESHOLD = 0.5  # 0.5 above iou_threshold

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped in DetectionLater
    DETECTION_MIN_CONFIDENCE = 0.7

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 120

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 40

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 6
    LEARNING_RATE = 0.001

    TRAIN_BN = False



############################################################
#  Dataset
############################################################

class PotatoDataset(utils.Dataset):

    def load_potato(self, dataset_dir, subset):
        """Load a subset of the Potato dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes. We have only one class to add.
        self.add_class("potato", 1, "potato")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        

        # Load annotations
        # from hb-potato-segmentation-dataset.json
        # image_size feature was added:
        #         {1040:{
        #     'id_projected_cell_boundary': 1040,
        #     'regions': { 
        #         '0': {
        #             'id_intersected_bounding_box': 230,
        #             'shape_attributes': {
        #                 'all_points_x':[..],
        #                 'all_points_y': [..]},
        #             'id_bounding_box': 0}},
        #          '1':{...
        #                           },
        #     'image_width': 256,
        #     'image_height': 256,
        #     'dataset_split': 'train'}}
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        # Our potato segmentation dataset builder saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images. Keep annnotations with right subset.
        annotations = [a for a in annotations if (a['regions'] and a['dataset_split']==subset)]
 
        
        # Add images
        for a in annotations:
            
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            self.add_image(
                "potato",
                image_id=a['id_projected_cell_boundary'],  # use file name as a unique image id
                path= os.path.join(dataset_dir,'images',str(a['id_projected_cell_boundary'])+'_rgb.tif'),
                path_vegetation_mask= os.path.join(dataset_dir,'images',str(a['id_projected_cell_boundary'])+'_vegetation.tif'),
                width=a['image_width'], height=a['image_height'],
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a potato dataset image, delegate to parent class.
        info = self.image_info[image_id]
#         if info["source"] != "potato":
#             return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        mask_vegetation_flat = self.load_vegetation_mask_image(image_id)
        
        mask = np.repeat(np.expand_dims(mask_vegetation_flat,axis=2),len(info["polygons"]),axis=2)
        mask_pol = np.zeros([mask.shape[0], mask.shape[1],
                     mask.shape[2]],
                    dtype=np.uint8)
        
        if ((info["width"]!=mask.shape[1]) | (info["height"]!=mask.shape[0])):
            raise print('Vegetation and RGB mask don\'t have the same size')
            
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask_pol[rr, cc, i] = 1
            mask[:,:,i] = mask[:,:,i]* mask_pol[:,:,i]
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "potato":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        hummingbird_image = self.load_hummingbird_image(image_id)
        image = compute_image_array_from_hummingbird_image(hummingbird_image)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_hummingbird_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        path_image = self.image_info[image_id]['path']
        return image_converter.read_image_from_path(path=path_image)

    def load_vegetation_mask_image(self,image_id):
        """Load the specified mask image and return a [H,W] numpy array
        """
        # Load mask image
        path_mask = self.image_info[image_id]['path_vegetation_mask']
        hummingbird_mask_image = image_converter.read_image_from_path(path=path_mask)
        mask_array = hummingbird_mask_image.bands[0].to_ndarray()
        mask_array = np.where(mask_array==255,1,0)
        return mask_array


def compute_image_array_from_hummingbird_image(hummingbird_image):
    """ Compute array from hummingbird image
    """
    alpha = []

    if len(hummingbird_image.bands) == 3:
            data_type = hummingbird_image.bands[0].data_type
            if data_type == np.dtype('uint8') or data_type == np.dtype('uint16'):
                no_data_value = hummingbird_image.bands[0].no_data_value
                max_value = np.iinfo(data_type).max
                alpha.append(hummingbird_image.bands[0].transform(
                    lambda x: np.where(x == no_data_value, 0, max_value)).as_type(data_type))

    return np.stack(hummingbird_image.bands + alpha, axis=2)


def train(model, config, augmentation = False, epochs = 30):
    """Train the model."""
    # Training dataset.
    dataset_train = PotatoDataset()
    dataset_train.load_potato(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PotatoDataset()
    dataset_val.load_potato(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    if augmentation:
        import imgaug

        augmentation_operations = imgaug.augmenters.Sometimes(0.5, [
            imgaug.augmenters.Fliplr(0.5),
            imgaug.augmenters.Rot90(1),
            imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
        ])

    else:
        augmentation_operations = None
    print('Model trained for {} epochs with augmentation: {}'.format(epochs, augmentation))
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                augmentation = augmentation_operations,
                epochs= epochs,
                layers='heads')


def evaluate(model, config):
    " Evaluate the model"
    # Validation dataset
    dataset_val = PotatoDataset()
    dataset_val.load_potato(args.dataset, "val")
    dataset_val.prepare()

    image_ids = dataset_val.image_ids
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP on the validation set: ", np.mean(APs))


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None):

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    hummingbird_image = image_converter.read_image_from_path(path=image_path)
    image = compute_image_array_from_hummingbird_image(hummingbird_image)
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    splash = color_splash(image, r['masks'])
    # Save output
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)

    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect potatos.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--augmentation',
                        required= False)
    parser.add_argument('--epochs',
                        type = int,
                        help = " number of epochs to train the model",
                        required=False)
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/potato/dataset/",
                        help='Directory of the Potato dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--config_file', required=False,
                        metavar="/path/to/config_file/",
                        help='Directory of the Config file')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    # Validate arguments
    if args.command == "train" or args.command == "evaluate":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image,\
               "Provide --image to apply color splash"


    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    print("Config file: ", args.config_file)

    # Configurations
    if args.config_file:
        sys.path.append(args.config_file.split('config.py')[0])
        from config import PotatoConfig
        config = PotatoConfig()
    else:
        config = PotatoConfig()
    if args.command == 'splash' or args.command == 'evaluate':
        class InferenceConfig(PotatoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        DEVICE = "/cpu:0"
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
    else:
        DEVICE = "/cpu:0"
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.augmentation:
        augmentation = True
    else:
        augmentation = False
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 30

    # Train or evaluate
    if args.command == "train":
        train(model, config, augmentation, epochs)
    elif args.command == "evaluate":
        evaluate(model, config)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
