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
    python3 potato.py train --dataset=/path/to/potato/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 potato.py train --dataset=/path/to/potato/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 potato.py train --dataset=/path/to/potato/dataset --weights=imagenet

    # Apply color splash to an image
    python3 potato.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 potato.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from hummingbirdtech.dic.container import Container

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class PotatoConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "potato"

    # We use a GPU with 4GB memory
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + potato

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


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
        self.image_converter = Container().get('hummingbird.image_converter')
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
        annotations = json.load(open(os.path.join(dataset_dir, "hb-potato-segmentation-dataset.json")))
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
        # Load image
        path_image = self.image_info[image_id]['path']
        hummingbird_image =self.image_converter.read_image_from_path(path=path_image)
        image = self.compute_image_array_from_hummingbird_image(hummingbird_image)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    def load_vegetation_mask_image(self,image_id):
        """Load the specified mask image and return a [H,W] numpy array
        """
        # Load mask image
        path_mask = self.image_info[image_id]['path_vegetation_mask']
        hummingbird_mask_image = self.image_converter.read_image_from_path(path=path_mask)
        mask_array = hummingbird_mask_image.bands[0].to_ndarray()
        mask_array = np.where(mask_array==255,1,0)
        return mask_array
        
        
    def compute_image_array_from_hummingbird_image(self, hummingbird_image):
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


def train(model):
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
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


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


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
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
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PotatoConfig()
    else:
        class InferenceConfig(PotatoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
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

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
