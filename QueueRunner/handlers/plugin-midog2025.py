from torchvision.models.resnet import resnet18
from typing import Callable
from tqdm import tqdm
import numpy as np
import logging
import zipfile
import torch
import os
import openslide 
from pathlib import Path
import cv2 
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import logging
import json
import traceback

import yaml

from dataclasses import dataclass
from openslide import OpenSlide
from abc import ABC, abstractmethod
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops.boxes import nms as torch_nms
from tqdm.autonotebook import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

Coords = Tuple[int, int]
ImageType = Union[np.ndarray, torch.Tensor]
import lightning.pytorch as pl


from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models._utils import  _ovewrite_value_param
from torchvision.models.resnet import ResNet50_Weights, ResNet
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone, _resnet_fpn_extractor

from torchvision.models.detection.fcos import FCOS
from torchvision.ops.feature_pyramid_network import LastLevelP6P7, LastLevelMaxPool



BACKBONES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2"
    ]


WEIGHTS = [
    'IMAGENET1K_V1', 
    'IMAGENET1K_V2', 
    None
]


def create_active_map(slide: OpenSlide) -> Tuple[np.ndarray, int]:
    """Create a binary mask of tissue-containing regions in a whole slide image.

    This function generates a low-resolution map indicating regions containing tissue,
    using Otsu thresholding and morphological operations.

    Args:
        slide (OpenSlide): OpenSlide object of the whole slide image

    Returns:
        Tuple[np.ndarray, int]: Binary mask of tissue regions and downsampling factor
    """
    downsamples_int = [int(x) for x in slide.level_downsamples]
    if 32 in downsamples_int:
        ds = 32
    elif 16 in downsamples_int:
        ds = 16

    # get overview image
    level = np.where(np.abs(np.array(slide.level_downsamples)-ds)<0.1)[0][0]
    overview = np.array(slide.read_region(level=level, location=(0,0), size=slide.level_dimensions[level]))
    
    # remove transparent alpha channel 
    alpha_zero_mask = (overview[:, :, 3] == 0)
    overview[alpha_zero_mask, :] = 255
    
    # OTSU
    gray = cv2.cvtColor(overview[:,:,0:3],cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # closing
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    dil = cv2.dilate(thresh, kernel=elem)
    activeMap = cv2.erode(dil, kernel=elem)
    
    return activeMap, ds



def load_resnet_backbone(
        backbone: str = 'resnet50',
        weights: str = 'IMAGENET1K_V1'
        ) -> Tuple[ResNet, Dict[str, List[float]]]:
    """Load a ResNet backbone with specified weights.

    Args:
        backbone (str, optional): ResNet architecture name. Defaults to 'resnet50'.
        weights (str, optional): Pretrained weights type. Defaults to 'IMAGENET1K_V1'.

    Raises:
        ValueError: If backbone is not in supported BACKBONES list
        ValueError: If weights is not in supported WEIGHTS list

    Returns:
        ResNet: Pretrained ResNet backbone model
    """
    if backbone not in BACKBONES:
        raise ValueError(f'Unsupported backbone: {backbone}.')

    if weights in WEIGHTS:
        backbone = resnet.__dict__[backbone](weights=weights)
    else:
        raise ValueError(f'Unsupported weights: {weights}.')

    return backbone



def make_fcos_model(
        num_classes: int = 2,
        backbone: str = 'resnet50',
        weights: str = 'IMAGENET1K_V2',
        extra_blocks: bool = False,
        returned_layers: List[int] = [1, 2, 3, 4],
        trainable_backbone_layers: int = 5,
        center_sampling_radius: float = 1.5,
        det_thresh: float = 0.2,
        patch_size: int = 512,
        detections_per_img: int = 300,
        topk_candidates: int = 1000,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        **kwargs) -> FCOS:
    """Create an FCOS (Fully Convolutional One-Stage) object detection model.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 2.
        backbone (str, optional): Backbone architecture. Defaults to 'resnet50'.
        weights (str, optional): Pretrained weights type. Defaults to 'IMAGENET1K_V2'.
        extra_blocks (bool, optional): Whether to add P6/P7 FPN levels. Defaults to False.
        returned_layers (List[int], optional): FPN layers to use. Defaults to [1,2,3,4].
        trainable_backbone_layers (int, optional): Number of trainable backbone layers. Defaults to 5.
        center_sampling_radius (float, optional): Center sampling radius. Defaults to 1.5.
        det_thresh (float, optional): Detection confidence threshold. Defaults to 0.2.
        patch_size (int, optional): Input image size. Defaults to 512.
        detections_per_img (int, optional): Maximum detections per image. Defaults to 300.
        topk_candidates (int, optional): Number of top candidates to keep. Defaults to 1000.
        image_mean (List[float], optional): Image normalization mean. Defaults to None.
        image_std (List[float], optional): Image normalization std. Defaults to None.
        **kwargs: Additional arguments for FCOS model.

    Returns:
        FCOS: Configured FCOS model
    """
    
    # load backbone
    backbone = load_resnet_backbone(backbone=backbone, weights=weights)

    if extra_blocks:
        extra_blocks = LastLevelP6P7(256, 256)
    else:
        extra_blocks = LastLevelMaxPool()

    # load backbone with FPN
    backbone = _resnet_fpn_extractor(
        backbone=backbone,
        trainable_layers=trainable_backbone_layers,
        returned_layers=returned_layers,
        extra_blocks=extra_blocks
        )
    

    # if anchors are provided
    anchor_sizes = ((8,), (16,), (32,), (64,), (128,))  # equal to strides of multi-level feature map
    anchor_sizes = tuple([anchor_sizes[0]] + [anchor_sizes[i] for i in returned_layers])  # select specific feature maps
    aspect_ratios = ((1.0,),) * len(anchor_sizes)  # set only one anchor
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)


    # create model 
    model = FCOS(
        backbone,
        num_classes,
        anchor_generator=anchor_generator,
        min_size = patch_size,
        max_size = patch_size,
        image_mean = image_mean,
        image_std = image_std,
        center_sampling_radius = center_sampling_radius,
        score_thresh = det_thresh,
        nms_thresh = 0.6,
        detections_per_img = detections_per_img,
        topk_candidates = topk_candidates,
        **kwargs
        )
        
    
    return model 

@dataclass(kw_only=True)
class ModelConfig:
    """Base configuration class for object detection models.

    This class serves as a base configuration container for all detection models,
    storing common parameters and providing methods for saving/loading configurations.

    Attributes:
        model_name (str): Name identifier for the model
        detector (str): Type of detector (e.g., 'FasterRCNN', 'RetinaNet')
        backbone (str): Backbone architecture (e.g., 'resnet50')
        checkpoint (str): Path to model checkpoint
        det_thresh (float): Detection confidence threshold
        num_classes (int): Number of object classes
        extra_blocks (bool): Whether to use extra FPN blocks
        weights (str, optional): Path to pretrained weights
        returned_layers (List[int], optional): Specific layers to return from backbone
        patch_size (int, optional): Size of input patches. Defaults to 512
    """
    model_name: str 
    detector: str
    backbone: str 
    checkpoint: str 
    det_thresh: float 
    num_classes: int 
    extra_blocks: bool
    weights: str = None
    returned_layers: List[int] = None
    patch_size: int = 512

    def update(self, new: Dict[str, Any]) -> None:
        """Update configuration parameters with new values.

        Args:
            new (Dict[str, Any]): Dictionary of parameters to update
        """
        for key, value in new.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, filepath: str) -> None:
        """Save configuration to a YAML file.

        Args:
            filepath (str): Path where to save the configuration
        """
        with open(filepath, 'w') as file:
            yaml.dump(self.__dict__, file)

    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from a YAML file.

        Args:
            filepath (str): Path to the configuration file

        Returns:
            ModelConfig: Loaded configuration object
        """
        with open(filepath, 'r') as file:
            config_dict = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**config_dict)
    
@dataclass(kw_only=True)
class FCOS_Config(ModelConfig):
    """Configuration class specific to FCOS models.

    Extends ModelConfig without additional parameters.
    """

CONFIG_MAPPING = {
        'FCOS': FCOS_Config
    }


MODEL_MAPPINGS = {
        'FCOS': make_fcos_model
    }

class BaseDetectionModule(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            batch_size: int = 16,
            lr: float = 0.0001,
            optimizer: str = 'AdamW',
            scheduler: Union[str, None] = None):
        super().__init__()

        # save hparams
        self.save_hyperparameters(ignore=['model'])

        # store model 
        self.model = model 


    def forward(self, 
                x: torch.Tensor, 
                y: torch.Tensor = None) -> List[Dict[str, torch.Tensor]]:
        """Forward pass during inference.

        Returns post-processed predictions as a list of dictionaries.
        """
        return self.model(x, y)
    







 
class ConfigCreator:
    """Factory class for creating model configurations.

    This class provides static methods for creating and loading model configurations
    based on the detector type.
    """

    @staticmethod
    def create(settings: Dict[str, Any]) -> ModelConfig:
        """Create a model configuration from settings dictionary.

        Args:
            settings (Dict[str, Any]): Dictionary containing model configuration parameters

        Raises:
            ValueError: If the specified detector type is not supported

        Returns:
            ModelConfig: Appropriate configuration object for the specified detector
        """
        detector = settings['detector']
        if detector not in CONFIG_MAPPING:
            raise ValueError(f"Model {detector} not supported.")
        return CONFIG_MAPPING[detector](**settings)

    @staticmethod
    def load(filepath: str) -> ModelConfig:
        """Load a model configuration from a file.

        Args:
            filepath (str): Path to the configuration file

        Raises:
            ValueError: If the model type cannot be determined from the filepath

        Returns:
            ModelConfig: Loaded configuration object
        """
        for name, config in CONFIG_MAPPING.items():
            if name in filepath:
                return config.load(filepath)
        raise ValueError(f"Model {filepath} not recognized.")
        
class ModelFactory:
    """Factory class for creating and loading detection models.

    This class provides static methods for instantiating detection models
    based on configuration parameters.
    """

    @staticmethod
    def create(
            model_name: str,
            model_kwargs: Dict[str, Any] = None,
            module_kwargs: Dict[str, Any] = None) -> BaseDetectionModule:
        """Create a new detection model instance.

        Args:
            model_name (str): Name of the model architecture
            model_kwargs (Dict[str, Any], optional): Model-specific parameters
            module_kwargs (Dict[str, Any], optional): Lightning module parameters

        Raises:
            ValueError: If the specified model name is not recognized

        Returns:
            BaseDetectionModule: Instantiated detection model
        """
        if model_name not in MODEL_MAPPINGS:
            raise ValueError(f"Model {model_name} not recognized.")

        if model_kwargs is None:
            model_kwargs = {}
        if module_kwargs is None:
            module_kwargs = {}

        model = MODEL_MAPPINGS[model_name](**model_kwargs)
        return BaseDetectionModule(model, **module_kwargs)

    @staticmethod
    def load(
            config: ModelConfig,
            det_thresh: float = None,
            **kwargs) -> torch.nn.Module:
        """Load a detection model from a configuration and checkpoint.

        Args:
            config (ModelConfig): Model configuration object
            det_thresh (float, optional): Detection threshold to override config
            **kwargs: Additional keyword arguments for model loading

        Returns:
            torch.nn.Module: Loaded detection model
        """
        if det_thresh is None:
            det_thresh = config.det_thresh

        model_func = MODEL_MAPPINGS[config.detector]
        model_kwargs = {
            'backbone': config.backbone,
            'det_thresh': det_thresh,
            'num_classes': config.num_classes,
            'extra_blocks': config.extra_blocks,
            'returned_layers': config.returned_layers,
            'weights': config.weights,
            'patch_size': config.patch_size
        }

        return BaseDetectionModule.load_from_checkpoint(
            model=model_func(**model_kwargs),
            checkpoint_path=config.checkpoint,
            strict=False
        )

class OpenSlideWrapper(openslide.OpenSlide):
    """
        Wraps an openslide.OpenSlide object. The rationale here is that OpenSlide.read_region does not support z Stacks / frames as arguments, hence we have to encapsulate it

    """

    @property 
    def nFrames(self):
        return 1

    @property
    def frame_descriptors(self) -> list[str]:
        """ returns a list of strings, used as descriptor for each frame
        """
        return ['']

    def read_region(self, location, level, size, frame=0):
        return openslide.OpenSlide.read_region(self, location, level, size)


    def __init__(self, slide_path):
        super().__init__(slide_path)
        self.slide_path = slide_path  # Store path for later reconstruction

    def __reduce__(self):
        # Define how to pickle the object
        return (self.__class__, (self.slide_path,))


def load_model_from_config(config_path: Path):
    """Load model from configuration file."""
    try:
        config = ConfigCreator.load(str(config_path))
        model = ModelFactory.load(config)
        return model, config
    except Exception as e:
        print(f"[red]Error loading model: {str(e)}[/red]")
        raise 
    

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

from exact_sync.v1.models import PluginResultAnnotation, PluginResult, PluginResultEntry, Plugin, PluginJob

@dataclass
class ProcessorConfig:
    """Configuration for image processing."""
    save_dir: Optional[str] = None
    overwrite: Optional[bool] = False

@dataclass
class PatchConfig:
    """Configuration for patch extraction parameters.

    Args:
        size: Size of patches (assumed square)
        overlap: Overlap between adjacent patches (0-1)
        level: Pyramid level for WSI
        tissue_threshold: Minimum tissue content required (0-1)
    """
    size: int = 1024
    overlap: float = 0.3
    level: int = 0
    tissue_threshold: float = 0.1

@dataclass
class InferenceConfig:
    """Configuration for inference parameters.

    Args:
        batch_size: Number of patches to process simultaneously
        num_workers: Number of worker processes for data loading
        device: Device to run inference on ('cuda' or 'cpu')
        nms_thresh: Non-maximum suppression threshold
        score_thresh: Minimum confidence score for detections
        is_wsi: Whether to use WSI or ROI dataloader
    """
    batch_size: int = 8
    num_workers: int = 4
    device: str = 'cuda'
    nms_thresh: float = 0.3
    score_thresh: float = 0.5
    is_wsi: bool = False

class Strategy(ABC):
    """Abstract base class defining the interface for inference strategies.

    This class serves as a template for implementing different inference strategies
    for processing images with deep learning models.
    """

    @abstractmethod
    def process_image(self, model: nn.Module, image: str, **kwargs) -> Dict[str, np.ndarray]:
        """Process an image using the specified model.

        Args:
            model (nn.Module): The neural network model to use for inference
            image (str): Path to the image file
            **kwargs: Additional keyword arguments for processing

        Returns:
            Dict[str, np.ndarray]: Dictionary containing inference results
        """
        pass

class BaseInferenceDataset(Dataset, ABC):
    """Base class for inference datasets handling patch-based processing.

    Args:
        patch_config: Configuration for patch extraction
        transforms: Optional transforms to apply to patches
    """
    def __init__(
        self,
        patch_config: PatchConfig,
        transforms: Optional[Union[List[Callable], Callable]] = None,
    ) -> None:
        self.config = patch_config
        self.transforms = self._setup_transforms(transforms)

        # To be set by child classes
        self.coords: List[Coords] = []
        self.image_size: Tuple[int, int] = (0, 0)

    def _setup_transforms(
        self,
        transforms: Optional[Union[List[Callable], Callable]]
    ) -> Optional[Callable]:
        """Set up transformation pipeline."""
        if transforms is None:
            return None
        if isinstance(transforms, (list, tuple)):
            return T.Compose(transforms)
        return transforms

    @abstractmethod
    def _load_image(self) -> None:
        """Load the image/slide and set necessary attributes."""
        pass

    @abstractmethod
    def _get_patch(self, coords: Coords) -> ImageType:
        """Extract a patch from the image at given coordinates."""
        pass

    def _normalize_patch(self, patch: ImageType) -> torch.Tensor:
        """Normalize patch and convert to tensor."""
        if isinstance(patch, np.ndarray):
            patch = torch.from_numpy(patch / 255.).permute(2, 0, 1).float()
        return patch

    def _get_coords(self) -> List[Coords]:
        """Generate patch coordinates based on image size and overlap."""
        width, height = self.image_size
        stride = int(self.config.size * (1 - self.config.overlap))

        coords = []
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Adjust coordinates to prevent going out of bounds
                x_adj = min(x, width - self.config.size)
                y_adj = min(y, height - self.config.size)
                coords.append((x_adj, y_adj))

        return coords

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x, y = self.coords[idx]
        patch = self._get_patch((x, y))

        if self.transforms is not None:
            patch = self.transforms(patch)

        patch = self._normalize_patch(patch)
        return patch, x, y

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[List[torch.Tensor], List[int], List[int]]:
        """Custom collate function for batching."""
        patches, x_coords, y_coords = zip(*batch)
        return list(patches), list(x_coords), list(y_coords)

class WSI_InferenceDataset(BaseInferenceDataset):
    """Dataset for whole slide image inference."""

    def __init__(
        self,
        slide_path: Union[str, Path],
        patch_config: Optional[PatchConfig] = None,
        transforms: Optional[Union[List[Callable], Callable]] = None
    ) -> None:
        self.slide_path = Path(slide_path)
        if not self.slide_path.exists():
            raise FileNotFoundError(f"Slide not found: {slide_path}")

        patch_config = patch_config or PatchConfig()
        super().__init__(patch_config, transforms)

        self._load_image()
        self.active_map, self.ds = self._create_active_map()
        self.coords = self._get_coords()

    def _load_image(self) -> None:
        """Load slide and set size."""
        self.slide = OpenSlideWrapper(str(self.slide_path))
        self.image_size = self.slide.dimensions
        self.level_downsample = self.slide.level_downsamples[self.config.level]

    def _create_active_map(self) -> np.ndarray:
        """Create tissue mask for the slide."""
        return create_active_map(self.slide)

    def _get_coords(self) -> List[Coords]:
        """Generate coordinates for tissue-containing regions."""
        coords = super()._get_coords()

        # Filter coordinates based on tissue content
        filtered_coords = []
        for x, y in coords:
            if self._check_tissue_content((x, y)):
                filtered_coords.append((x, y))

        return filtered_coords

    def _check_tissue_content(self, coords: Coords) -> bool:
        """Check if a patch contains sufficient tissue content."""
        x, y = coords
        x_ds = int(x / self.ds)
        y_ds = int(y / self.ds)
        patch_size_ds = int(self.config.size * self.level_downsample / self.ds)

        tissue_content = np.mean(self.active_map[
            y_ds:y_ds + patch_size_ds,
            x_ds:x_ds + patch_size_ds
        ])

        return tissue_content >= self.config.tissue_threshold

    def _get_patch(self, coords: Coords) -> np.ndarray:
        """Extract patch from whole slide image."""
        x, y = coords
        patch = self.slide.read_region(
            location=(x, y),
            level=self.config.level,
            size=(self.config.size, self.config.size)
        ).convert('RGB')
        return np.array(patch)


class Torchvision_Inference(Strategy):
    """Inference strategy for Torchvision-based object detection models.

    This class handles patch-based inference for both regular images and whole slide images,
    with support for various detection models (Faster R-CNN, Mask R-CNN, FCOS, etc.).

    Args:
        model: The detection model to use
        config: Inference configuration parameters
        logger: Optional logger instance
    """
    def __init__(
        self,
        model: nn.Module,
        config: Optional[InferenceConfig] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.model = model
        self.config = config or InferenceConfig()
        self.logger = logger or self._setup_logger()
        self.device = self._setup_device()

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _setup_device(self) -> torch.device:
        """Set up and validate the processing device."""
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Using CPU instead.")
            return torch.device('cpu')
        return torch.device(self.config.device)

    def _create_dataloader(
        self,
        image_path: Union[str, Path],
        patch_config: PatchConfig
    ) -> DataLoader:
        """Create appropriate dataloader based on image type."""
        dataset_class = WSI_InferenceDataset

        try:
            dataset = dataset_class(
                image_path,
                patch_config=patch_config
            )
        except Exception as e:
            self.logger.error(f"Failed to create dataset: {str(e)}")
            raise

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=dataset.collate_fn,
        )

    @torch.no_grad()
    def _process_batch(
        self,
        batch: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Process a batch of patches."""
        images = [img.to(self.device) for img in batch]
        try:
            predictions = self.model(images)
            return predictions
        except RuntimeError as e:
            self.logger.error(f"Error during foward pass: {str(e)}")
            raise

    def _post_process_predictions(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        coords: List[Coords]
    ) -> Dict[str, torch.Tensor]:
        """Post-process predictions including coordinate adjustment and NMS."""
        boxes_list = []
        scores_list = []
        labels_list = []

        for pred, (x_orig, y_orig) in zip(predictions, coords):
            if len(pred['boxes']) > 0:
                # Adjust coordinates to original image space
                boxes = pred['boxes'] + torch.tensor(
                    [x_orig, y_orig, x_orig, y_orig],
                    device=pred['boxes'].device
                )               

                boxes_list.append(boxes)
                scores_list.append(pred['scores'])
                labels_list.append(pred['labels'])

        if not boxes_list:
            return {
                'boxes': torch.empty((0, 4), device=self.device),
                'scores': torch.empty(0, device=self.device),
                'labels': torch.empty(0, device=self.device)
            }

        # Concatenate all predictions
        boxes = torch.cat(boxes_list)
        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)

        # Apply NMS per class
        final_boxes = []
        final_scores = []
        final_labels = []

        for label in labels.unique():
            mask = labels == label
            class_boxes = boxes[mask]
            class_scores = scores[mask]

            keep = torch_nms(class_boxes, class_scores, self.config.nms_thresh)

            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(labels[mask][keep])

        # Concatenate 
        final_boxes = torch.cat(final_boxes)
        final_scores = torch.cat(final_scores)
        final_labels = torch.cat(final_labels)

        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }

    def process_image(
        self,
        image_path: Union[str, Path],
        patch_config: Optional[PatchConfig],
        update_progress: Callable,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process an image using patch-based inference.

        Args:
            image_path: Path to the image or slide file
            patch_config: Configuration for patch extraction
            **kwargs: Additional arguments to override default configs

        Returns:
            Dict containing 'boxes', 'scores', and 'labels' as numpy arrays
        """
        # Update config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        patch_config = patch_config or PatchConfig()

        # Prepare model
        self.model.eval()
        self.model.to(self.device)

        # Create dataloader
        dataloader = self._create_dataloader(image_path, patch_config)

        # Initialize results storage
        all_predictions = []
        all_coords = []

        # Process batches
        with tqdm(dataloader, desc="Processing batches") as pbar:
            for batch_images, batch_x, batch_y in pbar:
                if update_progress is not None:
                    update_progress(pbar.n*90./pbar.total)
                predictions = self._process_batch(batch_images)
                all_predictions.extend(predictions)
                all_coords.extend(zip(batch_x, batch_y))

        # Post-process results
        results = self._post_process_predictions(all_predictions, all_coords)

        # Convert to numpy arrays
        return {
            'boxes': results['boxes'].cpu().numpy(),
            'scores': results['scores'].cpu().numpy(),
            'labels': results['labels'].cpu().numpy()
        }





class ImageProcessor:
    """High-level processor for handling image detection tasks.

    This class orchestrates the image processing pipeline, handling both single images
    and batches, with support for various inference strategies.

    Args:
        strategy: An initialized inference strategy
        processor_config: Configuration for the processor
        logger: Optional logger instance
    """
    def __init__(
        self,
        strategy: Optional[Strategy] = Torchvision_Inference,
        processor_config: Optional[ProcessorConfig] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.strategy = strategy
        self.config = processor_config or ProcessorConfig()
        self.logger = logger or self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with appropriate configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger



    def should_proceed(
            self, 
            image_path: Union[str, Path],
            output_dir: Optional[Union[str, Path]] = None
    ) -> bool:
        """Checks if image should be processed."""
        result_path = Path(output_dir) / f"{Path(image_path).stem}_detections.json"
        if result_path.exists() and not self.config.overwrite:
            return False
        elif result_path.exists() and self.config.overwrite:
            return True
        else:
            return True




    def process_single(
        self,
        image_path: Union[str, Path],
        patch_config: Optional[PatchConfig],
        update_progress: Callable, 
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """Process a single image.

        Args:
            image_path: Path to the image file
            patch_config: Configuration for patch extraction
            output_dir: Directory to save results
            **kwargs: Additional arguments passed to the strategy

        Returns:
            Dictionary containing detection results
        """
        # Run inference using strategy
        results = self.strategy.process_image(
            image_path,
            patch_config=patch_config,
            update_progress=update_progress,
            **kwargs
        )

        return results 

def setup_inference(
    model: torch.nn.Module,
    is_wsi: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    patch_size: int,
    overlap: float,
    overwrite: bool
) -> tuple:
    """Setup inference components."""
    inference_config = InferenceConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        is_wsi=is_wsi
    )

    patch_config = PatchConfig(
        size=patch_size,
        overlap=overlap
    )

    processor_config = ProcessorConfig(
        overwrite=overwrite
    )

    strategy = Torchvision_Inference(model, inference_config)
    processor = ImageProcessor(
        strategy,
        processor_config
    )

    return processor, patch_config


def inference(apis:dict, job:PluginJob, update_progress:Callable, **kwargs):

        image = apis['images'].retrieve_image(job.image)
        logging.info('Retrieving image set for job %d ' % job.id)


        update_progress(0.01)
        unlinklist=[] # files to delete
        imageset = image.image_set

 
        logging.info('Checking annotation type availability for job %d' % job.id)
        annotationtypes = {anno_type['name']:anno_type for anno_type in apis['manager'].retrieve_annotationtypes(imageset)}        
                    
        # The correct annotation type is required in order to be able to add the annotation
        # CAVE: The annotation type also needs to be a part of the product that you want to apply
        # the detection on.
        annoclass=None
        for t in annotationtypes:
            if 'MITOTIC FIGURE' in t.upper():
                annoclass = annotationtypes[t]
        
        if (annoclass is None):
            error_message = 'Error: Missing annotation type'
            error_detail = 'Annotation class Mitotic Figure is required but does not exist for imageset '+str(imageset)
            logging.error(str(error_detail))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False              
        

        try:
            tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)
            if not os.path.exists(tpath):
                if ('.mrxs' in str(image.filename).lower()):
                    tpath = tpath + '.zip'
                logging.info('Downloading image %s to %s' % (image.filename,tpath))
                apis['images'].download_image(job.image, target_path=tpath, original_image=False)
                if ('.mrxs' in str(image.filename).lower()):
                    logging.info('Unzipping MRXS image %s' % (tpath))

                    with zipfile.ZipFile(tpath, 'r') as zip_ref:
                        zip_ref.extractall('tmp/')
                        for f in zip_ref.filelist:
                            unlinklist.append('tmp/'+f.orig_filename)
                        unlinklist.append(tpath)
                    # Original target path is MRXS file
                    tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)
                    
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while downloading'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False            

        try:
            logging.info('Stage 1 for job %d' % job.id)

            wsi_extensions = ('.svs', '.tif', '.tiff', '.dcm', '.vms', '.ndpi', '.vmu', '.mrxs', '.czi')
#            regular_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
            if os.path.splitext(tpath)[-1].upper in wsi_extensions:
                is_wsi=True
            else:
                is_wsi=False

            config_path='handlers/configs/MIDOG25_FCOS_x50.yaml'
            model, config = load_model_from_config(config_path)

            processor, patch_config = setup_inference(model=model, is_wsi=is_wsi, batch_size=8, num_workers=4, device='cuda', patch_size=1024, overlap=0.3, overwrite=True)
            raw_results = processor.process_single(tpath, patch_config=patch_config,update_progress=update_progress)

            stage1_results = [box + [score, ] for box, score in zip(raw_results['boxes'], raw_results['scores'])]


        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing stage 1'
            error_detail = str(e)
            logging.error(str(e))
            logging.error("Exception type: %s", type(e).__name__)
            logging.error("Exception message: %s", str(e))
            logging.error("Traceback:\n%s", traceback.format_exc())            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

            

        try:
            logging.info('Creating plugin result')
            existing = [j.id for j in apis['processing'].list_plugin_results().results if j.job==job.id]
            if len(existing)>0:
                apis['processing'].destroy_plugin_result(existing[0])
            
            # Create Result for job
            # Each job is linked to a single result, which may consist of several result entries.
            result = PluginResult(job=job.id, image=image.id, plugin=job.plugin, entries=[])
            result = apis['processing'].create_plugin_result(body=result)

            
            logging.info('Creating plugin entry')
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result'
            error_detail = str(e)+f'Job {job.id}, Image {image.id}, Pliugin {job.plugin}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
            
        try:
            # Create result entry for result
            # Each plugin result can contain collection of annotations. 
            resultentry = PluginResultEntry(pluginresult=result.id, name='Mitotic Figures', annotation_results = [], bitmap_results=[], default_threshold=0.55)
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        try:
            # Loop through all detections
            for n, line in enumerate(tqdm(stage1_results,desc='Uploading annotations (skip imposters)')):

                if (n%update_steps == 0):
                    update_progress (90+10*(n/len(stage1_results))) # 90.100% are for upload

                predcoords, score = line[0:4], line[4], 


                vector = {"x1": predcoords[0], "y1": predcoords[1], "x2": predcoords[2], "y2": predcoords[3]}

                anno = PluginResultAnnotation(annotation_type=annoclass['id'], pluginresultentry=resultentry.id, image=image.id, vector=vector, score=score)
                anno = apis['processing'].create_plugin_result_annotation(body=anno, async_req=True)
                    
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while uploading the annotations'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
        
        try:
            os.unlink(tpath)
            for f in unlinklist:
                os.unlink(f)
        
        except Exception as e:
            logging.error('Error while deleting files: '+str(e)+'. Continuing anyway.')
        
        return True


plugin = {  'name':'MIDOG 2025 Baseline FCOS MIDOG++',
            'author':'Jonas Ammeling / Marc Aubreville', 
            'package':'org.deepmicroscopy.midog2025.baseline', 
            'contact':'jonas.ammeling@thi.de', 
            'abouturl':'https://github.com/DeepPathology/EXACT-QueueRunner/', 
            'icon':'handlers/logos/midog2025_logo.jpg',
            'products':[],
            'results':[],
            'inference_func' : inference}


