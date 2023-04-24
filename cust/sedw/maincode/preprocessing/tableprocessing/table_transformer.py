
import os
import xmltodict

from transformers import AutoConfig
from typing import Dict, Iterable, Optional, Optional, Sequence, List, Tuple, Any

import numpy as np
import PIL
from PIL import Image

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection
from transformers.models.detr.feature_extraction_detr import DetrFeatureExtractor
from transformers.models.detr.modeling_detr import DetrForObjectDetection
from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerForObjectDetection,
    TableTransformerObjectDetectionOutput,
)

def get_bbox_predictions(
    batch: Dict,
    out: TableTransformerObjectDetectionOutput,
    extractor: DetrFeatureExtractor,
    allowed_label_ids: Optional[Iterable] = None,
    threshold: float = 0.5,
):
    """Get bboxes from model output
    This resizes the bboxes to the original image size

    Parameters
    ----------
    batch
        A batch from DataLoader
    out :
        Batched model output
    extractor:
        Feature extractor used to preprocess the input
    allowed_label_ids (using numerical representations of the labels)
        Only return predictions of this category
    threshold
        Don't return predictions with lower confidence

    Returns
    -------
    bbox_predictions : List[np.ndarray]
        A list of bbox predictions arrays, one for each input example
    """

    """
    concat the original sizes in the batch, detach them from the computation graph, and add a new dimension 
    tensor will be of shape (batch_size, 2)
    """

    # Shape is 2 before the unsqueeze. After, shape is (1, 2) so that orig_sizes is of size (batch, 2)
    orig_sizes = torch.concat(
        [labels["orig_size"].detach().unsqueeze(0) for labels in batch["labels"]]
    )

    # This is post processing the output of the model. It rescales the bboxes to the original image size.

    # TODO - Matěj - all of the tensors are coming up blank when they go through
    # Matěj - It would be helpful to talk through this part of the code and the rescaling part of your code.
    postprocessed_outputs = extractor.post_process_object_detection(
        out, threshold=threshold, target_sizes=orig_sizes
    )

    res = []
    for i in range(len(out.logits)):
        scores = postprocessed_outputs[i]["scores"].detach().cpu().numpy()

        # Rearrange the bboxes (relative to original image) in descending order of confidence
        bboxes_scaled = (
            postprocessed_outputs[i]["boxes"].detach().cpu().numpy()[scores.argsort()[::-1]]
        )

        if allowed_label_ids is not None:
            # Only keep bboxes of the allowed labels
            bboxes_scaled = bboxes_scaled[
                np.array(
                    [li in allowed_label_ids for li in postprocessed_outputs[i]["labels"]],
                    dtype=bool,
                )
            ]

            # calculate the areas of the bounding boxes
            areas = (bboxes_scaled[:, 2] - bboxes_scaled[:, 0]).astype(np.float32) * (
                bboxes_scaled[:, 3] - bboxes_scaled[:, 1]
            )

            # remove bounding boxes with an area smaller than 4 pixels
            bboxes_scaled = bboxes_scaled[areas > 4]
        

        # append the post process bounding box to the results
        res.append(bboxes_scaled)
        
    return res

class TableTransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        coco_xml_dir: str,
        img_dir: str,
        extractor: AutoFeatureExtractor,
        filter_category_ids: Optional[Sequence[int]] = None,
        crop_bboxes: Optional[Sequence[Sequence[float]]] = None,
        load_ground_truth_crop_bboxes: bool = False,
        num_workers: int = 10,
    ):
        """Dataset for table-transformer/DETR

        Parameters
        ----------
        * extractor
            HF table-transformer feature extractor (This is table transformer or structure)
        * filter_category_ids
            keep only these category ids (0 for tables, 2 for line items)
        * crop_bboxes
            Crop bboxes, one for each image, in the (left, top, right, bottom)
            and values in pixels
        * load_ground_truth_crop_bboxes
            Use table bboxes as crop_bboxes
        * num_workers
            Number of processes to use for dataset preparation
        """
        self.coco_xml_dir = coco_xml_dir
        self.img_dir = img_dir

        self.extractor = extractor
        self.filter_category_ids = filter_category_ids
        self.num_workers = num_workers

        # the length is the number of annotations
        # self.coco_annots, gt_crop_boxes = self._get_coco_annots_and_crop_boxes()

        # This only processes data where there is an xml file

        # There is no annotation if there is no data
        self.data = self._parse_coco_xml_files()
        

    def _parse_coco_xml_files(self) -> List[Dict]:

        data = []
        for xml_file in os.listdir(self.coco_xml_dir):
            if xml_file.endswith(".xml"):
                file_path = os.path.join(self.coco_xml_dir, xml_file)
                with open(file_path, "r") as f:
                    annotation = xmltodict.parse(f.read())["annotation"]
                parsed_data = self._parse_coco_xml(annotation)
                if self.filter_category_ids is not None:
                    parsed_data["objects"] = [obj for obj in parsed_data["objects"] if obj["category_id"] in self.filter_category_ids]
                data.append(parsed_data)
        return data


    def _parse_coco_xml(self, annotation: Dict) -> Dict:
        objects = []

        if type(annotation["object"]) == dict:
            unprocessed_objects = [annotation["object"]]
        else:
            # If you have multiple objects, there is no need to wrap that in a list.
            unprocessed_objects = annotation["object"]

        for obj in unprocessed_objects:
            objects.append({
                "name": obj["name"],
                "category_id": 0 if obj["name"] == "tabledetected" else 2,
                "bbox": {
                    "xmin": int(obj["bndbox"]["xmin"]),
                    "ymin": int(obj["bndbox"]["ymin"]),
                    "xmax": int(obj["bndbox"]["xmax"]),
                    "ymax": int(obj["bndbox"]["ymax"]),
                },
            })

        return {
            "filename": annotation["filename"],
            "path": annotation["path"],
            "width": int(annotation["size"]["width"]),
            "height": int(annotation["size"]["height"]),
            "objects": objects,
        }




    def __len__(self) -> int:
        """Return number of examples in the dataset"""
        return len(self.data)


    def get_img_and_target(self, idx):
        """Return PIL image and target (bbox, label, area) of the example at index `idx`"""
        
        annotation_data = self.data[idx]
        img = self._load_image(annotation_data["filename"])


        # The box needs to be of the format in pixels 

        # The DETR format requires target to be
        # https://huggingface.co/transformers/v4.10.1/model_doc/detr.html
        # each Dict has to be a COCO object annotation

        def _get_bbox_dimensions(obj):
            left = obj["bbox"]["xmin"]
            top = obj["bbox"]["ymin"]
            bottom = obj["bbox"]["ymax"]
            right = obj["bbox"]["xmax"]

            bbox_width = right - left
            bbox_height = bottom - top
            return left, top, bottom, right, bbox_width, bbox_height

        def get_bbox_from_object(obj):
            left, top, bottom, right, bbox_width, bbox_height = _get_bbox_dimensions(obj)

            # bbox is a list with xmin, ymin, width, height
            bbox = [left, top, bbox_width, bbox_height]
            return bbox

        def get_bbox_area_from_object(obj):
            left, top, bottom, right, bbox_width, bbox_height = _get_bbox_dimensions(obj)
            area = bbox_width * bbox_height
            return area

        coco_annot = [{"bbox": get_bbox_from_object(obj),
                       "area": get_bbox_area_from_object(obj),
                       "category_id": obj["category_id"]}
                      for obj in annotation_data["objects"]]

        target = {"image_id": -1, "annotations": coco_annot}

        return img, target

    def _load_image(self, filename: str) -> Image.Image:
        """
        Load image from `filename` and convert it to RGB

        Parameters:
        -----------
        * filename
            Name of the image file
            
        Returns:
        --------
        * PIL image

        """
        img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path)
        return img.convert("RGB")


    def get_img(self, idx):
        """Return PIL image of the example at index `idx`"""
        return self.get_img_and_target(idx)[0]    

    def get_coco_target(self, idx):
        """Return COCO target (RAB w/ image id) of the example at index `idx`"""
        return self.get_img_and_target(idx)[1]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, target = self.get_img_and_target(idx)

        # Given the target and given the image, extract the features
        # The extractor will do the conversion into a format that the model can understand

        encoding = self.extractor(images=img, annotations=target, return_tensors="pt")

        # https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target_without_batch = encoding["labels"][0] # remove batch dimension
        
        return pixel_values, target_without_batch

    def crop_example(self, img: PIL.Image, coco_annot: List, crop_bbox: Sequence[float]):
        """Crop input image and annotations

        The annotations are cropped to their intersection with the image crop
        (and their bbox coordinates are changed to be relative to the crop top
        left corner)



        Parameters
        ----------
        coco_annot
            COCO target. A list of dicts with "bbox" (top, left, width,
            height), "area" etc. keys
        crop_bbox
            (top, left, right, bottom) (as returned by TableDetr)
        """



        # TODO - Matěj - it would be helpful to talk through your crop_example code in the call.




class TableDetr(pl.LightningModule):
    def __init__(
        self,
        description: str,
        train_dataset_name: str = "train",
        val_dataset_name: str = "val",
        task: str = None,
        initial_checkpoint: str = None,
        config_hf_id: str = None,
        data_dir=r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\data\collectedimages",
        load_ground_truth_crop_bboxes: bool = False,
        predictions_root_dir: str = r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\results\table_transformer\predictions",
        crop_bboxes_filename: Optional[str] = None,
        lr: float = 3e-5,
        lr_backbone: float = 3e-7,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_workers: int = 16,
        threshold: float = 0.5,
    ):
        """Table transformer Lightning Module

        Modified from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

        Primarily created to try out table-transformer pretrained checkpoints:
        https://huggingface.co/microsoft/table-transformer-structure-recognition
        https://huggingface.co/microsoft/table-transformer-detection

        All of the parameters are in teh model part of the config file

        Parameters
        ----------
        description
            A short experiment description
        train_dataset_name )
            Saved COCO dataset directory name (should exist in `data_dir`)
        val_dataset_name
            Saved COCO dataset directory name (should exist in `data_dir`)
        task
            Either table-line-item-detection or table-detection
        initial_checkpoint
            A checkpoint used to initialize model weights. Either a path to a
            .ckpt file, a HuggingFace id or None (then a default pretrained
            model will be used)
        * config_hf_id
            Use this HF model id to load model (useful when starting from a
            local checkpoint)
        * data_dir
            Directory with the COCO dataset
        * load_ground_truth_crop_bboxes
            Train on ground truth table crops (for structure recognition)
        * predictions_root_dir
            Directory with saved crop_bboxes ????
        * crop_bboxes_filename
            Name of the pickle file with saved crop bboxes
        * lr
            Learning rate of the Detr transformer
        * lr_backbone
            Learning rate of the ResNet-18 backbone
        * num_workers
            DataLoader num_workers (both for train and val)
        * threshold
            Only return detections with at least this confidence during inference
        """
        super().__init__()

        allowed_tasks = ["table-detection", "table-line-item-detection"]
        if task not in allowed_tasks:
            raise ValueError(f"task {task} not in {allowed_tasks}")
        
        # if you are loading the ground truth crop boxes, you must have the name of 
        # the pickle file with the crop boxes
        
        if load_ground_truth_crop_bboxes and crop_bboxes_filename is not None:
            raise ValueError(
                "both load_ground_truth_crop_bboxes and crop_bboxes_filename specified"
            )

        # if cropped bboxes have been saved, store them in a variable
        self.crop_bboxes_filename = crop_bboxes_filename


        if task == "table-detection":
            hf_id = "microsoft/table-transformer-detection"
            allowed_labels = ["table"]
            self.filter_category_ids = [0]

        # Torch is either going to load a pretrained model or a local checkpoint
        if initial_checkpoint is None:
            initial_checkpoint = hf_id
        self.initial_checkpoint = initial_checkpoint

        # RAB processing before checkpoint
        # a class can be set a variable name
        cls = TableTransformerForObjectDetection

        if config_hf_id is not None:
            # This is a situation where we are loading a local checkpoint
            hf_id = config_hf_id
            # if the input that comes through is detr, use the correct class
            # This would imply you are training from scratch
            if hf_id == "facebook/detr-resnet-50":
                cls = DetrForObjectDetection

        # The config will be generated either from table, table-struc or detr
        # this configuration information is used with the class
        config = AutoConfig.from_pretrained(hf_id)
        self.model = cls(config)

        if initial_checkpoint.endswith(".ckpt"):

            ckpt = torch.load(initial_checkpoint)

            # The following is going to load the weights and biases of the model
            # This is helpful when you are resuming fine tuning

            self.model.load_state_dict(
                {
                    k.replace("model.model.", "model.")
                    .replace("model.class_labels_classifier", "class_labels_classifier")
                    .replace("model.bbox_predictor", "bbox_predictor"): v
                    for k, v in ckpt["state_dict"].items()
                }
            )
        else:
            # If you are loading from HuggingFace, then the initial_checkpoint should also be huggingface
            if config_hf_id is not None and config_hf_id != initial_checkpoint:
                raise ValueError("Loading model from HF, but different config_hf_id specified")  
            
        # values from table-transformer (force for DETR)
        # This is just a mapping that maps 0 to table
        self.model.config.label2id["table"] = 0
        self.model.config.id2label[0] = "table"



        # This is HF specific. The feature extractor essential for preparing input data. 
        self.extractor = AutoFeatureExtractor.from_pretrained(hf_id)

        self.lr = lr  # Learning rate of the Detr transformer
        self.lr_backbone = lr_backbone  # Learning rate of the ResNet-18 backbone (training from scratch)

        """
        Weight decay is a regularization technique that penalizes large weights. It is helpful to prevent overfitting.
        """

        self.data_dir = data_dir  # Directory with saved COCO datasets
        self.predictions_root_dir = predictions_root_dir # ???
        self.weight_decay = weight_decay  # Weight decay of the optimizer
        self.batch_size = batch_size  # Batch size of the training and validation dataloaders
        self.num_workers = num_workers  # Number of workers of the training and validation dataloaders
        self.threshold = threshold  # Only return detections with at least this confidence during inference
        self.description = description  # A short experiment description
        self.train_dataset_name = train_dataset_name  # Saved COCO dataset directory name (should exist in `dataset_root_dir`)
        self.val_dataset_name = val_dataset_name  # Saved COCO dataset directory name (should exist in `dataset_root_dir`)

        """
        This relates to TableTransformerDataset 
        """

        self.load_ground_truth_crop_bboxes = load_ground_truth_crop_bboxes  # Train on ground truth table crops (for structure recognition)


        # I think this takes all arguments from init and saves them as hyperparameters. 
        self.save_hyperparameters()

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs
    
    def common_step(self, batch, batch_idx, split):
        """
        This is used in training, validation, and test
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]


        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        # This is a forward pass with lablels 
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss

        # This contains loss and other relevant information
        loss_dict = outputs.loss_dict

        # batch size is already determined in the dataloader
        batch_size = batch["pixel_values"].shape[0]
        self.log(f"{split}_loss", loss, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        """
        This is a wrapper within lighting to configure the training step
        """
        return self.common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
            return self.common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        out = self.model(**batch)
        return get_bbox_predictions(
            batch,
            out,
            self.extractor,
            allowed_label_ids=[self.model.config.label2id[label] for label in self.allowed_labels],
            threshold=self.threshold,
        )
 

    def configure_optimizers(self):
        """
        This is a wrapper within lighting to configure the optmizer
        """
        # If backbone is in named parameters, use the learning rate for backbone
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]

        # Weight decay looks at the magnitude of parameter values and penalizes the large ones
        # I am keeping weight decay in for now because it is in the original code
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
    
    def collate_fn(self, batch):
        """
        collate_fn is used to create the batch.
        It is triggered by the dataloader.
        """

        pixel_values = [item[0] for item in batch]

        # create the pixel mask based on feature specification
        encoding = self.extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch
    
    def get_dataloader(self, split: str):
        # I think this is the bboxes once they are scaled down to what the model expects

        # Split determines if data is train or val

        if split == "train":
            split_xml_dir: str = os.path.join(self.data_dir, self.train_dataset_name)
        elif split == "val":
            split_xml_dir: str = os.path.join(self.data_dir, self.val_dataset_name)

        split_img_dir: str = split_xml_dir


        dataset = TableTransformerDataset(
            coco_xml_dir=split_xml_dir,
            img_dir=split_img_dir,
            extractor=self.extractor,
            filter_category_ids=self.filter_category_ids,
            crop_bboxes=None,
            load_ground_truth_crop_bboxes=False,
            num_workers=10
        )
    
        # Only shuffle the data if split is train (only shuffle the training data).
        dataloader = DataLoader(
            dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=(split == "train"),
        )
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader("train")
    
    def val_dataloader(self):
        return self.get_dataloader("val")