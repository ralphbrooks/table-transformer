import unittest
import os


import torch
import xmltodict
from PIL import Image


from sedw.maincode.api.askquestion import AskWeaviateConversation

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection

from preprocessing.tableprocessing.table_transformer import TableTransformerDataset

class MyTestDataset(unittest.TestCase):


    def test_len(self):

        hf_id = "microsoft/table-transformer-detection"
        extractor = AutoFeatureExtractor.from_pretrained(hf_id)

        ds = TableTransformerDataset(
            coco_xml_dir=r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\data\collectedimages\train",
            img_dir=r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\data\collectedimages\train",
            extractor=extractor,
            filter_category_ids=[0],
            crop_bboxes=None,
            load_ground_truth_crop_bboxes=False,
            num_workers=10
        )
        
        ds_len = ds.__len__()

        xml_files_count = sum([1 for f in os.listdir(ds.coco_xml_dir) if f.endswith(".xml")])

        self.assertEqual(ds_len, xml_files_count, "Length of dataset does not match the number of XML files in the coco_xml_dir.")



    def test_getitem(self):

        hf_id = "microsoft/table-transformer-detection"
        extractor = AutoFeatureExtractor.from_pretrained(hf_id)

        ds = TableTransformerDataset(
            coco_xml_dir=r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\data\collectedimages\train",
            img_dir=r"D:\whiteowlconsultinggroup\cust\sedw\maincode\preprocessing\tableprocessing\data\collectedimages\train",
            extractor=extractor,
            filter_category_ids=[0],
            crop_bboxes=None,
            load_ground_truth_crop_bboxes=False,
            num_workers=10,
        )
        
        pixel_values, target_without_batch = ds.__getitem__(0)

        image_path = os.path.join(ds.img_dir, ds.data[0]['filename'])
        image = Image.open(image_path)

        preprocessed_image = ds.extractor(images=image, return_tensors="pt")["pixel_values"]

        assert torch.allclose(pixel_values, preprocessed_image), "Pixel values do not match."



