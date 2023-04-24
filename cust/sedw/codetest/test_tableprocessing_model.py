
import unittest
import os


import torch
import pytorch_lightning as pl
import xmltodict
from PIL import Image


from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForObjectDetection

from preprocessing.tableprocessing.table_transformer import TableDetr, get_bbox_predictions

from transformers.models.table_transformer.modeling_table_transformer import (
    TableTransformerObjectDetectionOutput,
)



class MyTestTableDetr(unittest.TestCase):

    def test_init(self):

        model = TableDetr(
            description="Exp 001: Test Table Detection init",
            task="table-detection",
            initial_checkpoint="microsoft/table-transformer-detection",
            config_hf_id=None)
        
        self.assertEqual(True, True)

    def test_get_dataloader(self):

        model = TableDetr(
            description="Exp 001: Test Table Detection init",
            task="table-detection",
            initial_checkpoint="microsoft/table-transformer-detection",
            config_hf_id=None)

        dataloader = model.train_dataloader()
        self.assertIsNotNone(dataloader)
        self.assertTrue(hasattr(dataloader, '__iter__'))

    def test_forward(self):
        model = TableDetr(
            description="Exp 001: Test Table Detection init",
            task="table-detection",
            initial_checkpoint="microsoft/table-transformer-detection",
            config_hf_id=None)

        dataloader = model.train_dataloader()
        sample_batch = next(iter(dataloader))
        pixel_values = sample_batch["pixel_values"]
        pixel_mask = sample_batch["pixel_mask"]

        # Test the forward method
        output = model.forward(pixel_values, pixel_mask)

        # Test the output type
        self.assertIsInstance(output, TableTransformerObjectDetectionOutput)
        self.assertTrue(len(output["logits"]) > 0)
        self.assertTrue(len(output["pred_boxes"]) > 0)


    def test_get_bbox_predictions(self):
        model = TableDetr(
            description="Exp 001: Test Table Detection init",
            task="table-detection",
            initial_checkpoint="microsoft/table-transformer-detection",
            config_hf_id=None)

        dataloader = model.train_dataloader()
        sample_batch = next(iter(dataloader))
        pixel_values = sample_batch["pixel_values"]
        pixel_mask = sample_batch["pixel_mask"]

        # Test the forward method
        output: TableTransformerObjectDetectionOutput = model.forward(pixel_values, pixel_mask)

        model_extractor = model.extractor

        bbox_predictions = get_bbox_predictions(
            batch=sample_batch,
            out=output,
            extractor=model_extractor,
            allowed_label_ids=[0],
            threshold=0.5
        )

        self.assertTrue(True, True)


    def test_train_model_(self):

        # TODO - Matěj,  I am completely guessing what to do here to test training the model

        if torch.cuda.is_available():
            accelerator = "gpu"
        else:
            print("CUDA not available, predicting on CPU")
            accelerator = "cpu"

        trainer = pl.Trainer(
            accelerator= accelerator,
        )

        model = TableDetr(
            description="Exp 001: Test Table Detection init",
            task="table-detection",
            initial_checkpoint="microsoft/table-transformer-detection",
            config_hf_id=None)
        
        trainer.fit(model=model, 
        train_dataloaders=model.train_dataloader())
        
