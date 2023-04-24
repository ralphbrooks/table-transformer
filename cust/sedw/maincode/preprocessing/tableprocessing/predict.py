
import argparse
import json
import logging
import pickle

import table_transformer
import torch 


def main():
    """
    This function is going to do the prediction of data at the point where the model has been fine tuned.
    """

    parser = argparse.ArgumentParser()

    # TODO - the problem is that when I am first starting to train, I don't have a checkpoint
    
    # TODO - this needs to be an actual checkpoint that you provide
    parser.add_argument("--checkpoint_path", 
                        required=True)
    #required
    parser.add_argument(
        "--prediction_type",
        choices=["table_detection", "table_line_item_detection"],
        default="table_detection"
    )
    #required
    parser.add_argument("--dataset_path", default=r"D:\whiteowlconsultinggroup\maincode\preprocessing\tableprocessing\data\collectedimages")
    #required
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")

    #required
    parser.add_argument(
        "--output_json_path",
        help="A dictionary doc_id->page_n->predictions. With --prediction_type "
        "table_detection, a pickle file with a list of bboxes (one for each "
        "page) will be saved beside the json file.",
        default=r"D:\whiteowlconsultinggroup\maincode\preprocessing\tableprocessing\output"
    )

    parser.add_argument(
        "--force-cpu",
        help="if used, CPU will be used for the accelerator",
        default=True
    )

    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_workers", default=16)
    
    # TODO - this is needed for table_line_item_detection. Not sure what to put there.
    parser.add_argument(
        "--table_detection_predictions_pickle",
        default=None,
        help="Path to a file created by this script for a table detection model.",
    )

    args = parser.parse_args()

    if (
        args.prediction_type == "table_line_item_detection"
        and args.table_detection_predictions_pickle is None
    ):
        raise ValueError(
            "--prediction_type table_line_item_detection requires --table_detection_predictions_pickle"
        )

    if torch.cuda.is_available():
        accelerator = "gpu"

    elif args.force_cpu:
        logging.warning("Forcing prediction on CPU. Change with --force-cpu")
        accelerator = "cpu"
    else:
        logging.warning("CUDA not available, predicting on CPU")
        accelerator = "cpu"

    table_detr = table_transformer.TableDetr.load_from_checkpoint(args.checkpoint_path)

    
    # TODO - Matěj - I need additional clarification on what is going on in your version of predict.py

    print("Prediction complete")


if __name__ == "__main__":
    main()

