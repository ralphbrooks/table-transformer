import table_transformer
from pytorch_lightning.cli import LightningCLI

"""
Notes on parameters:

* For what I am doing, I should never need TableDetr with facebook/detr-resnet-50
because this would mean I am training from scratch

* it is unclear if ground_truth_crop_bboxes needs to be True 

* For DocILE, Matej is NOT bootstrapping off of 

"""



def cli_main():
    # cli = LightningCLI(
    #     table_transformer.TableDetr,
    #     table_transformer.TableDetr.add_model_specific_args,
    #     seed_everything_default=42,
    #     description="Table Transformer",
    # )
    cli = LightningCLI(table_transformer.TableDetr)

if __name__ == "__main__":
    cli_main()
