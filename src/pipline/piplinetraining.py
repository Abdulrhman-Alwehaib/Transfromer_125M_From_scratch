from components import data_ingestion
from components import data_transfromation
from components import data_validation
from components import model_trainer
from components import Transfromer_Decoder_Arch
from utility import *
import constants
import os
import pandas as pd
import numpy as np
import yaml
import re
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import math

class runPipline():
    def __init__(self,constants_class: constants): 
        self.constants = constants_class()
        self.dataIngestionObj = data_ingestion(self.constants)
        self.datavalidationObj = data_validation(self.constants)
        self.dataTransfromationObj = data_transfromation(self.constants)
        self.modelTrainer = model_trainer(self.constants)
        self.Transformer_arch_decoderOBJ = Transfromer_Decoder_Arch()
    
    def dataIngestionExecute(self):
        self.dataIngestionObj.execute()
    
    def dataValidationExecute(self):
        self.datavalidationObj.execute()
    
    def dataTransformationExecute(self):
        self.dataTransfromationObj.execute()
    
    def modelTrainerExecute(self):
        self.modelTrainer.execute()
    

    


    