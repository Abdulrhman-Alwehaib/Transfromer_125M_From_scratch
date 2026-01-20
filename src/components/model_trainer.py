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
import Transfromer_Decoder_Arch





class modelTrainer():
    def __init__(self,constants_class):
        self.constants = constants_class()
        self.config = readYaml(self.constants.CONFIG_FILE_PATH)
        self.modelTrainerConfig = self.config["model_trainer"]
        self.params =  
        self.proccessedData = torch.load(self.modelTrainerConfig["inputDataProccessed"])
        self.proccessedDataMASK = torch.load(self.modelTrainerConfig["inputDataMASKS"])
        transfromerDecoder = Transfromer_Decoder_Arch()

        self.model = transfromerDecoder.decoderNextWordPrediction(configuration_Model["vocab_size"],configuration_Model["d_model"],
                                  configuration_Model["block_size"],
                                  configuration_Model["heads"],
                                  configuration_Model["layers"],
                                  configuration_Model["dropRate"]
                                  )
        model = model.to(configuration_hyperParameters["device"])
        optimizer = torch.optim.AdamW(model.parameters(),configuration_hyperParameters["learning_rate"])
        lossFN = nn.CrossEntropyLoss(ignore_index=3)
        

    