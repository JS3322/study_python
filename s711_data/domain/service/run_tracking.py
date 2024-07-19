import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.ai.model.cleancode_twinner import CleancodeTwinner

def run_tracking_save_input(model_name, date, input_path):
    
    device = "cpu"
    ngpu = 1 if "cuda" in device else 0
    cleancode_twinner = CleancodeTwinner(device=device, ngpu=ngpu)
    
    
    