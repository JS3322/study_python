import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.ai.model.cleancode_twinner import CleancodeTwinner

def run_tracking_save_input(model_name, date, input_path):
    
	device = "cpu"
	ngpu = 1 if "cuda" in device else 0
	cleancode_twinner = CleancodeTwinner(device=device, ngpu=ngpu)

	#env
   MODEL_PATH = f"{model_name}/MODEL"
   DATA_PATH = f"{model_name}/tracking/Artiface/{date}"
   if os.path.isdir(DATA_PATH) is False:
      os.makedirs(DATA_PATH)
      
   OUTPUT_PATH = f"{DATA_PATH}/tracking.csv"
   scaler_path_dict = {
   	"profile_saler" : f"{MODEL_PATH}/profile_scaler.pkl"
   }
   cleancode_sem_model_path = f"{MODEL_PATH}/sem_dist_model.pt"
   cleancode_tca_model_path = f"{MODEL_PATH}/tca_dist_model.pt"
   
   #tracking
   df_ture = pd.read_csv(INPUT_PATH, index_col=False, pares_dates=["Datetime"], date_format=cleancode_twinner.date_format)
   df_sem_pred = cdsem_twinner.tracking(
   	model=cleancode_twinner.sem_dist_model,
      preprocessor=cleancode_twinner.tracking_processor,
      postprocessor=cleancode_twinner.tracking_postprocessor,
      cleancode_data_path=INPUT_PATH,
      scaler_path_dict=scaler_path_dict,
      model_path=sem_model_path
   )
   
   #save result
   df_pred.to_scv(OUTPUT_PATH, index=False)
   
   #accuracy
   tg_cost = ["CD20", "CD21"]
   acc = 1 - np.abs(df_pred[tg_cols] - df_true[tg_cols]) / (np.abs(df_true[tg_colst])+1)
   acc = acc.dropna().values.mean()
   
   #save figure
    
    
    