import pandas as np     
import numpy as np  
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression,Ridge 
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor  
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os 
import sys   
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_trainer_path : str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_array,test_array):
        
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("data splited in to training and testing set")
            logging.info("models instantiated.")
            models={
                "liner_regression":LinearRegression(),
                "ridge":Ridge(),
                "Random forest":RandomForestRegressor(),
                "ada boost":AdaBoostRegressor(),
                "gradient decent":GradientBoostingRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "kNegibours":KNeighborsRegressor(),
                "xgboost":XGBRFRegressor(),
                "catboost":CatBoostRegressor(),
                
            }
            logging.info("model training started")
            test_r2score=[] 
            train_r2score=[]
            model_names=[]
            for name,mdl in models.items():
                model=mdl.fit(X_train,y_train) 
                # Prediction
                test_pred=model.predict(X_test)
                train_pred=model.predict(X_train)
                
                # r2 score for test and training.
                test_r2=r2_score(test_pred,y_test) 
                train_r2=r2_score(train_pred,y_train)
                
                # appending r2 score and model 
                test_r2score.append(test_r2)
                train_r2score.append(train_r2)
                model_names.append(mdl) 
            best_test_score=max(test_r2score)
            best_model=model_names[test_r2score.index(best_test_score)]
            save_object(
                file_path=ModelTrainerConfig.model_trainer_path,
                obj=best_model
            )
            logging.info("found best model with  best score and saved succsesfuly.")
            return {best_model:best_test_score}
            
            
        except Exception as e:
            raise CustomException(e,sys)

