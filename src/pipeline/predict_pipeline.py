import os   
import sys  
import pandas as pd       
import numpy as np         
from src.exception import CustomException 
from src.logger import logging 
from src.utils import  load_object

class PredictPipeline:
    def __init__(self):
         pass
     
    def predict(self,features):
        model_path="artifacts\model.pkl"
        preprocess_path="artifacts\preprocessing.pkl"
        model=load_object(obj_path=model_path)
        preprocess=load_object(obj_path=preprocess_path)
        scaled_data=preprocess.transform(features)
        preds=model.predict(scaled_data)
        return preds  
        
    
class CustomData:
    def __init__(self,
              gender:str,
              race_ethnicity:str,
              parental_level_of_education:str,
              lunch:str,
              test_preparation_course:str,
              reading_score:float,
              writing_score:float  
                 ):
        self.gender=gender,
        self.race_ethnicity=race_ethnicity,
        self.parental_level_of_education=parental_level_of_education,
        self.lunch=lunch,
        self.test_preparation_course=test_preparation_course,
        self.reading_score=reading_score,
        self.writing_score=writing_score
        
    def get_data_as_data_frame(self):
        try:
            data_frame_dict={
                "gender": [self.gender][0],
                "race_ethnicity": [self.race_ethnicity][0],
                "parental_level_of_education":[self.parental_level_of_education][0],
                "lunch":[self.lunch][0],
                "test_preparation_course":[self.test_preparation_course][0],
                "reading_score":[self.reading_score][0],
                "writing_score":[self.writing_score][0]
                
            }
            return pd.DataFrame(data_frame_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
         
    
    
        