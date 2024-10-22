import pandas as pd  
import numpy as np 
import os
import sys  
import pickle
from src.exception import CustomException   
from src.logger import logging 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # path to preprocessing model pkl file
    preprocessing_data_path: str = os.path.join("artifacts","preprocessing.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()
    
    def  get_data_transformer_object(self):
        
        try:
            num_columns=['reading_score', 'writing_score']
            cat_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline=Pipeline(
                steps=[
                    ("simple-imputer",SimpleImputer(strategy="median")),
                    ("standard-scaler",StandardScaler(with_mean=False))
                      ]
                                )
            cat_pipeline=Pipeline(
                steps=[
                    ("simple-imputer",SimpleImputer(strategy="most_frequent")),
                    ("one-hot-encoder",OneHotEncoder()),
                    ("standard-scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("numerical features transformer created.")
            logging.info("categorical features preproccessing created.")
            
            preprocess=ColumnTransformer(
                transformers=[
                    ("numeric-preprocess",num_pipeline,num_columns),
                    ("categorical-preproccess",cat_pipeline,cat_columns)
                ]
            )
            
            return preprocess
            
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path) 
            
            logging.info("read train and test data completed.")
            logging.info("obtaining preproccess object.")
            
            preproccessing=self.get_data_transformer_object()
            
            logging.info("preproccessing object obtained.")
            
            target_column="math_score"
            
            # train data preparation
            
            input_feature_train_df=train_df.drop(columns=target_column,axis=1)
            target_feature_train_df=train_df[target_column]
            
            # test data preparation
            
            input_feature_test_df=test_df.drop(columns=target_column,axis=1)
            target_feature_test_df=test_df[target_column]
            
            logging.info("preproccessing applied to train and test data.")
            
            input_feature_train_array=preproccessing.fit_transform(input_feature_train_df) 
            input_feature_test_array=preproccessing.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_array,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_array,np.array(target_feature_test_df)] 
            
            save_object(
                file_path=DataTransformationConfig.preprocessing_data_path,
                obj=preproccessing
            )
            
            logging.info("preproccessed obj saved.")
            return(
                train_arr,
                test_arr,
                DataTransformationConfig.preprocessing_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)  
        
        