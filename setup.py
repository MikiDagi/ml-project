from setuptools import find_packages,setup
from typing import List

env_file='-e .'
def get_requirments(file_path : str)-> List[str]:
    '''
    get neccesary packages from requrments.txt file
    '''
    
    requirements=[]
    
    with open(file_path,'r') as file:
        requirements= file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if env_file in requirements:
            requirements.remove(env_file)
    
    return requirements


    
setup(
    name="end to end ml project",
    version="0.0.1",
    author="miked",
    email="mikiasdagem@gmail.com",
    packages=find_packages(),
    install_requires=get_requirments("requirements.txt")
    
)