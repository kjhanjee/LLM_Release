"""Module for creating training data generator class
"""
# Standard imports
import os
import json
# Config imports
from utils.read_config import get_config

config = get_config()

class TrainingDataGen:
    """Module for creating training data generator for iterating over multiple files for training the model
    """
    def __init__(self, base_path:str = "./data/", config_path:str = None) -> None:
        """Init function to initialize the class variables
        
        Args:
            base_path (str): Path of the dataset
            config_path (str): To load existing saved config frile from training
        Returns:
            None
        """
        if not config_path:
            self.file_paths =[]
            dirs = os.listdir(base_path)
            for dir in dirs:
                files = os.listdir(os.path.join(base_path,dir))
                files2 = [int(file.replace(".txt","")) if file.find(".txt") > -1 else file for file in files]
                files2 = sorted(files2)
                for index, f in enumerate(files2):
                    file = files[index]
                    if file.find('.txt') > -1:
                        f_path = base_path+dir+'/'+str(f)+".txt"
                        self.file_paths.append(f_path)
                    elif file.find('.md') > -1:
                        f_path = base_path+dir+'/'+str(f)+".md"
                        self.file_paths.append(f_path)
            self.completed_files = []
            self.last_file = ""
        else:
            self.load_state(config_path)
     
    def save_state(self,config_path: str):
        """Save current state of the file iterator
        
        Args:
            config_path (str): Path to save the Iterator Config while training
        
        Returns:
            None
        """
        config_dict = {
            "file_paths":self.file_paths,
            "last_file":self.last_file,
            "completed_files":self.completed_files,
        }
        with open(config_path+'datagenconfig.json','w+') as f:
            f.write(json.dumps(config_dict))
            
    def load_state(self,config_path: str):
        """Read exisiting state of the file iterator
        
        Args:
            config_path (str): Path to load the Iterator Config from while training
        
        Returns:
            None
        """
        with open(config_path+'datagenconfig.json','r') as f:
            config_dict = json.loads(f.read())
        self.last_file = config_dict["last_file"]
        self.file_paths = config_dict["file_paths"]
        self.completed_files = config_dict["completed_files"]
    
    def get_sentences(self,path:str,text:str):
        """Read exisiting state of the file iterator
        
        Args:
            path (str): Path to do appropriate cleaning on text depending on the data
            text (str): text from the file
        
        Returns:
            list: List of Sentences from the text, either as whole text as one item or split accoring to dataset type
        """
        if path.find("books") > -1:
            sentences = []
            processed = config['cleaned_files']
            found = 0
            for processed_file in processed:
                if not path.find(processed_file) > -1:
                    found = 0
                else:
                    sentences = [text]
                    found = 1
                    break
            if found == 1:
                return sentences
            else:
                sentences = []
        elif path.find('pmc_extract') > -1:
            sections = text.split("=")
            if len(sections) > 10:
                for section_index, section in enumerate(sections):
                    if len(section) < 100:
                        sections.remove(section)
                    else:
                        sections[section_index] = section[:section.find("\n\n",len(section)-50)] + " [END_GEN]"
                sentences = []
                for section in sections:
                    sentences.extend(section.split('\n\n'))
            else:
                sections = text.split("*")
                if len(sections) > 10:
                    for section_index, section in enumerate(sections):
                        if len(section) < 100:
                            sections.remove(section)
                        else:
                            sections[section_index] = section[:section.find("\n\n",len(section)-50)] + " [END_GEN]"
                    sentences = []
                    for section in sections:
                        sentences.extend(section.split('\n\n'))
                else:
                    sentences = text.split("\n\n")
        elif path.find('pubmedextracts') > -1:
            sections = text.split("[SEP]\n")
            if len(sections) > 10:
                for section_index, section in enumerate(sections):
                    sections[section_index] = section + " [END_GEN]"
                sentences = []
                for section in sections:
                    sentences.extend(section.split('\n\n'))
            else:
                sentences = text.split("\n\n")
                sentences = [sentence.replace('[CLS]',' ').replace('[SEP]',' ') + ' [END_GEN]' for sentence in sentences]
        else:
            sentences = text.split('\n\n')
            sentences = [sentence + ' [END_GEN]' for sentence in sentences]
            
        return sentences
        
    def gen_batch(self):
        """Iterator for generating text for training the model
        
        Args:
           None
        
        Yields:
            list: List of Sentences from the text, either as whole text as one item or split accoring to dataset type
        """
        for path in self.file_paths:
            sentences = []
            if not path in self.completed_files: 
                self.last_file = path
                with open(path,"r") as f:
                    text = f.read()
                print("Getting Sentences from file",path)
                sentences = self.get_sentences(path,text)
                if len(sentences) == 0:
                    continue
                elif len(sentences) > 0:
                    self.completed_files.append(path)
                    yield sentences