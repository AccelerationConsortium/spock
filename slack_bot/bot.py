# Bot that notifies when a txt file is edited and gives data when triggered 
# with some predifined commands


import os
from datetime import datetime
import hashlib
import time


class Bot:
    def __init__(self, file) -> None:
        self.file = file
        self.running = True
        self.current_hash = self.__get_file_hash()
        
    def mention(self):
        return "You've mentioned me"
    
                
    def __get_file_hash(self) -> str:
        with open(self.file, 'r') as f:
            file_hash = hashlib.sha256(f.read().encode()).hexdigest()
        return file_hash
    
    def file_change(self) -> bool:
        current_hash = self.__get_file_hash()
        if self.current_hash != current_hash:
            self.current_hash = current_hash
            return True
        return False
    
    def last_modified(self) -> datetime:
        time = datetime.fromtimestamp(os.path.getmtime(self.file))
        return time
        
    def stop(self):
        self.running = False
        
        # Notify that bot stopped        
        
        


    
