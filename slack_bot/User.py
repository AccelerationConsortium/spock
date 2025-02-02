from typing import Optional
class User(object):
    def __init__(self, user_id: str, user_name: str="llama3.3", settings:Optional[dict[str, bool]]={'Summary':True, 'Questions':True,'Binary Response':True}) -> None:
        self.user_id = user_id
        self.user_model = user_name
        self.settings = settings
        
        
    def __repr__(self) -> str:
        return f"User({self.user_id}, {self.user_model})"
        
    def __str__(self) -> str:
        return f"Your model is: {self.user_model}"
    
    def __dict__(self) -> dict:
        return {self.user_id: {"user_model": self.user_model, "settings": self.settings}}
    
    

