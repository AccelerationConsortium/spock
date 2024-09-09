class User(object):
    def __init__(self, user_id: str, user_name: str="llama3.1"):
        self.user_id = user_id
        self.user_model = user_name
        self.analyzed_files = {}
        
        
    def __repr__(self) -> str:
        return f"User({self.user_id}, {self.user_model})"
        
    def __str__(self) -> str:
        return f"Your model is: {self.user_model}"
    
    def __dict__(self) -> dict:
        return {self.user_id: {"user_model": self.user_model, "analyzed_files": self.analyzed_files}}
    
    
    def add_file(self, file_name, file_url):
        self.analyzed_files[file_name] = file_url
        
        
if __name__ == "__main__":
    user = User("1234")
    user.add_file("file1", "url1")
    print(user.__dict__())
