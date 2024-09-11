from scidownl import scihub_download
import os
class Error(Exception):
    """Base class for exceptions in this module."""
    pass
    def __init__(self,paper):
        self.message = f"Pdf not found for {paper}"
        super().__init__(self.message)


class Downloader:
    def __init__(self, paper, paper_type, out, proxies = {
    'http': 'socks5://127.0.0.1:7890'
}
):
        self.paper = paper
        self.paper_type = paper_type
        self.out = out
        self.proxies = proxies
        
    def download(self):
        folder_paths = len(os.listdir(self.out.split("/")[0]))
        print(os.listdir(self.out.split("/")[0]))
        print(folder_paths)
        scihub_download(self.paper, paper_type=self.paper_type, out=self.paper.replace("/","_"), proxies=self.proxies)
        if len(os.listdir(self.out.split("/")[0])) == folder_paths:
            raise Error(self.paper)
        print(f"Downloaded {self.paper}")
        

        
if __name__ == "__main__":
    paper = "https://doi.org/10.1145/3375633"
    paper_type = "doi"
    out = f"pdfs/{paper}.pdf"
    proxies = {
    'http': 'socks5://127.0.0.1:7890'
}
    scihub_download(paper, paper_type=paper_type, out=out, proxies=proxies)
    #downloader = Downloader(paper, paper_type, out, proxies)
    """
    try:
        downloader.download()
    except Exception as e:
        print(e)
     """   
        
        
    '''
    To test later
    
    import tempfile
    import os

    # Create a temporary directory with the prefix 'test_function'
    with tempfile.TemporaryDirectory(prefix="test_function_") as temp_dir:
        print(f"Temporary folder created: {temp_dir}")
        
        # You can use the temp_dir as a regular directory for file operations
        # After the 'with' block, the directory and its contents will be automatically deleted
'''