from typing import Optional, Union, List
import json
import time
from pathlib import Path
from docling_core.types.doc import DocItemLabel, ImageRefMode
from docling_core.types.doc.document import DEFAULT_EXPORT_LABELS
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    smolvlm_picture_description,
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PictureDescriptionApiOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import os
import torch
from docling_core.types.doc import PictureItem, ImageRefMode

os.environ["EASYOCR_MODULE_PATH"] = "/home/m/mehrad/brikiyou/scratch/EasyOCR"




from docling.utils.model_downloader import download_models
download_models(output_dir=Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts"), with_easyocr=True, with_smolvlm=True)

# Plan:
# 1. Use docling for this 
# 2. Use VLM/OCR for parsing
# 3. Output -> Md or else
# 4. Metadata extraction 
# 5. Langchain integration - Document loaders
# 6. Use GPU + parallel processing of everything / Maybe run OCR on CPU 

def configure_vlm_server(use_gpt:bool, vlm_model: str, prompt: str, url: str): 
    """
    Configure options for the OpenAI or Tensort vision-language model API.

    Args:
        model (str): The model name to use for image description generation.
        prompt (str): The prompt to use for image description generation.
        
    Returns:
        PictureDescriptionApiOptions: A configuration object for the VLM API.
    """
    # Configure your model of choice:
    # - API endpoint for Ollama VLM
    # - Specify the desired model for image description
    # - Define maximum tokens allowed for the generated description
    # - Specify the prompt used to instruct the model
    # - Define the timeout for the API call (in seconds)
    if not use_gpt:
        return PictureDescriptionApiOptions(
            url="http://localhost:11434/v1/chat/completions",  # To change to match Tensorrt server or chatgpt
            params={
                "model": vlm_model,      
                "max_completion_tokens": 512
            },
            prompt=prompt,
            timeout=10,
        )
class Pdf_document_loader():
    def __init__(self, sources:Union[List[str], List[Path], List[Union[str, Path]]], **kwargs):
        self.sources = sources
        self.num_gpus = torch.cuda.device_count()
        
    def parse_document(self, use_vlm:Optional[bool]=False, use_ocr:Optional[bool]=True, **kwargs):
        """
        Parses Document to markdown or other formats.
        """
        
        scratch = Path("/home/m/mehrad/brikiyou/scratch/docling_artifacts")
        (scratch / "EasyOCR" / "model").mkdir(parents=True, exist_ok=True)
        (scratch / "EasyOCR" / "user_network").mkdir(parents=True, exist_ok=True)

        # Check if 
        if self.num_gpus > 0:
            print(f"Using {self.num_gpus} GPUs for processing.")
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA,
                num_threads=64,
            )
        else:
            accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU,
                num_threads=64,
            )
        pipeline_options = PdfPipelineOptions() # artifacts_path=str(scratch)
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_formula_enrichment = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.do_code_enrichment = True
        
        pipeline_options.ocr_options = EasyOcrOptions(
            use_gpu=(self.num_gpus > 0),
            model_storage_directory=str(scratch / "EasyOCR" / "model"),
           # user_network_directory=str(scratch / "EasyOCR" / "user_network"),
            download_enabled=True
            
        )

        
        if use_vlm:
            if self.num_gpus == 0:
                raise ValueError("No GPUs available for VLM processing.")
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = smolvlm_picture_description # To change to custom function above 
            pipeline_options.picture_description_options.prompt = ("Describe the image in detail and accurately.")                                                                     
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = False
            

        converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
        start  = time.time()
        print(f"Parsing {self.sources[0]}...")
        doc = converter.convert(self.sources[0]).document 
        
        annotations_list = []
        image_counter = 0
        
        for element, *_ in doc.iterate_items():
            if isinstance(element, PictureItem):
                image_counter += 1
                annotation = "\n".join([ann.text for ann in element.annotations]) or "No annotations"
                annotations_list.append(annotation)
        
        output_md_path = '/home/m/mehrad/brikiyou/scratch/spock_2/spock/spock_literature/utils/doc-with-images.md'
        
        doc.save_as_markdown(
            output_md_path,
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder="%%ANNOTATION%%"
        )
        
        with open(output_md_path, 'r') as file:
            md_content = file.read()
        
        for ann in annotations_list:
            md_content = md_content.replace("%%ANNOTATION%%", ann, 1)
        
        with open(output_md_path, 'w') as file:
            file.write(md_content)
        
    def __ocr(self):
        """
        Use EasyOCR to parse the document.
        """
        pass
    
    def __vlm(self):
        """
        Use VLM to parse the document.
        """
        pass

       


if __name__ == "__main__":
    test_file = [Path("/home/m/mehrad/brikiyou/scratch/spock_2/spock/examples/data-sample.pdf")]
    pdf_loader = Pdf_document_loader(test_file)
    pdf_loader.parse_document(use_vlm=True)
    




        
        