import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from pathlib import Path


class LLM:
    def __init__(self, llm_model:str="llama3", embedding_model:str="mxbai-embed-large") -> None :
        self.llm = Ollama(model=llm_model, temperature=0.2)
        self.embedding = OllamaEmbeddings(model=embedding_model)
        self.folder_path = None
        
    def set_folder_path(self, folder_path:str) -> None:
        self.folder_path = folder_path
        
        
    def pdf_to_md(self, pdf_path:str) -> str:
        from transformers import AutoProcessor, VisionEncoderDecoderModel
        import torch

        processor = AutoProcessor.from_pretrained("facebook/nougat-base")
        model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        from typing import Optional, List
        import io
        import fitz

        def rasterize_paper(
            pdf: Path,
            outpath: Optional[Path] = None,
            dpi: int = 96,
            return_pil=False,
            pages=None,
        ) -> Optional[List[io.BytesIO]]:
            """
            Rasterize a PDF file to PNG images.

            Args:
                pdf (Path): The path to the PDF file.
                outpath (Optional[Path], optional): The output directory. If None, the PIL images will be returned instead. Defaults to None.
                dpi (int, optional): The output DPI. Defaults to 96.
                return_pil (bool, optional): Whether to return the PIL images instead of writing them to disk. Defaults to False.
                pages (Optional[List[int]], optional): The pages to rasterize. If None, all pages will be rasterized. Defaults to None.

            Returns:
                Optional[List[io.BytesIO]]: The PIL images if `return_pil` is True, otherwise None.
            """

            pillow_images = []
            if outpath is None:
                return_pil = True
            try:
                if isinstance(pdf, (str, Path)):
                    pdf = fitz.open(pdf)
                if pages is None:
                    pages = range(len(pdf))
                for i in pages:
                    page_bytes: bytes = pdf[i].get_pixmap(dpi=dpi).pil_tobytes(format="PNG")
                    if return_pil:
                        pillow_images.append(io.BytesIO(page_bytes))
                    else:
                        with (outpath / ("%02d.png" % (i + 1))).open("wb") as f:
                            f.write(page_bytes)
            except Exception:
                pass
            if return_pil:
                return pillow_images
        images = rasterize_paper(pdf=filepath, return_pil=True)
        
        # Maybe see with that later if it works correctly or not
        image = Image.open(images[0])
        
        # prepare image for the model
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        #print(pixel_values.shape)
        generated = processor.batch_decode(outputs[0], skip_special_tokens=True)[0]
        generated = processor.post_process_generation(generated, fix_markdown=False)
        #print(generated)
        return generated
    
        
    def split_markdown(self, md_path:str):
        markdown_document = open(md_path, "r").read()
        
        # Maybe add header/update
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]

        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        md_header_splits = markdown_splitter.split_text(markdown_document)

        # Char-level splits
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = 1000
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        splits = text_splitter.split_documents(md_header_splits)
        return splits
        
    def embedding_chunk(self, chunks):
        if self.folder_path is None:
            raise ValueError("Please set the folder path using the set_folder_path method.")
        self.vectorstore = Chroma.from_documents(documents=chunks, embedding=self.embedding, persist_directory=self.folder_path)
        
    def query_rag(self, question:str, **kwags) -> None:
        try:
            docs = self.vectorstore.similarity_search(question)
            from langchain.chains import RetrievalQA
            qachain=RetrievalQA.from_chain_type(self.llm, retriever=self.vectorstore.as_retriever(), verbose=True)
            res = qachain.invoke({"query": question})
            print(res['result'])
            return res['result']


        except Exception as e:
            print(e)


            
