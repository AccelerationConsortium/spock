from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI



class Spock_GraphGenerator:
    
    def __init__(self, llm):
        self.llm = llm if isinstance(llm, ChatOpenAI) else ChatOpenAI(model=llm)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)


        
    async def agenerate_graph(self, text):
        """
        Generates a graph from the documents using the language model and prompt template.
        """
        documents = [Document(page_content=text)]
        graph_documents = await self.llm_transformer.aconvert_to_graph_documents(documents)
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")
        return {'nodes': graph_documents[0].nodes, 'relationships': graph_documents[0].relationships}


    def generate_graph(self, text):
        pass


    def visualize_graph(self, graph):
        """
        Visualizes the generated graph by generating a picture of the graph.
        """
        pass