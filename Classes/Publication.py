import json
import ollama
class Publication:
    def __init__(self,publication_filled, llm_use:bool) -> None:
        self.publication_filled = publication_filled
        
        self.title = self.get_publication_title()
        self.abstract = self.get_publication_abstract().lower()
        self.author = self.get_author_name()
        self.year = self.get_year()
        self.url = self.get_publication_url()
        self.citation = self.get_citation()
        self.topic = self.get_topic() if not llm_use else self.get_topic_LLM()
        
    def get_topic(self,output_file="json/ouput.json", # à voir cette histoire avec get_topic et __get_topic
                  input_file="json/response.json"):
        try:
            with open(output_file,'r') as file:
                data = json.load(file)
            return data[self.author]['topic']
        except Exception as e:
            return self.__get_topic(input_file)
        
    def get_publication_url(self):
        return self.publication_filled['pub_url']
    
    def get_publication_title(self):
        return self.publication_filled['bib']['title'] 

    def get_publication_abstract(self):
        return self.publication_filled['bib']['abstract']

    def get_author_name(self):
        return self.publication_filled['bib']['author']

    def get_year(self):
        return self.publication_filled['bib']['pub_year']
    
    def get_citation(self):
        return self.publication_filled['bib']['citation']
    
    def get_topic_LLM(self):
        with open("json/response.json", 'r') as file:
            data = json.load(file)
        
        template = f"""
        Here is a text: {self.abstract} Please identify the topics from the following list: {data.keys()}. If the topic is not found in the list, please determine the topic.
                    
        Output: I would like the topic(s) formatted as follows:
                    Topic1:: Keyword1, Keyword2, Keyword3;
                    Topic2:: Keyword1, Keyword2, Keyword3
        For example:
                    ML:: Machine Learning, Deep Learning, Neural Networks;
                    Batteries:: Lithium-ion, Solid-state
        Note: A single text can belong to multiple topics, so please list all relevant topics. Ensure that the keywords are extracted directly from the text. Don't use this symbol ":" anywhere in the output text except when it's required to avoid confusion"""
                    
        response = ollama.generate(model='llama3', prompt=template)['response']
        dico = {}
    
        
        for i in range(response.count(";")):
            try:
                var = response.split(";")[i]
                #print(f'var: {var}\n --------')
                topic, keywords = var.split("::")[0].split(":")[1], var.split("::")[1].split(",") # Enlever le `\n` et ne garder que la partie importante 
                dico[topic.replace("\n","")] = keywords
                #print(f' topic: {topic}, keywords: {keywords}\n --------')
            except Exception as e:
                try:
                    topic, keywords = var.split("::")[0], var.split("::")[1].split(",") # Enlever le `\n` et ne garder que la partie importante 
                    dico[topic.replace("\n","")] = keywords
                    #print(f' topic: {topic}, keywords: {keywords}\n --------')
                except Exception as e:
                    #print(f"Couldn't split the topic and keywords: {e}")
                    continue
                
        with open("json/response.json", 'r') as file:
            data = json.load(file)
        for key in dico:
            try:
                data[key]['keywords'] += list(map(lambda x: x.lower(),dico[key]))
                data[key]['keywords'] = list(set(data[key]['keywords']))
            except Exception as e:
                data[key]['keywords'] = list(map(lambda x: x.lower(),dico[key]))
                
        with open("json/response.json", 'w') as file:
            json.dump(data, file)
            
        return dico
        
                
    def __get_topic(self
            , file):
        
        topics = []
        with open(file, 'r') as file:
            data = json.load(file)
        
        for category, item in data.items():
            for keyword in item['keywords']:
                if keyword in self.abstract:
                    topics.append(category)
                if keyword in self.title:
                    topics.append(category)
    
                    
        return list(set(topics))
                
        
    
        
