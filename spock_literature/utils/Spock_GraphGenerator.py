from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import asyncio
import networkx as nx
import matplotlib.pyplot as plt


# TODO: finish the graph generation thing
load_dotenv()
 
class Spock_GraphGenerator():
    def __init__(self, llm):
        self.llm = llm if isinstance(llm, ChatOpenAI) else ChatOpenAI(model=llm)
        self.llm_transformer = LLMGraphTransformer(llm=self.llm)
        
    async def agenerate_graph(self, text):
        """
        Generates a graph from the documents using the language model and prompt template,
        then returns a dict with:
          - 'nodes': list of Node(...) objects
          - 'relationships': list of Relationship(...) objects
        """
        documents = [Document(page_content=text)]
        graph_documents = await self.llm_transformer.aconvert_to_graph_documents(documents)
        return {
            "nodes": graph_documents[0].nodes,
            "relationships": graph_documents[0].relationships
        }

    def generate_graph(self, text):
        """Synchronous wrapper if you want to call from non‐async code."""
        return asyncio.run(self.agenerate_graph(text))

    def visualize_graph(self, graph: dict, output_path: str = "spock_graph.png"):
        """
        Build a NetworkX DiGraph from the LLM output and save it to disk with more spacing.
        """
        node_ids = [node.id for node in graph["nodes"]]
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        edge_labels = {}
        for rel in graph["relationships"]:
            src_id = rel.source.id
            tgt_id = rel.target.id
            rel_type = rel.type
            G.add_edge(src_id, tgt_id)
            edge_labels[(src_id, tgt_id)] = rel_type
        n = G.number_of_nodes()
        k_val = 2.0  
        pos = nx.spring_layout(G, k=k_val, iterations=100, seed=42)
        plt.figure(figsize=(14, 14))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=1200,
            font_size=10,
            arrowsize=15,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            label_pos=0.5,
        )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Graph image saved to {output_path}")
        plt.close()


    
    
if __name__ == "__main__":
    llm = ChatOpenAI(model="gpt-4o-mini")
    graph_generator = Spock_GraphGenerator(llm)
    text = """
    ## 2. MATERIALS AND METHODS

2.1. Data Sets. 2.1.1. Hemolysis. Hemolysis is defined as the disruption of erythrocyte membranes that decrease the life span of red blood cells and causes the release of hemoglobin. Identifying nonhemolytic antimicrobial is critical to their applications as nontoxic and safe measurements against bacterial infections. However, distinguishing between hemolytic and nonhemolytic peptides is complicated, as they primarily exert their activity at the charged surface of the bacterial plasma membrane. Timmons and Hewage 47 differentiate between the two whether they are active at the zwitterionic eukaryotic membrane, as well as the anionic prokaryotic membrane. In this work, the model for hemolytic prediction is trained using data from the Database of Antimicrobial Activity and Structure of Peptides (DBAASP v3 51 ). The activity is defined by extrapolating a measurement assuming dose response curves to the point at which 50% of red blood cells (RBC) are lysed. If the activity is below 100 μ g/mL, it is considered hemolytic. Each measurement is treated independently, so sequences can appear multiple times. The training data contains 9316 sequences (19.6% positives and 80.4% negatives) of only L- and canonical amino acids. Note that due to the inherent noise in the experimental data sets used, in some observations ( ∼ 40%), an identical sequence appears in both negative and positive class. As an example, sequence 'RVKRVWPLVIRTVIAGYNLYRAIKKK', is found to be both hemolytic and nonhemolytic in two different lab experiments (i.e., two training examples).

2.1.2. Solubility. The training data contains 18,453 sequences (47.6% positives and 52.4% negatives) based on data from PROSO II. 46 Solubility was estimated by retrospective analysis of electronic laboratory notebooks. The notebooks were part of a large effort called the Protein Structure Initiative and consider sequences linearly through the following stages: Selected, Cloned, Expressed, Soluble, Purified, Crystallized, HSQC (heteronuclear single quantum coherence), Structure, and deposited in PDB. 52 The peptides were identified as soluble or insoluble as described in ref 46: 'comparing the experimental status at two time points, September 2009 and May 2010; we were able to derive a set of insoluble proteins defined as those which were not soluble in September 2009 and still remained in that state 8 months later'.

2.1.3. Nonfouling. Data for predicting resistance to nonspecific interactions (nonfouling) are obtained from ref 35. Positive data contains 3600 sequences. Negative examples are based on 13,585 sequences (20.9% positives and 79.1% negatives) coming from insoluble and hemolytic peptides, as well as the scrambled positives. The scrambled negatives are generated with lengths sampled from the same length range as their corresponding positive set, and residues were sampled from the frequency distribution of the soluble data set. Samples are weighted to account for the class imbalance caused by the data set size for negative examples. A nonfouling peptide (positive example) is defined using the mechanism proposed by White et al. 53 Briefly, White et al. showed that the exterior surfaces of proteins have a significantly different frequency of amino acids, and this increases in aggregation prone environments, like the cytoplasm. Synthesizing self-assembling peptides that follow this amino acid distribution and coating surfaces with the peptides creates nonfouling surfaces. This pattern was also found inside chaperone proteins, another area where resistance to nonspecific interactions is important. 54

2.2. Model Architecture. To identify the positioninvariant patterns in the peptide sequences, we build a recurrent neural network (RNN), using a sequential model from Keras framework 55 and the TensorFlow deep learning library back-end. 56 Specifically, the RNN employs bidirectional Long Short-term Memory (LSTM) networks to capture longrange sequence correlations. Compared to the conventional RNNs, LSTM networks with gate control units (input gate, forget gate, and output gate) can learn dependency information between distant residues within peptide sequences more effectively. 57 -59 They can also partly overcome the problem of vanishing or exploding gradients in the backpropagation phase of training conventional RNNs. 60 We use a bidirectional LSTM (bi-LSTM) to enhance the capability of our model in learning bidirectional dependence between Nterminal and C-terminal amino acid residues. An overview of the RNN architecture is shown in Figure 2.

Peptide sequences are represented as integer-encoded vectors of shape 200, where the integer at each position in

the vector corresponds to the index of the amino acid from the alphabet of the 20 essential amino acids: [A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V]. The maximum length of the peptide sequence is fixed at 200, and all sequences with higher lengths are excluded. For those sequences with shorter lengths, zeros are padded to the integer encoding representation to keep the shape fixed at 200 for all examples to allow input sequences with flexible lengths. Note that this is primarily applied to the training step for implementation considerations, and the trained model can make predictions on variable-length sequences as input. Every integer-encoded peptide sequence is first fed to an embedding layer. The embedding layer enables us to convert the indices of discrete symbols (i.e., essential amino acids) into a representation of a fixed-length vector of defined size. This is beneficial in the sense of creating a more compact representation of the input symbols, as well as yielding semantically similar symbols close to one another in the vector space. This embedding layer is trainable, and its weights can be updated during training along with the others layers of the RNN.

The output from the embedding layer either goes to a double stacked bi-LSTM layer or a single LSTM layer to identify patterns along a sequence that can be separated by large gaps. The former is used in predicting solubility and hemolysis, whereas the latter is for predicting a peptide's resistance to nonspecific interactions (nonfouling). The rationale behind this choice for the nonfouling model is that the bi-LSTM layer did not contribute to a better performance when compared with the LSTM layer (same ACC and AUROC of 82% and 0.93, respectively). The output from the LSTM layer is then concatenated with the relative frequency of each amino acid in the input sequences. This choice is partially based on our earlier work, 61 and helps with improving model performance. The concatenated output is then normalized and fed to a dropout layer with a rate of 10%, followed by a dense neural network with a ReLU activation function. This is repeated three times, and the final single-node dense layer uses a sigmoid activation function to force the final prediction as a value between 0 and 1. This scalar output shows the probability of the label being positive for the corresponding predicted peptide biological activity. We use this probability to evaluate the confidence of the model in making inferences on new sequences in our web-based implementation.

The hyperparameters are chosen based on a random search that resulted in the best model performance in terms of the Area Under the Receiver Operating Characteristic (AUROC) curve 62 and accuracy. The AUROC shows the model's ability to discriminate between positive and negative examples as the discrimination threshold is varied, and the accuracy is defined as the ratio of correct predictions to the total number of predictions made by the model. The embedding layer has the same input dimension of 21 (alphabet length added by one to account for the padded zeros) and output dimension of 32. The LSTM layer has 64 units, and the first, second, and third dense layers have 64, 16, and 1 units, respectively. We train with the Adam optimizer 63 of binary cross-entropy loss function, which is defined as

$$- \frac { 1 } { N } \sum _ { i = 1 } ^ { N } [ y _ { i } \log ( \widehat { y } _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - \widehat { y } _ { i } ) ] \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
 \quad ( 1 ) \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

̂

where y i is the true value of the i th example, y i the corresponding prediction, and N the size of the data set. The learning rate is adapted using a cosine decay schedule with an initial learning rate of 10 -3 , decay steps of 50, and minimum of 10 -6 . The data split for training, validation, and test is 81%, 9%, and 10%, respectively. To avoid overfitting, we add early stopping with patience of 5 that restores model weights from the epoch with the maximum AUROC on the validation set during training.

Previous models for peptide prediction tasks use a variety of deep learning and classical machine learning methods. The prediction server PROSO II employs a two-layered structure, where the output of a primary Parzen 64 window model for sequence similarity and a logistic regression classifier of amino acid k-mer composition are fed to a second-level logistic regression classifier. HAPPENN uses normalized features selected by SVM and ensemble of Random Forests, which are fed to a deep neural network with batch normalization and dropout regularization to prevent overfitting. DSResSol (1) takes advantage of the integration of Squeeze-and-Excitation (SE) 65 residual networks 66 with dilated convolutional neural networks. 67 Specifically, the model includes five architectural units, including a single embedding layer, nine parallel initial CNNs with different filter sizes, nine parallel SE-ResNet blocks, three parallel CNNs, and fully connected layers.

."""
    graph = graph = asyncio.run(graph_generator.agenerate_graph(text))
    
    print(graph)
    graph_generator.visualize_graph(graph)
