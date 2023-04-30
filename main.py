# How to use: python3 main.py --repo repo/path --model model/path > README.md :)
# Saves json file to query again
# Query index: python3 main.py --repo repo/path --query 'help'
# Customize readme prompt: python3 main.py --repo repo/path 
import os
from subprocess import Popen, PIPE
import argparse
import tiktoken
from langchain.llms import LlamaCpp
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import TokenTextSplitter
from adapter import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, Document, download_loader, GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, Document, LangchainEmbedding
from llama_index.node_parser import SimpleNodeParser

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=str, required=False)
parser.add_argument('--read', type=bool, required=False, default=False)
parser.add_argument('--save', type=str, required=False, default=False)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--query', type=str, required=False)
parser.add_argument('--prompt', type=str, required=False,
                    default="Summarize these files into a README document")
args = parser.parse_args()

# define the local llama model
llm_predictor = LLMPredictor(
        llm=LlamaCpp(
                model_path=args.model, 
                n_ctx=2048, 
                use_mlock=True, 
                top_k=1000, 
                max_tokens=800, 
                n_parts=-1, 
                temperature=0.8, 
                top_p=0.40,
                last_n_tokens_size=100,
                n_threads=6,
                f16_kv=True
            )
        )


embeddings = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(embeddings)

node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter())
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, embed_model=embed_model, node_parser=node_parser,
    prompt_helper=prompt_helper
)

if __name__ == "__main__":
    path_components = args.repo.rsplit(os.path.sep, 1)
    project_name = path_components[-1]
    project_json = f"{project_name}.json"
    if (os.path.isfile(project_name)):
        document = Document(args.repo)
        index = GPTListIndex.from_documents([document], service_context=service_context)
    elif (args.read is True and os.path.isfile(project_json)):
        index = GPTListIndex.load_from_disk(project_json, service_context=service_context)
    else:
        documents = SimpleDirectoryReader(args.repo, exclude_hidden=True, recursive=True).load_data()
        index = GPTListIndex.from_documents(documents, service_context=service_context)
    
    output = index.query(args.prompt, 
        service_context=service_context, mode="embedding", response_mode="compact"
    )
    readme = f"# {project_name}\n\n${output}"
    if (args.save is True):
        index.save_to_disk(project_json)
    print(readme)
