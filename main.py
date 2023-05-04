import os
from subprocess import PIPE
import argparse
from langchain.llms import LlamaCpp
from langchain.text_splitter import CharacterTextSplitter
from adapter import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, download_loader, GPTListIndex, LLMPredictor, PromptHelper, ServiceContext, Document, LangchainEmbedding, ResponseSynthesizer
from llama_index.indices.postprocessor import SimilarityPostprocessor
from langchain.chains import ConversationalRetrievalChain
from llama_index.retrievers import VectorIndexRetriever
from langchain.retrievers.llama_index import LlamaIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.node_parser import SimpleNodeParser
from llama_index.data_structs import Node

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 1024
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

parser = argparse.ArgumentParser()
parser.add_argument('--repo', type=str, required=False)
parser.add_argument('--file', type=str, required=False)
parser.add_argument('--read', type=str, required=False)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--query', type=str, required=False)
parser.add_argument('--prompt', type=str, required=False,
                    default="Summarize the files in a README style")
parser.add_argument('--readme', type=str, default="README.md")
parser.add_argument('--qa', type=bool, required=False)
args = parser.parse_args()


def query_llm(index, prompt, service_context, retriever_mode='embedding', response_mode='tree_summarize'):
    custom_retriever = VectorIndexRetriever(
        index=index, 
        similarity_top_k=2,
        service_context=service_context
    )
    response_synthesizer = ResponseSynthesizer.from_args(
        service_context=service_context,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.7)
        ]
    )
    retriever = index.as_retriever(retriever_mode=retriever_mode, service_context=service_context)
    query_engine = RetrieverQueryEngine.from_args(retriever, response_synthesizer=response_synthesizer, response_mode=response_mode,  service_context=service_context)
    return query_engine.query(prompt)


def readme_generator(qa, questions: [
    "What does this project do?",
    "How do I use this project?",
    "Where can I find more information about related information in this proejct?"
    "Who are the contributors to this project?",
    "What is the motivation for this project?"
]):
    for question in questions:  
        result = qa({"question": question, "chat_history": chat_history})
        output = output + (f"## {question} \n{result['answer']} \n")
    return output

if __name__ == "__main__":
    llama = LlamaCpp(
        model_path=args.model, 
        n_ctx=4096, 
        max_tokens=600, 
        n_parts=-1, 
        temperature=0.8, 
        top_p=0.40,
        last_n_tokens_size=400,
        n_threads=8,
        f16_kv=True,
        use_mlock=True
    )
    llm_predictor = LLMPredictor(
        llm=llama
    )
    embeddings = HuggingFaceEmbeddings()
    embed_model = LangchainEmbedding(embeddings)
    node_parser = SimpleNodeParser(text_splitter=CharacterTextSplitter(chunk_size=1000))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model, node_parser=node_parser,
        prompt_helper=prompt_helper
    )

    if args.read:
        index = GPTListIndex.load_from_disk(args.read, service_context=service_context)
    if args.repo:
        repo_loader = download_loader("GPTRepoReader")
        loader = repo_loader()
        path_components = args.repo.rsplit(os.path.sep, 1)
        project_name = path_components[-1]
        project_path = path_components[0]
        documents = loader.load_data(args.repo)
        index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
        output = query_llm(index, args.prompt, service_context)
        readme = f"# {project_name}\n\n${output}"
    if args.file:
        with open(args.file, 'r') as file:
            document = file.read()
        index = GPTVectorStoreIndex.from_documents([Node(document)], service_context=service_context)
        path_components = args.file.rsplit(os.path.sep, 1)
        project_name = path_components[-1]
        project_path = path_components[0]
        output = query_llm(index, prompt, service_context)
        readme = f"### {project_name}\n\n${output}"
    if args.qa:
        chat_history = []
        retriever = LlamaIndexRetriever(index=index)
        qa = ConversationalRetrievalChain.from_llm(llama, retriever=retriever)
        output = readme_generator(qa)
    
    print(readme)
    with open(f"{project_path}/{args.readme}", 'a') as file:
        file.write(readme)
        file.close()
