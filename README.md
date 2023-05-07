# readme-gpt

The `main.py` script is a Python script that performs language model evaluation on text data. It provides a command-line interface for running various tasks on the text data.

To use this script, you will need to have dependencies from `requirements.txt` installed.

## Repo mode

```
python main.py --repo repo/path --model model/path
```

## File mode

```
python main.py --file file/path --model model/path
```

## Command mode

```
python main.py --cmd "git diff main" --model model/path --prompt "Summarize this diff into a commit message'"
```

### What is the purpose of the "--readme" argument?
The "--readme" argument specifies the name of the file where the summary should be written. By default, it is set to "README.md".

### How can you use the "--prompt" argument to generate a summary of the given file?
You can use the "--prompt" argument to specify the text that will be used as input to the language model. In this case, the prompt is "Summarize the file in README.md format". This prompt tells the language model what kind of output is expected from it.

### What does the "--qa" argument do?
The "--qa" argument is an optional flag that allows you to run the language model in question and answer (QA) mode. When running in QA mode, the language model will iterate through a list of predefined questions.

### Sample generated output

This file contains a Python script that uses the `langchain` library to create an embedding model for text classification tasks. The script defines a `ServiceContext` object which includes an `LLMPredictor`, an `LangchainEmbedding`, and a `PromptHelper`. The `LLMPredictor` is used to generate predictions from the model, while the `LangchainEmbedding` provides the necessary embeddings for the model. Finally, the `PromptHelper` is used to handle user input during inference.

To use this script, you will need to install the required libraries (`langchain`, `llama_index`, and `huggingface-embeddings`) using pip or another package manager. You will also need to provide a path to your own trained language model as well as any other configuration parameters specified in the script.

Once you have set up the environment, you can run the script by calling the `get_service_context()` function with the path to your trained language model as an argument. This will return a `ServiceContext` object which you can use to generate predictions from your model.
