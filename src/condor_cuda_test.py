import torch
from transformers import pipeline, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer

def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            devices.append(f'cuda:{i}')
    devices.append('cpu')
    return devices


def get_device():
    """
    Checks for GPU availability and returns the appropriate device ('cuda' or 'cpu').

    Returns:
        str: The device type. 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        # If a GPU is available, return 'cuda' to use it
        return 0
    else:
        # If no GPU is available, default to using the CPU
        return "cpu"


from transformers import AutoTokenizer, AutoModel


def get_bert_sentence_embeddings(sent_list: list, model: AutoModel, tokenizer: AutoTokenizer,
                                 device: torch.device) -> torch.Tensor:
    """
    Generate BERT embeddings for given sentences.

    Args:
    - sent_list (list): The list of sentences to encode.
    - model (AutoModel): Pre-loaded transformers model.
    - tokenizer (AutoTokenizer): Pre-loaded transformers tokenizer.
    - device (torch.device): The device to perform the computation on ('cuda' or 'cpu').

    Returns:
    - torch.Tensor: The sentence embeddings as a PyTorch tensor.
    """
    # Prepare the model and tokenizer for evaluation
    model.eval()
    model.to(device)

    # Tokenize the sentences and prepare input tensors
    inputs = tokenizer(sent_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Disable gradient computation for inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Obtain the embeddings from the last hidden state
    # Here, we take the embeddings of the [CLS] token of each sentence in the batch
    sentence_embeddings = outputs.last_hidden_state[:, 0, :]
    return sentence_embeddings

def get_sentence_embeddings(sentlist, tokenizer, model, device):
    """
    Retrieves the sentence embeddings for a list of sentences using the provided tokenizer and model.

    Args:
        sentlist (list): A list of sentences (str) to be embedded.
        tokenizer: A tokenizer object capable of tokenizing the sentences.
        model: A pre-trained model capable of generating sentence embeddings.

    Returns:
        list: A list of sentence embeddings generated by the model.
    """

    embeddings = []
    with torch.no_grad():
        tokenized_batch = tokenizer(sentlist, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized_batch['input_ids'].to(device)
        attention_mask = tokenized_batch['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)

        # Extract the embeddings of the [CLS] token (first token of each sentence)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings.extend(cls_embeddings.cpu().numpy())
    return embeddings


# device = torch.device(0 if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# model = AutoModel.from_pretrained('distilbert-base-uncased')
#
sentence = ["Your example sentence here.", "Another example here."]
# embeddings = get_bert_sentence_embeddings(sentence, model, tokenizer, device)
# # embeddings = get_sentence_embeddings(sentence, tokenizer, model, device)
# #
# device_list = get_available_devices()
# device = get_device()


# # Create the pipeline
# sentiment_pipeline = pipeline(task="sentiment-analysis", device=device)
#
# # Use the pipeline for inference
# results = sentiment_pipeline("I love using Hugging Face Transformers!")

model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(sentence)

with open('gpu7.test', 'w', encoding='utf-8') as outfile:
    # outfile.write(f"device list: {device_list}\n")
    # outfile.write(f"device name: {device}\n")
    # # outfile.write(f"res:{results}\n")
    # outfile.write(f"embeddings: {embeddings}\n")
    outfile.write("made it to end")