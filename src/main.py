import json
from doc_processor.doc_processor import DocumentProcessor
from model.content_selector import ContentSelector
from model.information_orderer import InformationOrderer
from model.content_realizer import ContentRealizer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def get_realization_strategy(method_name):
    if method_name == 'simple':
        return SimpleJoinMethod()
    elif method_name == 'sentence_compression':
        return SentenceCompressionMethod()
    else:
        raise ValueError("Unknown realization strategy: {}".format(method_name))

def main(config):
    # Document Processing
    doc_processor = DocumentProcessor(config['document_processing']['input_xml_file'],
                                      config['document_processing']['output_dir'])
    docsets = doc_processor.process_documents()

    # Summarization
    content_selector = ContentSelector(config['model']['content_selection']['approach'])
    information_orderer = InformationOrderer()
    content_realizer = ContentRealizer()

    summaries = []
    for docset in docsets:
        selected_content = content_selector.select_content(docset)
        ordered_content = information_orderer.order_content(selected_content)
        summary = content_realizer.realize_content(ordered_content)
        summaries.append(summary)

    # Output results
    # ...

if __name__ == "__main__":
    config = load_config('config.json')
    main(config)
