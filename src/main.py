import json
import os
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

    doc_config = config['document_processing']

    # Document Processing
    input_xml_file = doc_config['input_xml_file']
    output_dir = doc_config['output_dir']
    data_ingested = doc_config.get('data_ingested', False)

    doc_processor = DocumentProcessor(input_xml_file, output_dir, data_ingested)
    docsets = doc_processor.load_or_process_documents()

    # Summarization
    content_selector = ContentSelector(config['model']['content_selection']['approach'])
    information_orderer = InformationOrderer()
    content_realizer = ContentRealizer()

    summaries = []
    for mode in data_ingested:
        if not data_ingested[mode]:
            continue
        for doc_set in docsets[mode]:
            selected_content = content_selector.select_content(docsets[mode][doc_set])
            ordered_content = information_orderer.order_content(selected_content)
            summary = content_realizer.realize_content(ordered_content)
            summaries.append(summary)


    # Output results
    # ...

if __name__ == "__main__":
    config = load_config(os.path.join('..', 'config.json'))
    main(config)
