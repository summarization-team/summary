import json
import os
from doc_processor.doc_processor import DocumentProcessor
from model.content_selector import ContentSelector
from model.information_orderer import InformationOrderer
from model.content_realizer import ContentRealizer

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def main(config):

    doc_config = config['document_processing']

    # Document Processing
    input_xml_file = doc_config['input_xml_file']
    output_dir = doc_config['output_dir']
    data_ingested = doc_config.get('data_ingested', False)

    doc_processor = DocumentProcessor(input_xml_file, output_dir, data_ingested)
    docsets = doc_processor.load_or_process_documents()

    # Summarization
    content_selector = ContentSelector(config['model']['content_selection']['additional_parameters']['num_sentences_per_doc'],
                                       config['model']['content_selection']['approach'])
    information_orderer = InformationOrderer()
    content_realizer = ContentRealizer()

    summaries = []
    for mode in data_ingested:
        if not data_ingested[mode]:
            continue
        for doc_set in docsets[mode]:
            selected_content = \
                content_selector.select_content(docsets[mode][doc_set])
            ordered_content = information_orderer.order_content(selected_content)
            summary = content_realizer.realize_content(ordered_content)
            summaries.append(summary)


    # Output results
    output_dir = config['output_dir']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    i = 0
    for summary in summaries[:5]:
        # Output file name: [id_part1]-A.M.100.[id_part2].[unique_integer]
        # Given topic ID, `id_part1` is first five char and `id_part2` is final.
        # Unique integer comes from content selection method.
        # e.g., D0901A -> D0901-A.M.100.A.{1,2}
        # topic_id = parent_dir.split('-')[0]
        # id_part1, id_part2 = topic_id[:-1], topic_id[-1]
        # output_filepath = output_dir + f'/{id_part1}-A.M.100.{id_part2}'
        output_filepath = output_dir + '/' + str(i)
        with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(summary)
        i += 1


if __name__ == "__main__":
    config = load_config(os.path.join('..', 'config.json'))
    main(config)
