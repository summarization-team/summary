import json
import os
from doc_processor.doc_processor import DocumentProcessor
from model.content_selector import ContentSelector
from model.information_orderer import InformationOrderer
from model.content_realizer import ContentRealizer, get_realization_info
from tqdm import tqdm


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
    content_selector = ContentSelector(
        config['model']['content_selection']['additional_parameters']['num_sentences_per_doc'],
        config['model']['content_selection']['approach'])
    information_orderer = InformationOrderer()
    content_realizer = ContentRealizer(
        get_realization_info(config['model']['content_realization'])
    )

    for mode in data_ingested:
        if not data_ingested[mode]:
            continue
        for doc_set in tqdm(docsets[mode], desc=f"Summarizing {mode}"):
            selected_content = \
                content_selector.select_content(docsets[mode][doc_set])
            ordered_content = information_orderer.order_content(selected_content)
            summary = content_realizer.realize_content(ordered_content)
            if 'SUMMARY' not in docsets[mode][doc_set]:
                docsets[mode][doc_set]['SUMMARY'] = summary
            else:
                raise Warning(f"Summary already exists for {doc_set}")

    # Output results
    output_dir = config['output_dir']
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    approach =  config['model']['content_selection']['approach']
    unique = 1 if approach == 'tfidf' else 2

    for mode, data in docsets.items():
        for full_dir, content in data.items():
            parent_dir = full_dir.split('/')[-1]
            summary = content['SUMMARY']
            topic_id = parent_dir.split('-')[0]
            id_part1, id_part2 = topic_id[:-1], topic_id[-1]
            output_filepath = output_dir + f'/{id_part1}-A.M.100.{id_part2}.{unique}'
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for sentence in summary:
                    outfile.write(sentence + '\n')


if __name__ == "__main__":
    config = load_config(os.path.join('..', 'config.json'))
    main(config)
