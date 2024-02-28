import json
import os
import re

from rouge_score import rouge_scorer
from tqdm import tqdm

from doc_processor.doc_processor import DocumentProcessor
from model.content_realizer import ContentRealizer, get_realization_info
from model.content_selector import ContentSelector
from model.information_orderer import InformationOrderer

RESULTS_FILE_NAME = 'rouge_scores-{}.out'
RECALL = 'R'
PRECISION = 'P'
F1 = 'F'
AVERAGE_R = 'Average_R'
AVERAGE_P = 'Average_P'
AVERAGE_F1 = 'Average_F'


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)


def output_results(docsets, output_dir):
    """
    Output the results of the summarization process to a directory.
    Args:
        docsets (dict): A dictionary of document sets, where each document set is represented as a dictionary
            with the following keys:
                - 'FULL': The full document set.
                - 'REDUCED': The reduced document set.
                - 'SUMMARY': The summary of the document set.
        output_dir (str): The directory to which the results should be written.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    unique = f"{config['model']['content_selection']['approach']}-{config['model']['information_ordering']['approach']}-{config['model']['content_realization']['method']}"

    for mode, data in docsets.items():
        for full_dir, content in data.items():
            parent_dir = full_dir.split('/')[-1]
            summary = content['SUMMARY']
            topic_id = parent_dir.split('-')[0]
            id_part1, id_part2 = topic_id[:-1], topic_id[-1]
            output_filepath = os.path.join(output_dir, f'{id_part1}-A.M.100.{id_part2}.{unique}')
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for sentence in summary:
                    outfile.write(sentence + '\n')


def calculate_rouge_scores(metrics, docsets, mode, reference_summaries_path, results_dir):
    """
    Calculate ROUGE scores for the generated summaries.
    Args:
        metrics (list): A list of ROUGE metrics to calculate.
        docsets (dict): A dictionary of document sets, where each document set is represented as a dictionary
            with the following keys:
                - 'FULL': The full document set.
                - 'REDUCED': The reduced document set.
                - 'SUMMARY': The summary of the document set.
        mode (str): The mode for which the ROUGE scores are calculated.
        reference_summaries_path (str): The path to the reference summaries.
        results_dir (str): The directory to which the results should be written.
    """
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    gold_summaries_files_names = os.listdir(reference_summaries_path)
    
    scores_dict = {metrics[0]: {AVERAGE_R:0, AVERAGE_P: 0, AVERAGE_F1: 0}, 
                   metrics[1]: {AVERAGE_R:0, AVERAGE_P: 0, AVERAGE_F1: 0}}
    approach = config['model']['content_selection']['approach']
    if approach == 'tfidf':
        unique = 1
    elif approach == 'textrank':
        unique = 2
    else:
        unique = 3
    results_path = os.path.join(results_dir, RESULTS_FILE_NAME.format(unique))
    summary_file_names = []

    for dir in docsets[mode]:
        parent_dir = dir.split('/')[-1]
        topic_id = parent_dir.split('-')[0]
        id_part1, id_part2 = topic_id[:-1], topic_id[-1]
        summary_file_name = f'{id_part1}-A.M.100.{id_part2}'
        summary_file_names.append(summary_file_name)
        generated_summary = '\n'.join(docsets[mode][dir]['SUMMARY'])
        
        if summary_file_name not in scores_dict[metrics[0]]:
            scores_dict[metrics[0]][summary_file_name] = {'R': 0, 'P': 0, 'F': 0}
            scores_dict[metrics[1]][summary_file_name] = {'R': 0, 'P': 0, 'F': 0}

        matching_filenames = [filename for filename in gold_summaries_files_names if re.match(f'^{summary_file_name}\.[^.]+$', filename)]
        gold_summary_path = ''
        for filename in matching_filenames:
            gold_summary_path = os.path.join(reference_summaries_path, filename)
            with open(gold_summary_path, 'r', encoding='cp1252') as gold_file:
                gold_summary = gold_file.read().strip()
                scores = scorer.score(gold_summary, generated_summary)
                scores_dict[metrics[0]][summary_file_name]['R'] += scores[metrics[0]].recall
                scores_dict[metrics[0]][summary_file_name]['P'] += scores[metrics[0]].precision
                scores_dict[metrics[0]][summary_file_name]['F'] += scores[metrics[0]].fmeasure
                scores_dict[metrics[1]][summary_file_name]['R'] += scores[metrics[1]].recall
                scores_dict[metrics[1]][summary_file_name]['P'] += scores[metrics[1]].precision
                scores_dict[metrics[1]][summary_file_name]['F'] += scores[metrics[1]].fmeasure
        scores_dict[metrics[0]][summary_file_name]['R'] /= len(matching_filenames)
        scores_dict[metrics[0]][summary_file_name]['P'] /= len(matching_filenames)
        scores_dict[metrics[0]][summary_file_name]['F'] /= len(matching_filenames)
        scores_dict[metrics[0]][AVERAGE_R] += scores_dict[metrics[0]][summary_file_name]['R']
        scores_dict[metrics[0]][AVERAGE_P] += scores_dict[metrics[0]][summary_file_name]['P']
        scores_dict[metrics[0]][AVERAGE_F1] += scores_dict[metrics[0]][summary_file_name]['F']
        scores_dict[metrics[1]][summary_file_name]['R'] /= len(matching_filenames)
        scores_dict[metrics[1]][summary_file_name]['P'] /= len(matching_filenames)
        scores_dict[metrics[1]][summary_file_name]['F'] /= len(matching_filenames)
        scores_dict[metrics[1]][AVERAGE_R] += scores_dict[metrics[1]][summary_file_name]['R']
        scores_dict[metrics[1]][AVERAGE_P] += scores_dict[metrics[1]][summary_file_name]['P']
        scores_dict[metrics[1]][AVERAGE_F1] += scores_dict[metrics[1]][summary_file_name]['F']
    scores_dict[metrics[0]][AVERAGE_R] /= len(docsets[mode])
    scores_dict[metrics[0]][AVERAGE_P] /= len(docsets[mode])
    scores_dict[metrics[0]][AVERAGE_F1] /= len(docsets[mode])
    scores_dict[metrics[1]][AVERAGE_R] /= len(docsets[mode])
    scores_dict[metrics[1]][AVERAGE_P] /= len(docsets[mode])
    scores_dict[metrics[1]][AVERAGE_F1] /= len(docsets[mode])
    
    with open(results_path, 'w') as results_file:
        for metric in scores_dict.keys():
            results_file.write(f'----------------------------------------------\n')
            results_file.write(f'{unique} {metric.upper()} {AVERAGE_R}: {scores_dict[metric][AVERAGE_R]}\n')
            results_file.write(f'{unique} {metric.upper()} {AVERAGE_P}: {scores_dict[metric][AVERAGE_P]}\n')
            results_file.write(f'{unique} {metric.upper()} {AVERAGE_F1}: {scores_dict[metric][AVERAGE_F1]}\n')
            results_file.write(f'..............................................\n')
            for summary_file_name in summary_file_names:
                 results_file.write(f'{unique} {metric.upper()} Eval {summary_file_name}  R:{scores_dict[metric][summary_file_name][RECALL]} P:{scores_dict[metric][summary_file_name][PRECISION]} F:{scores_dict[metric][summary_file_name][F1]} \n')


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
        config['model']['content_selection']['approach'],
        config['model']['content_selection']['additional_parameters']['similarity_threshold'])
    information_orderer = InformationOrderer(config['model']['information_ordering'])
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
    output_results(docsets=docsets, output_dir=config['output_dir'])

    # Calculate ROUGE scores
    metrics = config['evaluation']['metrics']
    reference_summaries_path = config['evaluation']['reference_summaries_path']
    results_dir = config['evaluation']['results_dir']
    data_subset = config['evaluation']['data_subset']
    calculate_rouge_scores(metrics, docsets, data_subset, reference_summaries_path, results_dir)


if __name__ == "__main__":
    config = load_config(os.path.join('config.json'))
    main(config)
