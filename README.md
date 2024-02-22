# Summary Project
This repo is used for our Summarization team for LING 575 Course at UW Seattle.

## Setup

To set up the project environment, follow the steps below:

1. Navigate to the "setup" folder using the command line:

   ```bash
   cd setup
   ```

2. Change the permission of the create_env.sh script to make it executable:

   ```bash
   chmod +x create_env.sh
   ```

3. Run the create_env.sh script to create the conda environment:

   ```bash
   ./create_env.sh
   ```

4. Activate the newly created environment:
   ```bash
   conda activate Summary
   ```

## Components

- Preprocessing

  - File: `src/doc_processor/doc_processing.py`
  - This file contains functions and code for preprocessing raw data, including cleaning, formatting, and transforming the data.

- Content Selection

  - File: `src/model/content_selector.py`
  - This file contains methods for selecting salient sentences for extractive summarization.

- Information Ordering

  - File: `src/model/information_orderer.py`
  - This file contains methods for coherently reordering selected content for inclusion in a summary.

- Content Realization

  - File: `src/model/content_realizer.py`
  - This file contains methods for realizing ordered content as a summary.
  
## Scripts
- `scripts/run_main.sh`: This is a script that runs the system per the parameters found in `config.json` Usage:
   ```bash
   cd scripts
  ./run_main.sh
   ```
- `scripts/proc_docset.sh`: This is a script that finds, collates, and tokenizes data stored in the `corpora/` directory on patas. *Note that `data/` is not tracked by git given the size of the files.* Usage:

    ```bash
      cd scripts
      ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"  "../data/training"  
      ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml"  "../data/devtest"
      ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml"  "../data/evaltest"
      ```

## Configs
All arguments for the system are passed through the config file (`config.json`):

- Primary Task

  - `"document_processing"`: identifies arguments associated with ingesting and processing the original ACQUAINT and ACQUAINT-2 files.
    - `"data_ingested"`: if any of the values are set to `true`, the system will load cached data from `data/`
    - `"input_xml_file"`: identifies the location on _patas_ for the XML files which identify the training, devtest, and evaltest documents.
    - `"output_dir"`: identifies the directories in which to write out preprocessed files from the corpora
  - `"model"`: arguments associated with the core summarization features of the system.
    - `"content_selection"`: arguments associated with the content selection component.
      - `"approach"`: identifies the content selection approach to use (`"tf-idf"`, `"textrank"`, or `"topic_focused"`).
      - `"additional_parameters"`: parameters associated with `"approach"`.
    - `"information_ordering"`: argument associated with the information ordering component.
      - `"approach"`: identifies the information ordering approach to use (`"random"`, `"TSP"`, or `"entity_grid"`).
  - `"evaluation"`: identifies the evaluation metrics and associated output paths for results.
  - `"output_dir"`: identifies the directory where the system summaries are written.

## License

Distributed under the Apache License 2.0. See [LICENSE](./LICENSE) for more information.

## Authors

- Ben Cote
  - Email: bpc23 at uw.edu
- Mohamed Elkamhawy
  - Email: mohame at uw.edu
- Karl Haraldsson
  - Email: kharalds at uw.edu
- Alyssa Vecht
  - Email: avecht at uw.edu
- Josh Warzecha
  - Email: jmwar73 at uw.edu

