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

  - File: `doc_processing.py`
  - This file contains functions and code for preprocessing raw data, including cleaning, formatting, and transforming the data.

## Scripts
- `scripts/proc_docset.sh`: This is a script that finds, collates, and tokenizes data stored in the `corpora/` directory on patas. *Note that `data/` is not tracked by git given the size of the files.* Usage:

```bash
  cd scripts
  ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/training/2009/UpdateSumm09_test_topics.xml"  "../data/training"  
  ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/devtest/GuidedSumm10_test_topics.xml"  "../data/devtest"
  ./proc_docset.sh "/mnt/dropbox/23-24/575x/Data/Documents/evaltest/GuidedSumm11_test_topics.xml"  "../data/evaltest"
  ```