import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize

HEADLINE = 'HEADLINE'
DATELINE = 'DATELINE'
PARAGRAPH = 'PARAGRAPH-{}'
SENTENCES = 'SENTENCES'


class DocumentProcessor:
    """
    A class for processing XML documents.

    This class provides functionalities to parse XML documents and process them
    into a structured format suitable for further NLP tasks.

    Attributes:
        input_path (str): The file path of the input XML file (in case of ingestion) OR Folder path if loading ingested data.
        output_path (str): The directory path where the processed output will be stored.
        data_ingested (bool): Flag indicating if data is already ingested.
    """

    def __init__(self, input_path, output_path, data_ingested=False):
        """
        Initializes the DocumentProcessor with the given XML file and output directory.

        This class provides functionalities to parse XML documents and process them
        into a structured format suitable for further NLP tasks.

        Args:
            input_path (str): The file path of the input XML file (in case of ingestion) OR Folder path if loading ingested data.
            output_path (str): The directory path where the processed output will be stored.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.data_ingested = data_ingested

    def load_or_process_documents(self):
        """
        Loads processed data if available, otherwise processes XML documents.

        Returns:
            list: List of processed documents or paths to processed documents.
        """
        doc_set = dict()
        for mode in self.data_ingested:
            if not self.data_ingested[mode]:
                print(f"Re-Processing Original Files: {mode}")
                self.process_documents(self.input_path[mode], mode)
            print(f"Loading Processed Data:{mode}")

            doc_set[mode] = self.load_processed_data(self.output_path[mode])
        return doc_set

    def load_processed_data(self, input_path):
        """
        Loads processed data from the specified output directory.

        Args:
            input_path (str): The directory path where the processed output is stored.

        Returns:
            list: List of paths to processed documents.
        """
        processed_docs = dict()
        for root, dirs, files in os.walk(input_path):
            if len(files) > 0:
                processed_docs[root] = dict()

            for file_name in files:
                file_path = os.path.join(root, file_name)

                if os.path.isfile(file_path):
                    processed_docs[root][file_name] = dict()
                    paragraph_count = 0
                    processed_docs[root][file_name][PARAGRAPH.format(paragraph_count)] = []
                    with open(file_path, 'r') as file:
                        file_content = file.readlines()
                        for line in file_content:
                            line = line.strip().strip("\n").strip(" ")
                            if len(line) <= 0:
                                continue
                            # if line in list format, it's a sentence that we want to analyze
                            if line[0] == "[":
                                line = line.strip("[]\n")
                                word_list = line.split(", ")
                                word_list = [word.strip("'") for word in word_list]
                                if word_list[0] == '``':
                                    paragraph_count += 1
                                    processed_docs[root][file_name][PARAGRAPH.format(paragraph_count)] = []
                                    word_list = word_list[1:]
                                processed_docs[root][file_name][PARAGRAPH.format(paragraph_count)].append(word_list)
                            elif HEADLINE in line:
                                processed_docs[root][file_name][HEADLINE] = line.replace(HEADLINE + ':', "").strip()
                            elif DATELINE in line:
                                processed_docs[root][file_name][DATELINE] = line.replace(DATELINE + ':', "").strip()
        return processed_docs

      
    def process_documents(self, input_path, mode):
        """
        Processes the XML documents by parsing and extracting relevant information.

        Reads the input XML file, parses its content, and processes it into a structured
        format (DocSets). The processed documents are then saved in the specified output directory.

        Args:
            input_path (str): The file path of the input XML file.

        Returns:
            list: A list of DocSets representing the processed documents.
        """
        # Implement XML parsing and document processing
        # Return a list of DocSets
        docsets = []
        tree = ET.parse(input_path)
        root = tree.getroot()

        for topic in root.iter('topic'):
            description = ''
            description += 'title: ' + topic.find('title').text.strip() + '\n'
            if len(topic.findall('narrative')) > 0:
                description += 'narrative: ' + topic.find('narrative').text.strip() + '\n'
    
            # Find all docsetA
            for docset in topic.iter('docsetA'):
                docsetID = docset.get('id')
                docsets.append(docsetID)
                path = self.output_path[mode] + '/' + docsetID
                
                for doc in docset:
                    docID = doc.get('id')
                    process_doc(path, docID)
                descriptionFile = os.path.join(path, 'description.txt') 
                with open(descriptionFile, 'a') as F:
                    F.write(description)

            # Find all docsetB
            for docset in topic.iter('docsetB'):
                docsetID = docset.get('id')
                docsets.append(docsetID)
                path = self.output_path[mode] + '/' + docsetID
                
                for doc in docset:
                    docID = doc.get('id')
                    process_doc(path, docID)
                descriptionFile = os.path.join(path, 'description.txt') 
                with open(descriptionFile, 'a') as F:
                    F.write(description)

        return docsets


def process_doc(dirPath, docID):
    """
    Processes a document by creating a directory, removing unwanted strings, and writing headers and tokenized text.

    Args:
        dirPath (str): The directory path where the processed document is to be saved.
        docID (str): The document identifier.

    Returns:
        None
    """
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    remove_these = ['&Cx1f', '&Cx13', '&Cx11', '&UR', '&LR', '&QL', '&HT', '&QC', '&QR', '&AMP']
    headerTags = ['DOCTYPE', 'DATE_TIME', 'DATETIME', 'CATEGORY', 'SLUG', 'HEADLINE', 'DATELINE']
    # AQUAINT
    if int(get_year(docID)) >= 1996 and int(get_year(docID)) <= 2000:
        file = get_AQUAINT_file(docID)
        docXML = get_doc_AQUAINT(docID, file)
        for x in remove_these:
            docXML = docXML.replace(x, '')
        headers = get_doc_headers_AQUAINT(docID, docXML, headerTags)
        newFile = os.path.join(dirPath, docID)
        with open(newFile, 'a') as F:
            if os.stat(newFile).st_size == 0:
                for h in headers:
                    F.write(h + '\n')
                paragraphs = separate_paragraphs(docXML)
                for p in paragraphs:
                    tokenized = tokenize_text(p)
                    F.write('\n')
                    for s in tokenized:
                        F.write(str(s) + '\n')

    # AQUAINT 2
    elif (int(get_year(docID)) >= 2004 and int(get_year(docID)) <= 2006):
        
        try:
            file = get_AQUAINT2_file(docID)
            if file.is_file():
                docXML = get_doc_AQUAINT2(docID, file)
            else: raise Exception()
        # if not found in AQUAINT 2 try 2009 corpus
        except:
            try:
                file = get_2009_file(docID)
                docXML = get_doc_2009(docID, file)
                for x in remove_these:
                    docXML = docXML.replace(x, '')
                headers = get_doc_headers_2009(docID, docXML, headerTags)
                newFile = os.path.join(dirPath, docID)
                with open(newFile, 'a') as F:
                    if os.stat(newFile).st_size == 0:
                        for h in headers:
                            F.write(h + '\n')
                        paragraphs = separate_paragraphs(docXML)
                        for p in paragraphs:
                            tokenized = tokenize_text(p)
                            F.write('\n')
                            for s in tokenized:
                                F.write(str(s) + '\n')
                return
            except:
                return

        for x in remove_these:
            docXML = docXML.replace(x, '')
        headers = get_doc_headers_AQUAINT2(docID, docXML, headerTags)
        newFile = os.path.join(dirPath, docID)
        
        with open(newFile, 'a') as F:
            if os.stat(newFile).st_size == 0:
                for h in headers:
                    F.write(h + '\n')
                paragraphs = separate_paragraphs(docXML)
                for p in paragraphs:
                    tokenized = tokenize_text(p)
                    F.write('\n')
                    for s in tokenized:
                        F.write(str(s) + '\n')
    # 2009 files
    else:
        file = get_2009_file(docID)
        docXML = get_doc_2009(docID, file)
        for x in remove_these:
            docXML = docXML.replace(x, '')
        headers = get_doc_headers_2009(docID, docXML, headerTags)
        newFile = os.path.join(dirPath, docID)
        with open(newFile, 'a') as F:
            if os.stat(newFile).st_size == 0:
                for h in headers:
                    F.write(h + '\n')
                paragraphs = separate_paragraphs(docXML)
                for p in paragraphs:
                    tokenized = tokenize_text(p)
                    F.write('\n')
                    for s in tokenized:
                        F.write(str(s) + '\n')


def get_year(docID):
    """
    Extracts the year from a document identifier.

    Args:
        docID (str): The document identifier.

    Returns:
        str: The extracted year as a four-character string.
    """
    return docID[-13:-9]


def get_AQUAINT_file(docID):
    """
    Constructs the filepath for an AQUAINT XML document based on the document identifier.

    Args:
        docID (str): The identifier of the AQUAINT document.

    Returns:
        str: The filepath of the specified AQUAINT document.
    """
    filePath = '/corpora/LDC/LDC02T31/'
    filePath += docID[:3].lower() + '/'
    filePath += get_year(docID) + '/'
    filePath += docID[-13:-5] + '_'
    if docID[:3] == 'APW':
        filePath += 'APW_ENG'
    elif docID[:3] == 'XIE':
        filePath += 'XIN_ENG'
    else:
        filePath += docID[:3]
    return filePath


def get_AQUAINT2_file(docID):
    """
    Constructs the filepath for an AQUAINT 2 XML document based on the document identifier.

    Args:
        docID (str): The identifier of the AQUAINT 2 document.

    Returns:
        str: The filepath of the specified AQUAINT 2 document.
    """
    filePath = '/corpora/LDC/LDC08T25/data/'
    filePath += docID[:7].lower() + '/'
    if docID[:3] == 'XIE':
        filePath += docID[4:-5] + 'XIN_ENG'
    else:
        filePath += docID[:7].lower() + '_' + docID[-13:-7] + '.xml'
        
    return filePath


def get_2009_file(docID):
    """
    Constructs the filepath for a 2009 XML document based on the document identifier.

    Args:
        docID (str): The identifier of the 2009 document.

    Returns:
        str: The filepath of the specified 2009 document.
    """
    filePath = '/dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw/'
    filePath += docID[:7].lower() + '/'
    filePath += docID[8:-5] + '/'
    filePath += docID
    if get_year(docID) == '2006':
        filePath += '.LDC2007T07.sgm'
    else:
        filePath += '.LDC2009T13.sgm'
    return filePath


def get_doc_headers_AQUAINT(docID, docXML, tags):
    """
    Extracts headers from an AQUAINT document XML.

    Args:
        docID (str): The document identifier.
        docXML (str): The XML content of the document.
        tags (list of str): The tags to look for in the XML content.

    Returns:
        list of str: The extracted headers.
    """
    headers = []
    root = ET.fromstring(docXML)
    if root.find('DOCNO').text.strip() == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers


def get_doc_headers_AQUAINT2(docID, docXML, tags):
    """
    Extracts headers from an AQUAINT 2 document XML.

    Args:
        docID (str): The document identifier.
        docXML (str): The XML content of the document.
        tags (list of str): The tags to look for in the XML content.

    Returns:
        list of str: The extracted headers.
    """
    headers = []
    root = ET.fromstring(docXML)
    if root.get('id').strip() == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers


def get_doc_headers_2009(docID, docXML, tags):
    """
    Extracts headers from a 2009 document XML.

    Args:
        docID (str): The document identifier.
        docXML (str): The XML content of the document.
        tags (list of str): The tags to look for in the XML content.

    Returns:
        list of str: The extracted headers.
    """
    headers = []
    root = ET.fromstring(docXML)
    if root.find('DOCID').text.strip()[:21] == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers


def get_doc_AQUAINT(docID, filepath):
    """
    Retrieves the XML content of an AQUAINT document given its identifier and filepath.

    Args:
        docID (str): The document identifier.
        filepath (str): The filepath to the AQUAINT document.

    Returns:
        str: The XML content of the document.
    """
    doc = '<DOC>'
    with open(filepath) as F:
        inDoc = False
        for line in F:
            if line == '<DOCNO> ' + docID + ' </DOCNO>\n':
                inDoc = True
            if inDoc:
                doc += line
            if inDoc and line == '</DOC>\n':
                return doc


# Return xml with specified doc ID from AQUAINT 2 xml file
def get_doc_AQUAINT2(docID, filepath):
    doc = ''
    with open(filepath) as F:
        inDoc = False
        for line in F:
            if '<DOC id="' + docID + '"' in line:
                inDoc = True
            if inDoc:
                doc += line
            if inDoc and line == '</DOC>\n':
                return doc


def get_doc_2009(docID, filepath):
    """
    Retrieves the XML content of a 2009 document given its identifier and filepath.

    Args:
        docID (str): The document identifier.
        filepath (str): The filepath to the 2009 document.

    Returns:
        str: The XML content of the document.
    """
    doc = '<DOC>'
    with open(filepath) as F:
        inDoc = False
        for line in F:
            if '<DOCID> ' + docID in line:
                inDoc = True
            if inDoc:
                doc += line
            if inDoc and line == '</DOC>\n':
                return doc


def separate_paragraphs(docXML):
    """
    Separates text into paragraphs from the given document XML.

    Args:
        docXML (str): The XML content of the document.

    Returns:
        list of str: A list of paragraphs extracted from the document.
    """
    paragraphs = []
    text = docXML.split('<TEXT>')
    text = text[1].split('</TEXT>')[0]
    if '<P>' in docXML:
        paras = text.split('<P>')
        for p in paras:
            p = p.replace('</P>', '')
            p = p.replace('\n', ' ')
            p = p.replace('\t', ' ')
            p = p.strip()
            if len(p) > 0:
                paragraphs.append(p)
    else:
        paras = text.split('\n')
        for p in paras:
            p = p.replace('\n', ' ')
            p = p.replace('\t', ' ')
            p = p.strip()
            if len(p) > 0:
                paragraphs.append(p)
    return paragraphs


def tokenize_text(text):
    """
    Tokenizes the given text into sentences, and then tokenizes each sentence into words.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list of list of str: A list of sentences, where each sentence is a list of word tokens.
    """
    sentences = []
    for s in sent_tokenize(text):
        sentences.append(word_tokenize(s))
    return sentences