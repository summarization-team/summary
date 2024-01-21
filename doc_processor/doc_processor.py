import os
import xml.etree.ElementTree as ET
from nltk.tokenize import sent_tokenize, word_tokenize


class DocumentProcessor:
    def __init__(self, input_xml_file, output_dir):
        self.input_xml_file = input_xml_file
        self.output_dir = output_dir
        
    def process_documents(self):
        # Implement XML parsing and document processing
        # Return a list of DocSets
        docsets = []
        tree = ET.parse(self.input_xml_file)
        root = tree.getroot()
        
        # Find all docsetA
        for docset in root.iter('docsetA'):
            docsetID = docset.get('id')
            docsets.append(docsetID)
            for doc in docset:
                docID = doc.get('id')
                path = self.output_dir + '/' + docsetID
                process_doc(path, docID)
                
        # Find all docsetB
        for docset in root.iter('docsetB'):
            docsetID = docset.get('id')
            docsets.append(docsetID)
            for doc in docset:
                docID = doc.get('id')
                path = self.output_dir + '/' + docsetID
                process_doc(path, docID)
                
        return docsets

# Process doc
def process_doc(dirPath, docID):
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
        file = get_AQUAINT2_file(docID)
        try:
            docXML = get_doc_AQUAINT2(docID, file)
        # if not found in AQUAINT 2 try 2009 corpus
        except:
            try:
                docXML = get_doc_2009(docID, file)
                for x in remove_these:
                    docXML = docXML.replace(x, '')
                headers = get_doc_headers_2009(docID, docXML, headerTags)
                newFile = os.path.join(dirPath, docID)
                with open(newFile, 'a') as F:
                    for h in headers:
                        F.write(h + '\n')
                    paragraphs = separate_paragraphs(docXML)
                    for p in paragraphs:
                        tokenized = tokenize_text(p)
                        F.write('\n')
                        for s in tokenized:
                            F.write(str(s) + '\n')
            except:
                return

        for x in remove_these:
            docXML = docXML.replace(x, '')
        headers = get_doc_headers_AQUAINT2(docID, docXML, headerTags)
        newFile = os.path.join(dirPath, docID)
        with open(newFile, 'a') as F:
            for h in headers:
                F.write(h + '\n')
            paragraphs = separate_paragraphs(docXML)
            for p in paragraphs:
                tokenized = tokenize_text(p)
                F.write('\n')
                for s in tokenized:
                    F.write(str(s) + '\n')
    
# Return year from doc ID
def get_year(docID):
    return docID[-13:-9]
    
# Return filepath of AQUAINT xml document
def get_AQUAINT_file(docID):
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
    
# Return filepath of AQUAINT 2 xml document
def get_AQUAINT2_file(docID):
    filePath = '/corpora/LDC/LDC08T25/data/'
    filePath += docID[:7].lower() + '/'
    if docID[:3] == 'XIE':
        filepath += docID[4:-5] + 'XIN_ENG'
    else:
        filePath += docID[:7].lower() + '_' + docID[-13:-7] + '.xml'
    return filePath

# Return filepath of 2009 xml document
def get_2009_file(docID):
    filePath = '/dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw/'
    filePath += docID[:7].lower() + '/'
    filePath += docID[8:-5] + '/'
    filePath += docID
    if get_year(docID) == '2006':
        filePath += '.LDC2007T07.sgm'
    else:
        filePath += '.LDC2009T13.sgm'
    return filePath

# Return headers to go in the top of the output file (DATE_TIME, CATEGORY, HEADLINE etc.)
def get_doc_headers_AQUAINT(docID, docXML, tags):
    headers = []
    root = ET.fromstring(docXML)
    if root.find('DOCNO').text.strip() == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers

# Return headers to go in the top of the output file (DATE_TIME, CATEGORY, HEADLINE etc.)
def get_doc_headers_AQUAINT2(docID, docXML, tags):
    headers = []
    root = ET.fromstring(docXML)
    if root.get('id').strip() == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers

# Return headers to go in the top of the output file (DATE_TIME, CATEGORY, HEADLINE etc.)
def get_doc_headers_2009(docID, docXML, tags):
    headers = []
    root = ET.fromstring(docXML)
    if root.find('DOCID').text.strip()[:21] == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers



# Return xml with specified doc ID from AQUAINT xml file
def get_doc_AQUAINT(docID, filepath):
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

# Return xml with specified doc ID from 2009 xml file
def get_doc_2009(docID, filepath):
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


            
# Split text into paragraphs
# Return list of paragraphs
def separate_paragraphs(docXML):
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

# Tokenize text
#Return list of tokenized sentences
def tokenize_text(text):
    sentences = []
    for s in sent_tokenize(text):
        sentences.append(word_tokenize(s))
    return sentences


            
            