import xml.etree.ElementTree as ET


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
                # AQUAINT
                if int(get_year(docID)) >= 1996 and int(get_year(docID)) <= 2000:
                    file = get_AQUAINT_file(docID)
                    docXML = get_doc(docID, file)
                    remove_these = ['&Cx1f', '&Cx13', '&Cx11', '&UR', '&LR', '&QL', '&HT', '&QC', '&AMP']
                    for x in remove_these:
                        docXML = docXML.replace(x, '')
                    headers = get_doc_headers(docID, docXML, ['DOCTYPE', 'DATE_TIME', 'CATEGORY', 'SLUG', 'HEADLINE'])
                    #docText = get_doc_test(docID)
                # AQUIAINT 2
                elif int(get_year(docID)) >= 2004 and int(get_year(docID)) <= 2006:
                    file = get_AQUAINT2_file(docID)
                # 2009 files
                elif int(get_year(docID)) == 2009:
                    file = get_2009_file(docID)
                             
        return docsets
    
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
    filePath = '~/dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw/'
    filePath += docID[:7].lower() + '/'
    filePath += docID[8:-5] + '/'
    filePath += docID + '.LDC2009T13.sgm'
    return filePath

# Return text to be tokenized

# Return headers to go in the top of the output file (DATE_TIME, CATEGORY, HEADLINE etc.)
def get_doc_headers(docID, docXML, tags):
    headers = []
    root = ET.fromstring(docXML)
    if root.find('DOCNO').text.strip() == docID:
        for t in tags:
            for i in root.iter(t):
                headers.append(t + ': ' + i.text.strip())
    return headers

# Return xml with specified doc ID from xml file
def get_doc(docID, filepath):
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
            
            
            