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
                if int(get_year(docID)) >= 1996 and int(get_year(docID)) <= 2000:
                    file = get_AQUAINT_file(docID)
                elif int(get_year(docID)) >= 2004 and int(get_year(docID)) <= 2006:
                    file = get_AQUAINT2_file(docID)
                elif int(get_year(docID)) == 2009:
                    file = get_2009_file(docID)
                             
        return docsets
    
# Return year from doc ID
def get_year(docID):
    return docID[-13:-9]
    
# Return filepath of AQUAINT xml document
def get_AQUAINT_file(docID):
    filePath = '/corpora/LDC/LDC02T31/'
    filePath += docID[:3] + '/'
    filePath += get_year(docID) + '/'
    filePath += docID[-13:-5] + '_' + docID[:3] + '_ENG'
    return filePath
    
# Return filepath of AQUAINT 2 xml document
def get_AQUAINT2_file(docID):
    filePath = '/corpora/LDC/LDC08T25/data/'
    filePath += docID[:7].lower() + '/'
    filePath += docID[:7].lower() + '_' + docID[-13:-7] + '.xml'
    return filePath

# Return filepath of 2009 xml document
def get_2009_file(docID):
    filePath = '~/dropbox/23-24/575x/TAC_2010_KBP_Source_Data/data/2009/nw'
    # TODO figure out where any 2009 documents are and add logic
    return filePath