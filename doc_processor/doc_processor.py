import xml.etree.ElementTree as ET


class DocumentProcessor:
    def __init__(self, input_xml_file, output_dir):
        self.input_xml_file = input_xml_file
        self.output_dir = output_dir

    # Return year from doc ID
    def get_year(self, docID):
        return int(docID[-13:])
    
    # Return filepath of AQUAINT xml document
    def get_AQUAINT_file(docID):
        filePath = '/corpora/LDC/LDC02T31/'
        filePath += docID[:2] + '/'
        filePath += get_year(docID) + '/'
        filePath += docID[-13:-6] + '_' + docID[:2] + '_ENG'
        return filePath

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
                if get_year(docID) >= 1996 and get_year(docID) <= 2000:
                    file = get_AQUAINT_file(docID)
                elif get_year(docID) >= 2004 and get_year(docID) <= 2006:
                    file = get_AQUAINT2_file(docID)
                elif get_year(docID) == 2009:
                    file = get_2009_file(docID)
                             
        return docsets