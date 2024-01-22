from doc_processor import DocumentProcessor
import sys

doc_processor = DocumentProcessor(r'%s' %str(sys.argv[1]), r'%s' %str(sys.argv[2]))
docsets = doc_processor.process_documents()