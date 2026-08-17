from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
loaders = DirectoryLoader(
    path = "./papers",
    glob = "**/*.pdf",
    loader_cls = UnstructuredFileLoader,
    show_progress = True,
    use_multithreading = True 
)

docs = loaders.load()

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap = 200,
    add_start_index = True,
    strip_whitespace = True,
    separators = MARKDOWN_SEPARATORS
)

splits = text_splitter.split_documents(docs)
from pprint import pprint

pprint(splits) 