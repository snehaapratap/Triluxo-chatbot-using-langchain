from langchain.document_loaders import WebBaseLoader

def load_data_from_url(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

if __name__ == "__main__":
    url = "https://brainlox.com/courses/category/technical"
    documents = load_data_from_url(url)
    print(f"Loaded {len(documents)} documents")
