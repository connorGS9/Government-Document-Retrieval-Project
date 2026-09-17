import ollama
import os
import qdrant_client as qdrant
from qdrant_client.models import PointStruct, VectorParams, Distance
def loadFiles(path)-> list: # type: ignore
    doc_list = []
    if not os.listdir(path):
        print("No files found in the directory.")
        return doc_list

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path):
            cur_dict = {}
            with open(file_path, 'r') as file:
                cur_dict["source"] = filename
                cur_dict["text"] = file.read()
                doc_list.append(cur_dict)
    return doc_list


def answerQuestion(question, doc_data):
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {"role": "system", "content": "You are an assistant that answers qusetions based on given documents. Only respond if you know"},
            {"role": "user", "content": f"Answer the following question based on the document data provided. If the answer is not present in the document data, respond with 'I don't know'.\n\nDocument Data:\n{doc_data}\n\nQuestion:\n{question}"}
        ]
    )
    return response["message"]["content"]


def createEmbeddings(chunked_docs):
    embeddings_list = []
    for i in range(len(chunked_docs)):
        chunk = chunked_docs[i]
        embedding = ollama.embeddings(
            model="nomic-embed-text",
            prompt=chunk["chunk"]
        )
        _p = PointStruct(
            id=i,
            vector=embedding["embedding"],
            payload={"source": chunk["source"], "chunk": chunk["chunk"]}
        )
        embeddings_list.append(_p)
    return embeddings_list

path = "../documents"
document_list = loadFiles(path)


def chunkDocs(doc_list):
    chunkedDocs = []
    for doc in doc_list:
        source = doc["source"]
        split_text = doc["text"].splitlines() # list of words split by \n
        for i in range(len(split_text)):
            with_name = {}
            with_name["source"] = source
            with_name["chunk"] = split_text[i]
            chunkedDocs.append(with_name)

    return chunkedDocs

        



print(len(document_list))
chunked = chunkDocs(document_list)
embeddings_list = createEmbeddings(chunked)
print(len(embeddings_list))

qdrant_client = qdrant.QdrantClient(host="localhost", port=6333)
qdrant_client.create_collection(
    collection_name="document_collection",
    vectors_config=VectorParams(size=len(embeddings_list[0].vector), distance=Distance.COSINE)
)

for point in embeddings_list:
    qdrant_client.upsert(
        collection_name="document_collection",
        points=[point]
    )

print(qdrant_client.get_collection(collection_name="document_collection"))