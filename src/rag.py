import ollama
import os

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

path = "../documents"
document_list = loadFiles(path)

print(len(document_list))
question = input("Enter your question about the luzon straight")
response = answerQuestion(question, document_list[0]["text"] if document_list else "")
print(response)