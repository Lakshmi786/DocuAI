import gradio as gr
import os
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma


def qa_system(pdf_files, openai_key, prompt, chain_type, k):
    os.environ["OPENAI_API_KEY"] = openai_key

    texts = []

    # load documents from PDF files
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        # split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts.extend(text_splitter.split_documents(documents))

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type=chain_type, retriever=retriever, return_source_documents=True)

    # get the result
    result = qa({"query": prompt})
    return result['result'], ''.join(doc.page_content for doc in result["source_documents"])


# define the Gradio interface
input_file = gr.File(file_count="multiple", label="PDF File")
openai_key = gr.Textbox(label="OpenAI API Key", type="password")
prompt = gr.Textbox(label="Question Prompt")
chain_type = gr.Radio(['stuff', 'map_reduce', "refine", "map_rerank"], label="Chain Type")
k = gr.Slider(minimum=1, maximum=5, label="Number of Relevant Chunks")

output_text = gr.Textbox(label="Answer")
output_docs = gr.Textbox(label="Relevant Source Text")

gr.Interface(qa_system, inputs=[input_file, openai_key, prompt, chain_type, k], outputs=[output_text, output_docs],
             title="DocuAI",
             description="Upload a PDF file, enter your OpenAI API key, type a question prompt, select a chain type, and choose the number of relevant chunks to use for the answer.").launch(
    debug=True)