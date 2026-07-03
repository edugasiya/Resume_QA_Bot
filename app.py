import os
from dotenv import load_dotenv

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
# Safely load the OpenAI API key from the local .env file
load_dotenv()

# ==========================================
# SECTION 1: Introduction to LLMs & Temperature
# ==========================================
from langchain_openai import ChatOpenAI

print("--- Testing Basic LLM (Temperature 0.7) ---")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
response = llm.invoke("Why do stars twinkle?")
print(response.content)
print("\n")

# ==========================================
# SECTION 2: LangChain - Basic Chatbot
# ==========================================
print("--- Testing LangChain Question ---")
llm_langchain = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
response_langchain = llm_langchain.invoke("What is LangChain and how is it used?")
print(response_langchain.content)
print("\n")

# ==========================================
# SECTION 3: Dynamic Prompts & Templates
# ==========================================
from langchain_core.prompts import ChatPromptTemplate

print("--- Testing f-strings ---")
y = 66
x = f"apple {y} is good"
print(x)
print("\n")

print("--- Testing Prompt Templates ---")
template = "Explain the concept of {concept} in simple terms in a minimum of {min} and maximum of {max} lines."
prompt = ChatPromptTemplate.from_template(template)
messages = prompt.format_messages(concept="Sky", min=5, max=10)

llm_template = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
response_template = llm_template.invoke(messages)
print(response_template.content)
print("\n")

# ==========================================
# SECTION 4: Chains with Single Arguments
# ==========================================
from langchain_core.prompts import PromptTemplate

print("--- Testing Single Argument Chains ---")
prompt_template = PromptTemplate(input_variables=["concept"],
                                 template="Explain the concept of {concept}")

llm_chain = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
chain_one = prompt_template | llm_chain
print(chain_one.invoke({"concept": "waterfall"}))
print("\n")

# Define a second prompt for the sequential chain
second_prompt = PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} and describe it to me as i am a 5 year old child",
)
chain_two = second_prompt | llm_chain

# ==========================================
# SECTION 5: Sequential Chains (LCEL)
# ==========================================
from langchain_core.globals import set_debug
set_debug(True)

print("--- Testing Sequential Chain ---")
overall_chain = chain_one | chain_two
result = overall_chain.invoke({"concept": "cars"})
print(result.content)
print("\n")

# ==========================================
# SECTION 6: Python Functions & Lambda
# ==========================================
print("--- Python Function & Lambda Tests ---")
def square(x):
  return x**2

k = square(5)
print(k)

ld = lambda x: x**2
print(ld(5))
print("\n")

# ==========================================
# SECTION 7: Chains with Multiple Arguments
# ==========================================
from langchain_core.output_parsers import StrOutputParser

print("--- Testing Complex Chains ---")
llm_complex = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

prompt1 = PromptTemplate(
    input_variables=["concept", "lang"],
    template="Explain the concept of {concept} in {lang}."
)
chain_one_complex = prompt1 | llm_complex | StrOutputParser()

prompt2 = PromptTemplate(
    input_variables=["explanation", "lang"],
    template=(
        "Summarize this explanation in a few lines as if I am a 5-year-old child "
        "in {lang}:\n{explanation}"
    )
)
chain_two_complex = prompt2 | llm_complex | StrOutputParser()

overall_chain_complex = (
    {"explanation": chain_one_complex, "lang": lambda x: x["lang"]}
    | chain_two_complex
)

result_complex = overall_chain_complex.invoke({
    "concept": "machine learning",
    "lang": "malayalam"
})
print("For children:", result_complex)
print("\n")

# ==========================================
# SECTION 8: LlamaIndex - Index and Query PDFs
# ==========================================
try:
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
    from llama_index.readers.file import PDFReader

    print("--- Testing LlamaIndex ---")
    reader = PDFReader()
    # Note: Ensure edwin_resume (2).pdf is in your folder if you want this specific block to run locally
    documents = reader.load_data(file='/content/edwin_resume (2).pdf') 

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    response_pdf = query_engine.query("Summarize this document in 1 line")
    print(response_pdf)
    print("\n")
except Exception as e:
    print(f"Skipping LlamaIndex local test (File missing or error): {e}\n")

# ==========================================
# SECTION 9: Project - Resume Q&A Bot
# ==========================================
import gradio as gr
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7)

def analyze_resume(pdf_file, question):
    if pdf_file is None:
        return "Please upload a PDF first."
    if not question or not question.strip():
        question = "Summarize this document."

    reader = PDFReader()
    documents = reader.load_data(file=pdf_file)

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    response = query_engine.query(question)
    return str(response)

def clear_all():
    return None, "", ""

custom_css = """
.gradio-container {background-color: #0f1117 !important;}
#analyze-btn {background: #6a4cff !important; color: white !important; font-weight: 600;}
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("## PDF Analyzer AI")
    gr.Markdown("Upload a PDF and ask any question about it.")

    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question", placeholder="e.g. summarize this document")

    with gr.Row():
        analyze_btn = gr.Button("Analyze", elem_id="analyze-btn")
        clear_btn = gr.Button("Clear")

    gr.Markdown("### AI Response")
    output_box = gr.Textbox(label="", lines=8, interactive=False)

    analyze_btn.click(fn=analyze_resume, inputs=[pdf_input, question_input], outputs=output_box)
    clear_btn.click(fn=clear_all, inputs=[], outputs=[pdf_input, question_input, output_box])

if __name__ == "__main__":
    iface.launch(share=True)
