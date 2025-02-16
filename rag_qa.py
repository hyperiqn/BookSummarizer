#!/usr/bin/env python
import transformers
import torch
from summarizer import load_model

_tokenizer, _pipe = load_model()

def format_docs(docs):
    """
    Formats a list of document strings for prompt insertion.
    """
    return "\n".join(f"<doc{i+1}>:\n{doc}\n</doc{i+1}>" for i, doc in enumerate(docs))

class ChatPromptTemplate:
    def __init__(self, messages):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        return cls(messages)

    def format(self, **kwargs):
        formatted_messages = []
        for role, message in self.messages:
            formatted_messages.append({"role": role, "content": message.format(**kwargs)})
        return formatted_messages

class LocalLLMWrapper:
    def __init__(self, temperature=0.5, max_new_tokens=150):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def __call__(self, prompt_messages):
        outputs = _pipe(
            prompt_messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            num_return_sequences=1,
            return_full_text=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
        return outputs[0]["generated_text"]

class StrOutputParser:
    def parse(self, text):
        return text.strip()

def check_each_doc_relevance(docs, question, max_tokens=50):
    """
    Checks the relevance of each individual document to the user's question.
    Returns a list of documents that are deemed relevant.
    """
    tokenizer, pipe = load_model()
    relevant_docs = []

    for doc in docs:
        relevance_prompt = [
            {"role": "system", "content": "You are an assistant that evaluates document relevance."},
            {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {question}\n\nIs this document relevant to answer the question? Answer with 'Yes' or 'No'."}
        ]
        outputs = pipe(
            relevance_prompt,
            max_new_tokens=max_tokens,
            temperature=0.0,
            do_sample=False,
            num_return_sequences=1,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = outputs[0]["generated_text"].strip().lower()
        if "yes" in response:
            relevant_docs.append(doc)
    
    return relevant_docs

def answer_question(vectorstore, question, max_tokens=150):
    """
    Answers the given question using a RAG-style chain.
    This version uses a vectorstore to retrieve relevant document chunks,
    evaluates each chunk individually for relevance, and then generates an answer.
    """
    retrieved_docs = vectorstore.similarity_search(question, k=4)
    docs = [doc.page_content for doc in retrieved_docs]

    relevant_docs = check_each_doc_relevance(docs, question)
    if not relevant_docs:
        return "The retrieved documents do not appear to be relevant to your question."

    system_message = (
        "You are an assistant for question-answering tasks. Answer the question based "
        "on your knowledge and the retrieved documents provided. \n"
        "Use five to ten sentences maximum and keep the answer concise. \n"
        "Ensure each sentence and point is unique, avoiding repetition. "
        "Give the final output in bullet points."
    )
    human_message_template = (
        "Retrieved documents: \n\n<docs>{documents}</docs>\n\n"
        "User question: <question>{question}</question>"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message_template)
    ])
    chain_input = {"documents": format_docs(relevant_docs), "question": question}
    final_prompt = prompt_template.format(**chain_input)
    local_llm = LocalLLMWrapper(temperature=0.5, max_new_tokens=max_tokens)
    response = local_llm(final_prompt)
    output_parser = StrOutputParser()
    generation = output_parser.parse(response)
    return generation
