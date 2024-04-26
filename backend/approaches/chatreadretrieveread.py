import os
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

AZURE_SEARCH_INDEX = os.getenv ("AZURE_SEARCH_INDEX")
CHAT_GPT_MAX_CONTENT = os.getenv ("CHAT_GPT_MAX_CONTENT")
# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    pi_system_prompt = """
    
    ## Task Goal
        The task goal is to generate an answer about a specific context based on the message HISTORY, user QUESTION and the provided SOURCES.

    ## Task instructions
        You are an assistant helps company employees with their questions on various topics such as vacations, benefits, company programs, company policies, and others. 
        The name of the company is Pi Data and the employees are recognized like Pi Member.
        Be brief in your answers. 
        Please respond ONLY with the data listed in the list of sources below. 
        If there is not enough information below, say you don't know. Please do not generate answers that do not use the sources below. If asking a clarifying question to the user would help, ask the question.
        To get tabular information, return it as an html table. Do not return markdown format.
        Generate three very brief follow-up questions that the user would likely ask next about their healthcare plan and employee handbook. 
        Try not to repeat questions that have already been asked.
        Below is a history of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base about employee healthcare plans and the employee handbook.
        Do not include cited source filenames and document names e.g info.txt or doc.pdf in the search query terms.
        Do not include any text inside [] or <<>> in the search query terms.
        If the question is not in English, translate the question to English before generating the search query.

    ## Task Input:
        "HISTORY": "{history}"
        "QUESTION": "{ask}"
        "SOURCES": "{sources}"

    ## Task Output:
    """

    slovenia_system_prompt = """

    ## Task Goal
        The task goal is to generate an answer about a specific context based on the message HISTORY, user QUESTION and the provided SOURCES.

    ## Task instructions
        You are an assistant helps customers about services from a FAB company. 
        The meaning of FAB is Fabulous Guided Adventures
        The main service are body rafting and canyoning.
        You will be given a list of SOURCES that you can use to ANSWER the QUESTION. 
        You must use the SOURCES to ANSWER the QUESTION. 
        You must not use any other SOURCES. 
        ALWAYS generate the ANSWER in the same language(ex: Spanish, English, Portuguese) of the QUESTION. 
        When answering, please use the format detailed below:
            - Use simple words, avoid jargon and **highlighting** keywords in bold. 
            - Responses should be concise, not exceeding 60 words. 
            - YOU MUST USE EMOJI to enumerate different options of courses.
            - Speak in FIRST PERSON SINGULAR.

    ## Task Input:
        "HISTORY": "{history}"
        "QUESTION": "{ask}"
        "SOURCES": "{sources}"

    ## Task Output:
    """
    
    
    query_prompt_template = """

        Chat History:
        {chat_history}

        Question:
        {question}

        Search query:
    """

    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.search_index = AZURE_SEARCH_INDEX
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.max_content = CHAT_GPT_MAX_CONTENT

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        # Generate an optimized keyword search query based on the chat history and the last question
        """prompt = self.query_prompt_template.format(chat_history=self.get_chat_history_as_text(history, include_last_turn=False), question=history[-1]["user"])
        completion = openai.Completion.create(
            engine=self.gpt_deployment, 
            prompt=prompt, 
            temperature=0.7, 
            max_tokens=32, 
            n=1, 
            stop=["\n"])
        q = completion.choices[0].text"""
        
        # Search -----------------------------------------
        ask = history[len(history)-1]["user"]
        r = self.search_client.search(ask, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)
        # Search -----------------------------------------
        
        # actualizar el historial de la conversación ------------
        messages = self.get_chat_history_as_messages(history)
        if "slovenia" in self.search_index:
            prompt  = [{"role": "system", 
                        "content": self.slovenia_system_prompt.format(
                                    sources = content[:int(self.max_content)],
                                    ask = ask,
                                    history = messages)
                        }]
        elif "pi-search" in self.search_index:
            prompt  = [{"role": "system", 
                        "content": self.pi_system_prompt.format(
                                    sources = content[:int(self.max_content)],
                                    ask = ask,
                                    history = messages)
                        }]
        else:
            prompt  = [{"role": "system", 
                        "content": self.pi_system_prompt.format(
                                    sources = content[:int(self.max_content)],
                                    ask = ask,
                                    history = messages)
                        }]
        # actualizar el historial de la conversación ------------

        # ChatGPT ---------------------------- 
        completion = openai.ChatCompletion.create(
                engine=self.chatgpt_deployment,
                messages=prompt,
                temperature=overrides.get("temperature") or 0.7, 
                max_tokens=500,
            )
        response = completion["choices"][0]["message"]["content"]
        # ChatGPT ---------------------------- 
        
        return {"data_points": results, 
                "answer": response,
                #"thoughts": f"Searched for:<br>{ask}<br><br>Prompt:<br>" + prompt.replace('\n', '<br>')}
                "thoughts": ""
                }


    def get_chat_history_as_text(self, history, include_last_turn=True, approx_max_tokens=1000) -> str:
        history_text = ""
        for h in reversed(history if include_last_turn else history[:-1]):
            history_text = """<|im_start|>user""" +"\n" + h["user"] + "\n" + """<|im_end|>""" + "\n" + """<|im_start|>assistant""" + "\n" + (h.get("bot") + """<|im_end|>""" if h.get("bot") else "") + "\n" + history_text
            if len(history_text) > approx_max_tokens*4:
                break    
        return history_text


    def get_chat_history_as_messages(self, history):
        history_list = []
        if len(history) == 0:
            return history_list
        for h in reversed(history):
            key = next(iter(h))
            try:
                history_item = {"role": "assistant", "content": h["bot"]}
                history_list.insert(0, history_item)
            except:
                pass
            key = next(iter(h))
            try:
                history_item = {"role": "user", "content": h["user"]}                
                history_list.insert(0, history_item)
            except:
                pass
        return history_list
