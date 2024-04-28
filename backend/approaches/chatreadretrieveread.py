import os
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from approaches.approach import Approach
from text import nonewlines

AZURE_SEARCH_INDEX = os.getenv ("AZURE_SEARCH_INDEX")
AZURE_OPENAI_MAX_CONTENT = os.getenv ("AZURE_OPENAI_MAX_CONTENT") or 4500


# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadRetrieveReadApproach(Approach):
    system_prompt = """
    
    ## Task Goal
       The task goal is to generate an ANSWER for the QUESTION, based on the provided SOURCES.

    ## Task instructions
        You will be given a list of SOURCES that you can use to ANSWER the QUESTION. 
        You must use the SOURCES to ANSWER the QUESTION. 
        You must not use any other SOURCES. 
        Always include the SOURCE name for each fact in the response, referencing its full path with square brackets, e.g., [info1.txt]. 
        Do not combine SOURCES; list each source separately, e.g., [folder_a/info1.txt][info2.pdf].
        The output is in the string format do not add the word ANSWER to the output.
        The answer must be generated in {{$language}} language..

    ## Task Input:
        "HISTORY": "{history}"
        "QUESTION": "{ask}"
        "SOURCES": "{sources}"

    ## Task Output:
    """


    def __init__(self, search_client: SearchClient, chatgpt_deployment: str, gpt_deployment: str, sourcepage_field: str, content_field: str):
        self.search_client = search_client
        self.search_index = AZURE_SEARCH_INDEX
        self.chatgpt_deployment = chatgpt_deployment
        self.gpt_deployment = gpt_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.max_content = AZURE_OPENAI_MAX_CONTENT

    def run(self, history: list[dict], overrides: dict) -> any:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
        # Search -----------------------------------------
        ask = history[len(history)-1]["user"]
        r = self.search_client.search(ask, filter=filter, top=top)
        if use_semantic_captions:
            results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        else:
            results = [nonewlines(doc[self.content_field]) for doc in r]
        content = "\n".join(results)
        print (content)
        # Search -----------------------------------------
        
        # actualizar el historial de la conversación ------------
        messages = self.get_chat_history_as_messages(history)
        prompt  = [{"role": "system", 
                        "content": self.system_prompt.format(
                                    sources = content[:int(self.max_content)],
                                    ask = ask,
                                    history = messages)
                        }]

        # Answer ---------------------------- 
        completion = openai.ChatCompletion.create(
                engine=self.chatgpt_deployment,
                messages=prompt,
                temperature=overrides.get("temperature") or 0.7, 
                max_tokens=500,
            )
        response = completion["choices"][0]["message"]["content"]
        # Answer ---------------------------- 
        return {"data_points": results, 
                "answer": response,
                "thoughts": ""
                }


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
