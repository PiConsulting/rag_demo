import openai
import os
import logging

# Simple retrieve-then-read implementation, using the Cognitive Search and OpenAI APIs directly. It first retrieves
# top documents from search, then constructs a prompt with them, and then uses OpenAI to generate an completion
# (answer) with that prompt. 

AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_SERVICE = os.environ.get("AZURE_OPENAI_PATH")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_TEMPERATURE = os.environ.get("AZURE_OPENAI_TEMPERATURE") or "0.0"
AZURE_OPENAI_TEMPERATURE = float(AZURE_OPENAI_TEMPERATURE)
AZURE_OPENAI_TOP_P = os.environ.get("AZURE_OPENAI_TOP_P") or "0.27"
AZURE_OPENAI_TOP_P = float(AZURE_OPENAI_TOP_P)
AZURE_OPENAI_RESP_MAX_TOKENS = os.environ.get("AZURE_OPENAI_MAX_TOKENS") or "1000"
AZURE_OPENAI_RESP_MAX_TOKENS = int(AZURE_OPENAI_RESP_MAX_TOKENS)
SYSTEM_MESSAGE_PATH = f"orc/prompts/no_search.prompt"


openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_SERVICE
openai.api_version = "2023-03-15-preview"
openai.api_key = AZURE_OPENAI_KEY


def get_answer(history):
        req, last_msg, prompt_prefix, user_q_formated = None, None, None, None
        prompt = {}
        prompt_prefix = "You are a AI assitant that helps people."
        #get user question
        messages = get_chat_history_as_messages(history, include_last_turn=True)
        prompt  = [{"role": "system", "content": prompt_prefix}]
        prompt.extend(messages)
        logging.info("[no search] PROMPT GENERADO")
        logging.info(f"[no search] {[prompt]}")
        completion = openai.ChatCompletion.create(
                engine=AZURE_OPENAI_DEPLOYMENT,
                messages=prompt,
                temperature=AZURE_OPENAI_TEMPERATURE,
                max_tokens=500,
                top_p=0.2,
                stop=["<|im_end|>", "<|im_start|>"],
            )
        response = completion["choices"][0]["message"]["content"]
        logging.info("[no search] RESPUESTA DEL CHAT")
        return {"data_points": "", "answer": response, "thoughts": ""}


def get_chat_history_as_messages(history, include_previous_questions=True, include_last_turn=True, approx_max_tokens=1000):
        history_list = []
        if len(history) == 0:
            return history_list
        for h in reversed(history if include_last_turn else history[:-1]):
            key = next(iter(h))
            try:
                history_item = {"role": "assistant", "content": h["bot"]}
                history_list.insert(0, history_item)
            except:
                pass
            key = next(iter(h))
            print('key: {}'.format(key))
            try:
                history_item = {"role": "user", "content": h["user"]}                
                history_list.insert(0, history_item)
            except:
                pass
        return history_list
