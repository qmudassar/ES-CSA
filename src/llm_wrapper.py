# Description: Module for complete LLM system prompt wrapper.

# Initializing Prompt Wrapper

class CoherePromptWrapper:
    def __init__(self, pipeline, tokenizer):
        self.pipeline = pipeline
        self.tokenizer = tokenizer

    def build_prompt(self, query, intent, user_info, docs, session_context=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and professional telecom customer service assistant.\n"
                    "You have been given a customer query and user-specific retrieved context relevant for answering that query.\n"
                    "Your task is to ANALYZE and REASON over the documents provided in the retrieved context to accurately respond to the customer query.\n"
                    "You MUST answer ONLY what the query is specifically asking for.\n"
                    "Limit your response to **MAXIMUM** 1 or 2 concise, clear sentences, unless the query specifically asks for multiple records.\n"
                    "ALWAYS speak directly to the customer. DO NOT refer to them in third person."
                    "You MUST respond professionally, politely and keep your responses to the point."
                    "You MUST utilize the provided conversation history to determine if the query is related to a previous one"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Customer Query: {query}\n"
                    f"Conversation History: {session_context}"
                    f"Identified Intent: {intent}\n"
                    f"User Profile: {user_info}\n"
                    f"Retrieved Context: {docs}"
                )
            }
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def __call__(self, query_bundle):
        return self.invoke(query_bundle)

    def invoke(self, query_bundle):
        query = query_bundle["query"]
        intent = query_bundle.get("intent", "")
        user_info = query_bundle.get("user_info", "")
        docs = query_bundle.get("docs", "")
        session_context = query_bundle.get("session_context", "")

        prompt = self.build_prompt(query, intent, user_info, docs, session_context)
        return self.pipeline(prompt)[0]["generated_text"]