INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
CONTENT: {question}

FILENAME:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gemini-2.5-flash-lite'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query):
        boost_dict = {'content': 3.0}

        return self.index.search(
            query,
            boost_dict=boost_dict,
            num_results=5
        )

    def build_context(self, search_results):
        lines = []
        number = 0;

        for doc in search_results:
            number += 1
            lines.append(f"Lesson {number}")
            lines.append('Content: ' + doc['content'])
            lines.append('File Name: ' + doc['filename'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages
        )
        
        if response.usage:
            print(f"\n--- TOKEN USAGE ---")
            print(f"Input tokens: {response.usage}")
            print(f"Output tokens: {response.usage.completion_tokens}")
            print(f"-------------------\n")
        else:
            print("\n[Warning: No usage data returned by API]\n")

        return response.choices[0].message.content

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
