from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    information=\
    '''Foundation models are large-scale AI models trained on vast, typically unlabeled, data using self-supervised learning, acting as a flexible base for diverse, specialized applications. Coined by Stanford researchers in 2021, these models (e.g., GPT, BERT) use transfer learning to perform tasks like text synthesis, image generation, and code generation. 

    Key Characteristics

    Massive Scale: Trained on enormous, broad datasets allowing them to learn general patterns, representations, and structures.
    Versatility: One model can be adapted, fine-tuned, or prompted for many downstream tasks (e.g., Q&A, translation, classification).
    Adaptation: Often fine-tuned on smaller, task-specific datasets to improve performance in niche domains.
    Common Architectures: Generally built on transformers, neural networks, and increasingly, multimodal data. 

    Examples and Use Cases
    Language Generation/Analysis: GPT series (OpenAI), BERT (Google).
    Image Generation: Stable Diffusion, Midjourney.
    Applications: Customer service, software development (code debugging), data analysis, and healthcare diagnostics. 

    Advantages and Challenges
    Advantages: Reduces the need for labeled data, increases accessibility for developers, and accelerates AI adoption across industries.
    Challenges: High computational costs for training, risks of inheriting societal biases, potential misinformation, and, in some cases, limited transparency. 
    Foundation models are considered the key technology driving the current wave of generative AI, acting as the foundation upon which more specific applications are built.'''

    summary_template = \
    '''Given the following information, summarize it in a concise manner: 
    focus on the key characteristics, examples, and advantages/challenges of foundation models.'''

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template)

    want = "Gemma3"
    want = "OpenAI"

    if want == "OpenAI":
        llm = ChatOpenAI(model="gpt-5", temperature=0.1)
    elif want == "Gemma3":
        llm = ChatOllama(model="gemma3:4b", temperature=0.1)  
    else:
        llm = None
        print(">>> please define a model")

    if llm is not None:
        chain = summary_prompt_template | llm

        response = chain.invoke(input={"information": information})

        print("Foundation Models Summary:\n")
        print(response.content)

if __name__ == "__main__":
    main()
