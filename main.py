import dspy
import json
import asyncio
import dotenv
dotenv.load_dotenv()

class MemoryExtract(dspy.Signature):
    """Extracts relevent information from the conversation. Create memory entries that you should remember when speaking with the user later. you will provide  a list of info """
    transcript:str=dspy.InputField()
    information:list[str]=dspy.OutputField()


memeory_extractor=dspy.Predict(MemoryExtract)

async def extract_memories_from_messages(messages:list[dict[str,str]]):
    """
    Docstring for extract_memories_from_messages
    
    :param messages: Description
    :type messages: list[dict[str, str]]
    """
    transcript=json.dumps(messages)
    with dspy.context(lm=dspy.LM(model="openai/gpt-3.5-turbo")):
        out=await memeory_extractor.acall(transcript=transcript)
    
    print(out)



if __name__ == "__main__":
    message=[
        {
            "role":"user",
            "content":"Hello, how are you?"
        },
        {
            "role":"assistant",
            "content":"I am a large language model created by OpenAI, how can I help you today?"
        },
        {
            "role":"user",
            "content":"My name is Omji and like coffe"
        }
    ]
    asyncio.run(extract_memories_from_messages(message))
