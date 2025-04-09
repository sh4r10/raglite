import os
import configparser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load config
config_path = os.getenv(
    "RAGLITE_CONFIG", "/mnt/lts/projects/raglite/config.ini")
config = configparser.ConfigParser()
config.read(config_path)

llm = config["models"]["llm"]


def load_tinyllama():
    tokenizer = AutoTokenizer.from_pretrained(llm)

    model = AutoModelForCausalLM.from_pretrained(
        llm,
        low_cpu_mem_usage=True
    )

    pipe = pipeline("text-generation", model=model,
                    tokenizer=tokenizer, device="cpu")
    return pipe


pipe = load_tinyllama()


def query_llm(query):
    response = pipe(query, max_new_tokens=512, do_sample=True)
    return response[0]["generated_text"]
