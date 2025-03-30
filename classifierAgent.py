import yaml
import json
import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

# Set Azure OpenAI environment variables
os.environ["OPENAI_API_KEY"]


# Load rules and classifiers from YAML files
with open("utteranceRules.yaml", "r") as f:
    rules_data = yaml.safe_load(f)

with open("classifiers.yaml", "r") as f:
    classifiers_data = yaml.safe_load(f)

# Prepare embeddings and vector stores
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

rules_text = "".join([f"{key}: {value['description']}" for key, value in rules_data['coding_rules'].items()])
# vectorstore_rules = Chroma.from_texts(rules_texts, embeddings)

classifier_texts = []
for cls_name, details in classifiers_data["classification_rules"].items():
        classifier_texts.append(f"{cls_name} ({details['code']}): {details['definition']}")
        if 'sub_definitions' in details:
            for sub_name, sub_def in details['sub_definitions'].items():
                classifier_texts.append(f"{cls_name}.{sub_name}: {sub_def['definition']} Examples: {', '.join(sub_def['examples'])}")

# vectorstore_classifiers = Chroma.from_texts(classifier_texts, embeddings)

# Initialize LLM and Memory
llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
memory = ConversationBufferMemory(memory_key="history")

# Prompt templates
prompt = PromptTemplate(
    template=f"""
    You are classifying an utterance based on the following rules:

    Splitting Rules:
    {rules_text}

    Classifiers:
    {''.join(classifier_texts)}

    Conversation history:
    {{history}}

    Current utterance by {{speaker_role}}:
    \"{{utterance}}\"

    Decide clearly:
    1. How should this utterance be segmented according to the rules?
    2. Which classifier best matches each segment?

    Respond in JSON format:
    {{{{
      \"segments\": [
        {{{{
          \"segment\": \"text\",
          \"classifier_code\": \"code\",
          \"reasoning\": \"brief reasoning\"
        }}}}
      ]
    }}}}
""",
    input_variables=["history", "speaker_role", "utterance"]
)

chain = (prompt | llm).with_config(memory=memory)

def classify_utterance(utterance, speaker_role="parent"):
    response = chain.invoke({
        "history": "",
        "speaker_role": speaker_role,
        "utterance": utterance
    })
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        # Remove triple backticks and leading/trailing whitespace
        cleaned = content.strip("` \n")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"segments": str(content)}

# Function to extract speaker from timestamp column
def extract_speaker(timestamp):
    if "parent" in timestamp.lower():
        return "parent"
    elif "child" in timestamp.lower():
        return "child"
    else:
        return " ".join(timestamp.split()[:2])  # e.g., "Speaker 1"

# Function to process CSV with timestamp and transcript columns
def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    results = []

    for _, row in df.iterrows():
        utterance_col = next((col for col in row.index if col.lower() in ["transcript", "utterance"]), None)
        utterance = row[utterance_col] if utterance_col else ""
        timestamp_col = next((col for col in row.index if col.lower() == "timestamp"), None)
        timestamp = row[timestamp_col] if timestamp_col else ""
        speaker = extract_speaker(timestamp)
        result = classify_utterance(utterance, speaker_role=speaker)
        results.append({"timestamp": timestamp, "utterance": utterance, "speaker": speaker, "classification": result})

    results_df = pd.DataFrame(results)
    results_df.to_csv("classified_utterances.csv", index=False)
    print("Classification completed. Results saved to 'classified_utterances.csv'.")

# Example usage
if __name__ == "__main__":
    folder_path = "transcripts/"  # folder containing transcript CSVs
    all_results = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".csv"):
            full_path = os.path.join(folder_path, file_name)
            print(f"Processing {file_name}...")
            df = pd.read_csv(full_path)
            for _, row in df.iterrows():
                utterance_col = next((col for col in row.index if col.lower() in ["transcript", "utterance"]), None)
                utterance = row[utterance_col] if utterance_col else ""
                timestamp_col = next((col for col in row.index if col.lower() == "timestamp"), None)
                timestamp = row[timestamp_col] if timestamp_col else ""
                speaker = extract_speaker(timestamp)
                result = classify_utterance(utterance, speaker_role=speaker)
                all_results.append({"file": file_name, "timestamp": timestamp, "utterance": utterance, "speaker": speaker, "classification": result})

                results_df = pd.DataFrame(all_results)
            output_file = f"classified_{os.path.splitext(file_name)[0]}.csv"
            results_df.to_csv(output_file, index=False)
            print(f"Results saved to '{output_file}'")

    print("All files processed.")