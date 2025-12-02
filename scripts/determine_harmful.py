import time
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel

API_KEY = "AIzaSyC64vbuYXInWZK3YrG5JLsOVUE97A8iLDM"

class HarmfulJudgement(BaseModel):
    """Schema for structured, detailed notes."""
    category: str
    justification: str

gemini_2_5_flash_lite = {
    "name": "gemini-2.5-flash-lite",
    "RPM": 15,
    "TPM": 250000,
    "RPD": 1000,
    "input_token_limit": 1048576,
    "output_token_limit": 65536
}

gemini_2_0_flash_lite = {
    "name": "gemini-2.0-flash-lite",
    "RPM": 30,
    "TPM": 250000,
    "RPD": 1000,
    "input_token_limit": 1048576,
    "output_token_limit": 65536
}

HARMFUL_JUDGEMENT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "category": types.Schema(type=types.Type.STRING, description="One of: \"Not Harmful\", \"Slightly Harmful\", \"Harmful\", \"Very Harmful\"."),
        "justification": types.Schema(type=types.Type.STRING, description="A brief explanation for the category chosen.")
    },
    required=["category", "justification"]
)

def _get_genai_response(
    model: dict,
    contents,
    API_KEY: str,
    config: dict | None = None,
):
    """Initialize and return a GenAI client using the configured model."""
    print(f"-> Calling model: {model['name']}...")
    genai_client = genai.Client(api_key=API_KEY)

    try:
        response_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=HARMFUL_JUDGEMENT_SCHEMA,
            **(config or {}) # Merge any other config settings
        )
        
        response = genai_client.models.generate_content(
            model=model["name"],
            contents=contents,
            config=response_config,
        )

        # Rate limiting logic (wait based on RPM)
        if model.get("RPM", 0) > 0:
            # Sleep for just over 1 minute / RPM
            time.sleep(61 / model["RPM"])

        return response

    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

df_ethnicity = pd.read_csv("notebooks/data/df_ethnicity.csv").head(500)
df_ethnicity["category"] = ""
df_ethnicity["justification"] = ""


for idx, row in df_ethnicity.iterrows():
    article_text = row.get("text", "")
    
    prompt = f"""
        Please see an article below. Determine whether the way the article was written is harmful or contains harmful biases 
        or stereotypes related to ethnicity, sexual orientation, religion, or other sensitive attributes. 
        Please answer this question from the perspective of a reader of this article.
        Categorize the article in one of: \"Not Harmful\", \"Slightly Harmful\", \"Harmful\", \"Very Harmful\".\n\n
        Article:\n{article_text}\n\n
        Return the output as a JSON object strictly conforming to the provided schema.
        """

    response = _get_genai_response(
        model=gemini_2_0_flash_lite,
        contents=prompt,
        API_KEY=API_KEY,
    )

    if not response:
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = "API call failed"
        continue

    response_json_str = getattr(response, "text", None)
    
    if not response_json_str:
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = "No response text"
        continue

    try:
        # Use Pydantic's model_validate_json for direct validation and parsing
        response_data = HarmfulJudgement.model_validate_json(response_json_str) 
        
        df_ethnicity.loc[idx, "category"] = response_data.category
        df_ethnicity.loc[idx, "justification"] = response_data.justification
        
        print(f"Processed index {idx}: Category='{response_data.category}'")

    except Exception as e:
        print(f"Error parsing/validating response JSON for index {idx}: {e}\nRaw Response: {response_json_str}")
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = f"Failed to parse or validate response: {e}"


# Save the result
output_path = "notebooks/data/df_ethnicity_labeled.csv"
df_ethnicity.to_csv(output_path, index=False)
print(f"\nProcessing complete. Results saved to {output_path}")