import os
import time
import re
import json
from google.genai.errors import APIError
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel

API_KEY = os.environ.get("GOOGLE_GENAI_API_KEY", "")

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

gemini_2_5_flash_lite_preview = {
    "name": "gemini-2.5-flash-lite-preview",
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

gemini_2_0_flash = {
    "name": "gemini-2.0-flash",
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

# ... (HarmfulJudgement, model definitions, EmptyGenAIResponse class remain the same) ...

# Assuming API_KEY is defined globally

class EmptyGenAIResponse:
    """
    Dummy klasse om een GenerateContentResponse-object na te bootsen bij fouten.
    Dit voorkomt een AttributeError wanneer de aanroepende code probeert .text te gebruiken.
    """
    @property
    def text(self):
        """Retourneert een lege string voor de .text-eigenschap."""
        return ""

    @property
    def candidates(self):
        return []

def _get_genai_response(
    model: dict,
    contents: str | types.Content,
    config: dict | None = None,
    file_upload: str | None = None,
):
    """Initialize and return a GenAI client using the configured model, with retries."""
    genai_client = genai.Client(api_key=API_KEY)

    if file_upload:
        uploaded_file = genai_client.files.upload(file=file_upload)
        contents = [contents, uploaded_file]

    # --- Retry configuration ---
    max_retries = 3
    base_wait_seconds = 5
    # ---------------------------

    call_config = config.copy() if config else {}
    if "response_schema" in call_config and issubclass(call_config["response_schema"], BaseModel):
        call_config["response_mime_type"] = "application/json"
    
    response = None 

    for attempt in range(max_retries):
        try:
            # --- API Call Logic ---
            response = genai_client.models.generate_content(
                model=model["name"],
                contents=contents,
                config=call_config,
            )
            
            # If successful, exit the retry loop
            break

        except APIError as e:
            error_message = str(e)
            
            # Check for the specific 503 UNAVAILABLE (Overloaded) error
            if "503 UNAVAILABLE" in error_message or "The model is overloaded" in error_message:
                if attempt < max_retries - 1:
                    # Handle 503 with exponential backoff
                    wait_time = base_wait_seconds * (2 ** attempt)
                    print(f"Model overloaded (503). Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"Model overloaded (503). Max retries ({max_retries}) reached. Returning empty.")
                    return EmptyGenAIResponse()
                
            # --- NEW: Handle 429 RESOURCE_EXHAUSTED (Quota) Error ---
            elif "429 RESOURCE_EXHAUSTED" in error_message:
                # Attempt to extract the required retry delay from the error message
                match = re.search(r'Please retry in ([\d\.]+)s', error_message)
                retry_time = float(match.group(1)) if match else 60.0 # Default to 60s if not found

                # The 429 error means a quota limit (like per day) was hit. 
                # Retrying immediately after a short delay is unlikely to fix a daily limit.
                # We log the error and stop the script gracefully, returning an error signal.
                
                print("\n" + "="*80)
                print(f"🚨 RESOURCE EXHAUSTED (429): Daily or minute quota hit for model {model['name']}.")
                print(f"   Stopping execution. The API suggested a retry delay of {retry_time:.2f} seconds.")
                print(f"   You may need to wait until the next day or check your billing plan.")
                print("="*80 + "\n")
                
                # Immediately return an error response that is distinguishable
                return EmptyGenAIResponse()

            else:
                # Re-raise any other, unexpected API error immediately
                raise

    # --- Final Rate Limit Sleep (Existing Logic) ---
    time.sleep(61 / model["RPM"])
    
    return response if response is not None else EmptyGenAIResponse()

# Note: You still need to place this entire function definition 
# at the top of your script along with the other imports and classes.

def _extract_json_from_text(text: str) -> str | None:
    """Try to extract a JSON object string from `text`.

    This removes markdown code fences (``` or ```json) and returns the substring
    from the first '{' to the last '}'. Returns None if no braces found.
    """
    if not isinstance(text, str):
        return None

    # Remove common markdown code fences
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Trim whitespace
    text = text.strip()

    # If the text already looks like JSON, return it
    if text.startswith('{') and text.endswith('}'):
        return text

    # Otherwise, try to find the first {...} block
    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
        return candidate

    return None

# Removed the unnecessary HARMFUL_JUDGEMENT_SCHEMA types.Schema definition

# Load the data and limit the size
df_ethnicity = pd.read_csv("notebooks/data/df_ethnicity.csv").head(190)
df_ethnicity = df_ethnicity.sample(frac=1).reset_index(drop=True)
df_ethnicity["category"] = ""
df_ethnicity["justification"] = ""


for idx, row in df_ethnicity.iterrows():
    article_text = row.get("text", "")
    
    prompt = f"""
        Please see an article below. Determine whether the way the article was written is harmful or contains harmful biases 
        or stereotypes related to ethnicity, sexual orientation, religion, or other sensitive attributes. 
        Please answer this question from the perspective of a reader of this article.
        Categorize the article in one of: "Not Harmful", "Slightly Harmful", "Harmful", "Very Harmful".\n\n
        Article:\n{article_text}\n\n
        Return the output as a JSON object strictly conforming to the provided schema.
        """

    # --- Refactored API Call with Response Schema ---
    response = _get_genai_response(
        model=gemini_2_0_flash,
        contents=prompt,
        config={
            "response_schema": HarmfulJudgement,  # Pass the Pydantic model here
            "temperature": 0.2,                   # Use a low temp for stable output
        }
    )

    if not response:
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = "API call failed"
        continue

    # The SDK handles JSON parsing and validation internally when 'response_schema' is used.
    # The validated data can be retrieved as a string from response.text, which is guaranteed 
    # to be JSON that matches the schema structure.
    response_json_str = getattr(response, "text", None)
    
    if not response_json_str:
        # Check if the model failed to generate any text
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = "Model failed to generate response text."
        continue

    try:
        # 1. Use Pydantic's model_validate_json to parse the guaranteed-to-be-correct JSON string
        response_data = HarmfulJudgement.model_validate_json(response_json_str) 
        
        # 2. Assign the validated fields directly
        df_ethnicity.loc[idx, "category"] = response_data.category
        df_ethnicity.loc[idx, "justification"] = response_data.justification
        
        print(f"Processed index {idx}: Category='{response_data.category}'")

    except Exception as e:
        # This fallback is now only for true JSON errors (like truncation/API issues)
        # and not for internal schema mismatch
        print(f"Error parsing/validating response JSON for index {idx}: {e}\nRaw Response: {response_json_str}")
        df_ethnicity.loc[idx, "category"] = "Error"
        df_ethnicity.loc[idx, "justification"] = f"Failed to parse validated JSON: {e}"


# Save the result
output_path = "notebooks/data/df_ethnicity_labeled.csv"
df_ethnicity.to_csv(output_path, index=False)
print(f"\nProcessing complete. Results saved to {output_path}")