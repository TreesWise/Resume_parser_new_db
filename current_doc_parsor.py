import os
import json
import base64
import fitz
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

current_date = datetime.now().strftime("%d-%m-%Y")  # get today's date as dd-mm-yyyy

print("current_date:", current_date)
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

def convert_to_base64(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    images_b64 = []

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            images_b64.append(b64)
    elif ext in [".jpg", ".jpeg", ".png"]:
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            images_b64.append(b64)
    else:
        raise ValueError("Unsupported file format.")
    print("Image converted.............................................")
    return images_b64


def get_default_json():
    return {
        "docName": "null",
        "DocNumber": "null",
        "uploadedDate": current_date,
        "issuedCountry": "null",
        "IssuedPlace": "null",
        "issueDate": "null",
        "expDate": "null",
        "isNationalDoc": "null"
    }


def extract_json(images_b64):
    prompt_text = f"""You are an expert in document data extraction. Extract and translate into English only the following details in the following JSON format:

    JSON Format:
    {{
    "docName": "...",              
    "DocNumber": "...",            
    "uploadedDate": "{current_date}",
    "issuedCountry": "...",  
    "IssuedPlace": "...",   
    "issueDate": "dd-mm-yyyy",     
    "expDate": "dd-mm-yyyy",       
    "isNationalDoc": "Yes" or "No"   
    }}

    
    ### Instructions:
    1. Extract all **valid certificates, endorsements, medical documents, and training courses** from the document and return each as a separate JSON object.
    2.1. **For Visa documents** (i.e., if `docName` is "Visa" or contains the word "Visa"):
    - Extract only the **Visa Control Number** or **Visa Grant Number** as the `DocNumber`.
    - **Strictly ignore the Passport Number**, even if it appears on the same page.
    - The `DocNumber` must never be set to a **Passport Number** in Visa documents under any condition.
    - If **both numbers are found**, always choose the **Visa-specific number**, and discard the Passport Number.

    ### Special Rules for IssuedPlace:
    - **IssuedPlace** refers to the place of issue (city, port, or institution).
    - Do not use **place of birth, residence, or home city** as `IssuedPlace`.
    - **Never set `IssuedPlace` to `"null"`** if the country is known; use the country as a fallback if `IssuedPlace` is missing.
    - Set `IssuedPlace` to `"null"` only if both the place and country are missing or invalid.
    - **This rule applies to all documents**. Even if `IssuedPlace` is missing in any document, it must always fallback to the `IssuedCountry` if the country is available. Do **not apply this logic to just the first document**, but **ensure it is applied consistently to all documents in the list**.


    ### Special Rules for DocNumber:
    - **"DocNumber"**: Select the most relevant number based on priority:
    1. **For Visa documents** (if `docName` is "Visa" or contains the word "Visa"): 
        - Extract the **Visa Control Number** and use it as the `DocNumber`.
        - **Do not** extract the **Passport Number** for Visa documents, even if both numbers are present.
    2. **For Passport documents**: 
        - Extract the **Passport Number** and use it as the `DocNumber`.
    3. **Seaman's Book No.**.
    4. **Certificate No.** / **Doc Number**.
    5. **Strictly ignore any numbers without an explicit label indicating a document number** (e.g., "Passport Number", "Seaman's Book No.", "Certificate No.", "Doc Number", "Visa Control Number", "Visa Grant Number"). This includes numbers found in fields like "Application ID", "Serial No", "Control Number" or any unlabelled number. If no valid DocNumber with an explicit label is found, set "DocNumber" to "null".
    6.For e-Business Visa use Applictaion id as DocNumber.
    ### Special Rules for docName:
    - **if the docName  is visa then check which visa also.



    - **Do not use** "Serial No", "SL No", "Control Number", or "ID No".

    

    ### Field Rules:
    1. **"docName"** – Name of the certificate, endorsement, or training (e.g., "Certificate of Competency", "Advanced Fire Fighting", "Passport", "Visa").
    2. **"DocNumber"** – Use explicit labels like “Certificate No.”, “Doc Number”, “Passport Number” etc.
    3. **"uploadedDate"** – Use today’s date: **{current_date}**.
    4. **"issuedCountry"** – Country where the document was issued.
    5. **"IssuedPlace"** – City or port of issue. If only an institution name is listed (e.g., "Ministry of..."), set to **"null"**.
    6. **"issueDate"** and **"expDate"** – Format as **dd-mm-yyyy**.
    7. **"isNationalDoc"** – 
    - Set to **"Yes"** only if the document is a **Passport** (i.e., `"docName"` is `"Passport"`).
    - For **all other documents**, set it to **"No"**.

    8. Use **"null"** for any missing fields (like DocNumber, IssuedPlace, etc.).

    ### Extraction Guidelines:
    - Extract valid sections of the document:
    - Certificates, Endorsements, Health Certificates, Training Courses, Seaman’s Book, Passport.
    - Only extract sections with valid **document numbers** and clear details.
    - If multiple numbers or expiry dates exist, return them as separate objects.
    - Ignore irrelevant sections like revalidations, administrative stamps, or incomplete details.
    - If no valid information is found, return `null`.
    - Extract all valid certificates and courses, no prioritization.
    - **Translate content into English** where needed.

    ### Special Case: Course Lists or Tables
    - If training courses are listed in tables, extract each row as a separate object.
    - Set `"DocNumber"`, `"issueDate"`, `"expDate"`, `"IssuedPlace"` to `"null"` if not available.
    - Do not skip course entries even if embedded in other sections.

    ### Output Format:
    - Return a flat JSON array with no markdown, code blocks, or extra commentary.
    - Each object must include all fields listed in the JSON format.
    - Ensure the output is valid JSON.

    ### Examples:
    - Certificate of Competency (Master)
    - Endorsement (GMDSS Radio Operator)
    - Seafarer's Medical Certificate
    - Seaman’s Book
    - Passport
    - Familiarization and Basic Safety Training
    - Advanced Fire Fighting
    - Medical First Aid
    - ARPA
    - Radar Simulator
    - Security Awareness Training
    """

    prompt = [
        {
            "type": "text",
            "text": prompt_text
        },
        *[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}"
                }
            } for img in images_b64
        ]
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()

    return content


def postprocess_json(raw_json):
    if not isinstance(raw_json, str):
        return raw_json  # Nothing to do

    try:
        json_objects = raw_json.split('}\n{')
        if json_objects:
            json_objects[0] += '}'
            json_objects[-1] = '{' + json_objects[-1]
        results = [json.loads(obj) for obj in json_objects]
    except json.JSONDecodeError:
        return raw_json  # or return [] if you'd rather fail silently

    formatted = []
    for obj in results:
        known_fields = {
            "docName": obj.get("docName", "").strip(),
            "DocNumber": obj.get("DocNumber", "").strip(),
            "uploadedDate": obj.get("uploadedDate", "Not Available"),
            "issuedCountry": obj.get("issuedCountry", "").strip(),
            
            "IssuedPlace": obj.get("IssuedPlace", "").strip(),
            
            "issueDate": obj.get("issueDate", "").strip(),
            "expDate": obj.get("expDate", "Not Available").strip(),
            "isNationalDoc": obj.get("isNationalDoc", "No").strip(),
        }

        extra_fields = {
            k: v for k, v in obj.items()
            if k not in known_fields and k not in known_fields.keys()
        }

        if extra_fields:
            known_fields["metadata"] = extra_fields

        formatted.append(known_fields)

    return formatted





def process_document_to_json(file_path):
    images_b64 = convert_to_base64(file_path)
    raw_json = extract_json(images_b64)

    # If it's a string, try parsing
    if isinstance(raw_json, str):
        try:
            # Try to parse as proper JSON list
            raw_json = json.loads(raw_json)
        except json.JSONDecodeError:
            # If it's not valid JSON, try fallback string splitting logic
            return postprocess_json(raw_json)

    # If it's already a list, return it as-is
    if isinstance(raw_json, list)and raw_json:
        return raw_json

    # If it's something else, return safely
    return  [get_default_json()]
