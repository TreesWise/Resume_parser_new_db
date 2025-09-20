from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.storage.blob import BlobServiceClient
import json
import os
from datetime import datetime
import json
import tempfile
from fastapi import HTTPException
from docx2pdf import convert
import platform
import subprocess
import asyncio
import logging
import sys

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# with open("output_format.json", "r", encoding="utf-8") as f:
#     expected_structure = json.load(f)

# Configure logging globally for both files
logging.basicConfig(
level=logging.INFO, # Change to DEBUG for detailed logs
format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
handlers=[
logging.StreamHandler(sys.stdout)
]
)
logger = logging.getLogger("resume_parser")

def extract_date_fields(structured_json):
    date_fields = {}

    # Extract Dob
    basic_details = structured_json["data"].get("basic_details", [])
    if len(basic_details) > 1:
        for key, value in basic_details[0].items():
            if value == "Dob":
                date_fields[f"basic_details_{value}"] = basic_details[1].get(key)
    
    # Extract dates from experience_table
    if "experience_table" in structured_json["data"]:
        experience_table = structured_json["data"]["experience_table"]
        for key, value in experience_table[0].items():
            if value in ["FromDt", "ToDt"]:
                for i, row in enumerate(experience_table[1:]):
                    date_fields[f"experience_table_{i}_{value}"] = row.get(key)
    
    # Extract dates from certificate_table
    if "certificate_table" in structured_json["data"]:
        certificate_table = structured_json["data"]["certificate_table"]
        for key, value in certificate_table[0].items():
            if value in ["DateOfIssue", "DateOfExpiry"]:
                for i, row in enumerate(certificate_table[1:]):
                    date_fields[f"certificate_table_{i}_{value}"] = row.get(key)

    return date_fields


def update_date_fields(structured_json, corrected_dates):
    """
    Updates the structured JSON with corrected date values using the mapping.
    
    :param structured_json: The original structured JSON.
    :param corrected_dates: Dictionary containing the corrected date values.
    :return: Updated structured JSON with corrected dates if valid JSON, else raises an error.
    """
    # Update Dob in basic_details
    if f"basic_details_Dob" in corrected_dates:
        basic_details = structured_json["data"].get("basic_details", [])
        if len(basic_details) > 1:
            for key, value in basic_details[0].items():
                if value == "Dob":
                    basic_details[1][key] = corrected_dates[f"basic_details_Dob"]

    # Update experience_table dates
    if "experience_table" in structured_json["data"]:
        experience_table = structured_json["data"]["experience_table"]
        for key, value in experience_table[0].items():
            if value in ["FromDt", "ToDt"]:
                for i, row in enumerate(experience_table[1:]):
                    date_key = f"experience_table_{i}_{value}"
                    if date_key in corrected_dates:
                        row[key] = corrected_dates[date_key]

    # Update certificate_table dates
    if "certificate_table" in structured_json["data"]:
        certificate_table = structured_json["data"]["certificate_table"]
        for key, value in certificate_table[0].items():
            if value in ["DateOfIssue", "DateOfExpiry"]:
                for i, row in enumerate(certificate_table[1:]):
                    date_key = f"certificate_table_{i}_{value}"
                    if date_key in corrected_dates:
                        row[key] = corrected_dates[date_key]

    # Validate if the updated structured JSON is still valid JSON
    try:
        json.dumps(structured_json)  # This will raise an error if it's not valid JSON
        return structured_json
    except (TypeError, ValueError) as e:
        raise ValueError(f"Updated JSON is invalid: {e}")

def transform_extracted_info(extracted_info):
    structured_json = {"status": "success", "data": {}, "utc_time_stamp": datetime.utcnow().strftime("%d/%m/%Y, %H:%M:%S")}

    # Basic Details
    fields = extracted_info.get("fields", {})
    basic_details_keys = [
        "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", "Doa", "Dob", 
        "Address1", "Address2", "Address3", "Address4", "City", "State", "Country", "ZipCode", 
        "EmailId", "MobileNo", "AlternateNo", "Rank"
    ]
    
    # Normalize extracted field keys to lowercase for case-insensitive matching
    normalized_fields = {key.lower(): value for key, value in fields.items()}
    
    # Map fields using case-insensitive lookup while keeping original key names
    basic_details_values = [normalized_fields.get(key.lower(), None) for key in basic_details_keys]
    structured_json["data"]["basic_details"] = [
    {str(i): key for i, key in enumerate(basic_details_keys)},
    {str(i): value if value is not None else None for i, value in enumerate(basic_details_values)}]

    # Tables (Experience & Certificate)
    for table in extracted_info.get("tables", []):
        table_name = table["table_name"]
        headers = {str(i): col for i, col in enumerate(table["columns"])}
        rows = [{str(i): (value if value is not None else None) for i, value in enumerate(row)} for row in table["rows"]]
        structured_json["data"][table_name] = [headers] + rows

    return structured_json



def upload_to_blob_storage(file_path, container_name, connection_string):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=os.path.basename(file_path))
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
            
        logger.info("File %s successfully uploaded to %s.", file_path, container_name)
    except Exception as e:
        logger.error("Error uploading file to Blob Storage: %s", e, exc_info=True)
    #     print(f"File {file_path} successfully uploaded to {container_name}.")
    # except Exception as e:
    #     print(f"Error uploading file to Blob Storage: {e}")

def validate_parsed_resume(extracted_info, file_path, confidence_threshold=0.8, container_name=None, connection_string=None):
    errors = []
    
    # Check confidence score
    # print("confidence score ----------------------------------------------", extracted_info.get("confidence", 1))    
    logger.info("Confidence score----------------------------------------------: %s", extracted_info.get("confidence", 1))
    if extracted_info.get("confidence", 1) < confidence_threshold:
        errors.append("Low confidence score")
        
        # Upload file for retraining
        if container_name and connection_string:
            upload_to_blob_storage(file_path, container_name, connection_string)
    
    return errors if errors else ["Resume parsed successfully."]

def extract_resume_info(endpoint, key, model_id, path_to_sample_documents):
    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    with open(path_to_sample_documents, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(model_id=model_id, body=f)
    result: AnalyzeResult = poller.result()

    extracted_info = {}
    tables = []
    
    if result.documents:
        for idx, document in enumerate(result.documents):
            extracted_info["doc_type"] = document.doc_type
            extracted_info["confidence"] = document.confidence
            extracted_info["model_id"] = result.model_id
            
            if document.fields:
                extracted_info["fields"] = {}
                for name, field in document.fields.items():
                    field_value = field.get("valueString") if field.get("valueString") else field.content
                    extracted_info["fields"][name] = field_value
            
            # Extract table information
            for field_name, field_value in document.fields.items():
                if field_value.type == "array" and field_value.value_array:
                    col_names = []
                    sample_obj = field_value.value_array[0]
                    if "valueObject" in sample_obj:
                        col_names = list(sample_obj["valueObject"].keys())
                    
                    table_rows = []
                    for obj in field_value.value_array:
                        if "valueObject" in obj:
                            value_obj = obj["valueObject"]
                            row_data = [value_obj[col].get("content", None) for col in col_names]
                            table_rows.append(row_data)
                    
                    tables.append({"table_name": field_name, "columns": col_names, "rows": table_rows})
    
    extracted_info["tables"] = tables
    return extracted_info

def basic_openai(basic_table):

    prompt = f"""
    You are given a Python dictionary representing extracted personal information from a resume. Transform it into a JSON object with a key "basic_details", whose value is a list of two dictionaries:

    1. The first dictionary maps indexes (as string keys) from "0" to "19" to specific field names in this exact order:
    ["Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender", "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", "City", "State", "Country", "ZipCode", "EmailId", "MobileNo", "AlternateNo", "Rank"]

    2. The second dictionary maps the same indexes to the actual values found in the input dictionary (or `null` if not present or value is None).

    Ensure:
    - The output is valid JSON.
    - All keys in the first and second dictionaries are stringified numbers ("0", "1", ..., "19").
    - The order of the fields is preserved exactly as described.
    - Use `null` in JSON for missing or None values, not the Python `None`.

    3. Convert all dates to DD-MM-YYYY format
    4. Replace country names in Nationality with their demonym (e.g., Russia → Russian)
    5. Normalize gender values (male → Male, female → Female)
    # The above code is outlining a set of normalization rules for the "Country" field in a dataset.
    # It specifies steps to clean and standardize the country names by removing leading labels,
    # splitting on delimiters, identifying known country names, and handling cases where multiple
    # parts are present. Additionally, it provides instructions for populating the "City" field based
    # on the normalized country values. The goal is to ensure consistent and formatted country and
    # city data in the dataset.
    6. Normalize the "Country" field:
    - Remove any leading labels or descriptors (e.g., words like "country", "residency", "city", etc.).
    - If the remaining value contains multiple parts separated by delimiters (any of `/ | , ; - : >`), split on the first delimiter.
    - Take the **country** as:
        a) the part that matches a known country name (case-insensitive), if present; otherwise
        b) the first non-empty part after trimming.
    - Trim extra spaces, collapse repeated whitespace, and title-case names (preserve accents and standard abbreviations like USA, UK, UAE).
    7. Populate the "City" field:
        - If, after applying rule 6, there is a second non-empty part from the split, assign it to **City** after trimming and normalizing spacing (title-case).
        - If no second part exists, set City to `null`.
        - If the detected country and city are identical, keep the value as Country and set City to `null`.

    Here is the input dictionary:
    {basic_table}

    Return the final JSON as a list of dictionaries under the key "experience_table".
    """

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data formatter."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    res_json = json.loads(response.choices[0].message.content)
    return res_json



def certificate_openai(certificate_table):
    prompt = f"""
You are given a dictionary with the following structure:
- "table_name": the name of the table.
- "columns": a list of column names.
- "rows": a list of lists, where each sublist represents a row of data.

Your task is to transform this dictionary into a list of dictionaries under the key "certificate_table", following these rules:

1. Column Mapping  
   - The first dictionary should map column indices as strings ("0", "1", ..., "N") to their corresponding column names in order.

2. Row Transformation  
   - Each subsequent dictionary should represent one row of data.  
   - Keys = column indices as strings.  
   - Values = cell values from that row.  
   - If a value is missing or the row has fewer elements than the number of columns, fill the missing values with null.

3. **Certificate Name Handling**  
   - If a cell in the Certificate Name column contains multiple certificates separated by newline characters ("\\n"):  
     - Keep the first certificate name in the original row.  
     - The second certificate name must **not create a new duplicate row**.  
     - Instead, place it into the nearest following row’s empty Certificate Name field (index "1"), while keeping that row’s existing data unchanged.  
     - Only if no such row exists, create a new row with that certificate name.  
     - Ensure no duplication of certificate numbers, dates, or country values.
    Example Input Rows:
    [
        ['00490/2024', 'Medical Care\\nShip Security Officer', '13.03.2024', '13.03.2029\\n13.03.2029', 'Ukraine', None, None, None, None],
        ['00737/2024', None, '10.03.2024', None, 'Ukraine', None, None, None, None]
    ]

    Example Output Rows:
    {{
    "0": "00490/2024",
    "1": "Medical Care",
    "2": null,
    "3": null,
    "4": "13-03-2024",
    "5": "13-03-2029",
    "6": null,
    "7": null,
    "8": "Ukraine"
    }},
    {{
    "0": "00737/2024",
    "1": "Ship Security Officer",
    "2": null,
    "3": null,
    "4": "10-03-2024",
    "5": null,
    "6": null,
    "7": null,
    "8": "Ukraine"
    }}
4. Date Splitting & Correction  
   - If any field contains two dates separated by "/" or "," (e.g., "17.04.2019/17.04.2024", "17.04.2019,17.04.2024"):  
     - Place the first date in the "DateOfIssue" column (index "4").  
     - Place the second date in the "DateOfExpiry" column (index "5").  
     - This applies regardless of which column originally contained the dates.
   - If "DateOfExpiry" contains two dates separated by "/" or "," (e.g., "17.04.2019/17.04.2024", "17.04.2019,17.04.2024"):
     - Place the first date in "DateOfExpiry" column of same cell(index "5").
     - Place it into the nearest following row’s empty "DateOfExpiry" (index "5"), while keeping that row’s existing data unchanged.
     - Also make sure that it doesn't create a new cell. 
     Example Input Rows:
     [
     ['MUM 131403', 'Seaman Book', '29-Jan-2014', '28-Jan-2024\n09-Nov-2031', 'India', None, None, None, None],
     ['Z6083144', 'Passport', '10-Nov-2021', None, 'India', None, None, None, None]
     ]
     
     Example Output Rows:
     {{
        "0": "CertificateNo",
        "1": "CertificateName",
        "2": "PlaceOfIssue",
        "3": "IssuedBy",
        "4": "DateOfIssue",
        "5": "DateOfExpiry",
        "6": "Grade",
        "7": "Others",
        "8": "CountryOfIssue"
      }},
      {{
        "0": "MUM 131403",
        "1": "Seaman Book",
        "2": null,
        "3": null,
        "4": "29-01-2014",
        "5": "28-01-2024",
        "6": null,
        "7": null,
        "8": "India"
      }},
      {{
        "0": "Z6083144",
        "1": "Passport",
        "2": null,
        "3": null,
        "4": "10-11-2021",
        "5": "09-Nov-2031",
        "6": null,
        "7": null,
        "8": "India"
      }}
      
      Apply this Mapping all over in the certificate_table section. 

5. Date Formatting  
   - Convert all dates into DD-MM-YYYY format.

6. Data Preservation  
   - Maintain the same number of fields per row as the number of columns.  
   - Do not drop or skip any rows or fields.  
   - Preserve the original row order.

7. Country of Issue Inference  
   - If PlaceOfIssue is provided but CountryOfIssue is missing, determine the country corresponding to the place and populate it.

8. Text Cleaning  
   - Fix broken words caused by accidental spaces (e.g., "Carri er" → "Carrier").

---



---

Here is the input dictionary:
{certificate_table}

Return the final JSON as a list of dictionaries under the key "certificate_table".
"""



    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data formatter."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    res_json = json.loads(response.choices[0].message.content)
    return res_json




def experience_openai(experience_table):

    prompt = f"""
    You are given a dictionary with the following structure:
    - "table_name": the name of the table.
    - "columns": a list of column names.
    - "rows": a list of lists, where each sublist represents a row corresponding to the columns.

    Your task is to convert this dictionary into a list of dictionaries with the following rules:

    1. The first dictionary should map column indices as strings ("0", "1", ..., etc.) to their corresponding column names in order.
    2. Each subsequent dictionary should represent one row of data, where:
       - Keys are column indices as strings.
       - Values are cell values from that row, matched to the correct column index.
       - If a value is missing or the row has fewer elements than the number of columns, fill the missing ones with null.
    3. If any cell contains multi-line values separated by newline characters (`\\n`), treat them as multiple rows, properly aligned.
    4. If any cell in the FromDt (index "5") or ToDt (index "6") column contains **two dates separated by a newline character (`\\n`)**, such as `"09.08.18\\n30.03.19"` or `"27.04.19\\n20.10.20"`:
        - Do NOT split this row into multiple rows.
        - Instead, split the two dates into separate values.
        - Assign the **first date** to FromDt (index "5").
        - Assign the **second date** to ToDt (index "6").
        - Keep all other data in the row unchanged.
        - This ensures the row remains one complete record with FromDt and ToDt properly filled.
    5. If any row contains mostly null values except for **one or more isolated non-null fields**, consider this row a **continuation or supplemental data** for a nearby related row:
       - The model must identify the nearest logical row missing that data.
       - Merge or assign the orphan data into that row’s appropriate field(s).
       - Remove these orphan rows from the final output after merging.
       - This applies to *any* column, such as dates, flags, places, or others.
    6. Maintain the same number of fields per row and ensure no data is lost.
    7. Preserve the original order as much as possible while maintaining data integrity.
    8. Convert all dates to DD-MM-YYYY format.
    9. Fix broken words caused by accidental spaces (e.g., 'Carri er')
    10. Do not merge or overwrite existing entries. If a new row contains only one non-null field and the rest are null, it must be treated as a separate, junk row and excluded from the final output. Do not use such rows to update or modify the previous row.
    11. DO NOT drop or skip any rows or fields in the experience_table
    


    Here is the input dictionary:
    {experience_table}

    Return the final JSON as a list of dictionaries under the key "experience_table".
    """

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data formatter."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    res_json = json.loads(response.choices[0].message.content)
    return res_json



def send_to_gpt(transformed_data):

    # date_fields_str = json.dumps(date_fields)
    # extracted_info_str = json.dumps(transformed_data)

    prompt = f"""
    you have a JSON {transformed_data}, Please apply the following transformations in the JSON:

    "Convert all dates to DD-MM-YYYY format",
    "Replace country names in Nationality with their demonym (e.g., Russia → Russian)",
    "Normalize gender values (male → Male, female → Female)",
    "Fix PlaceOfIssue and CountryOfIssue according to defined rules",
    "Deduplicate values and propagate where needed in tables",
    "Remove \\n from all string fields",
    "Fix broken words caused by accidental spaces (e.g., 'Carri er')",
    "Normalize Position values and standardize rank terms",
    "⚠️ DO NOT drop or skip any rows or fields in the certificate_table or experience_table"

    """

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=OPENAI_API_VERSION,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data formatter."},
            {"role": "user", "content": f"Apply the following transformations to the uploaded JSON objects. Return both fully. Do not skip or remove any data. Instructions: {prompt}"}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    res_json = json.loads(response.choices[0].message.content)

    return res_json


async def convert_docx_to_pdf(docx_path):
    """ Converts DOCX to PDF using LibreOffice (Linux) or Microsoft Word (Windows). """
    pdf_path = docx_path.replace(".docx", ".pdf")

    try:
        if platform.system() == "Windows":
            import win32com.client
            word = win32com.client.Dispatch("Word.Application")
            doc = word.Documents.Open(os.path.abspath(docx_path))
            doc.SaveAs(os.path.abspath(pdf_path), FileFormat=17)  # PDF format
            doc.Close()
            word.Quit()
            # print(f" Converted {docx_path} to {pdf_path} using Microsoft Word")
            logger.info("Converted %s to %s using Microsoft Word", docx_path, pdf_path)
        else:
            libreoffice_path = "/usr/bin/libreoffice"
            if not os.path.exists(libreoffice_path):
                raise FileNotFoundError(f"LibreOffice not found at {libreoffice_path}")

            process = await asyncio.create_subprocess_exec(
                libreoffice_path, "--headless", "--convert-to", "pdf",
                "--outdir", os.path.dirname(docx_path), docx_path
            )
            await process.communicate()  # Ensure subprocess completes
            logger.info("Converted %s to %s using LibreOffice", docx_path, pdf_path)
            # print(f" Converted {docx_path} to {pdf_path} using LibreOffice")

        return pdf_path
    except Exception as e:
        # print(f" DOCX to PDF conversion failed: {e}")
        logger.error("DOCX to PDF conversion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"DOCX to PDF conversion failed: {e}")

def replace_values(data, mapping):
    if isinstance(data, dict):
        return {mapping.get(key, key): replace_values(value, mapping) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_values(item, mapping) for item in data]
    elif isinstance(data, str):
        return mapping.get(data, data)  # Replace if found, else keep original
    return data

def replace_rank(json_data, rank_mapping):
    # Convert rank_mapping keys to lowercase for case-insensitive replacement
    rank_mapping = {key.lower(): value for key, value in rank_mapping.items()}

    if isinstance(json_data, dict):
        return {
            key: replace_rank(value, rank_mapping) if key != "2" else  # "2" corresponds to "Position"
            rank_mapping.get(value.lower(), value) if isinstance(value, str) else value
            for key, value in json_data.items()
        }
    elif isinstance(json_data, list):
        return [replace_rank(item, rank_mapping) for item in json_data]
    return json_data



def replace_country(data, mapping):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            # Process both keys and values
            new_key = mapping.get(key, key) if isinstance(key, str) else key
            new_value = replace_country(value, mapping)
            new_dict[new_key] = new_value
            
            # Print if value changed (only for string values)
            if isinstance(value, str) and value != new_value:
                # print(f"Mapping country: '{value}' → '{new_value}'")
                logger.debug("Mapping country: '%s' → '%s'", value, new_value)
        return new_dict
        
    elif isinstance(data, list):
        new_list = []
        for item in data:
            new_item = replace_country(item, mapping)
            new_list.append(new_item)
        return new_list
        
    elif isinstance(data, str):
        return mapping.get(data, data)
        
    return data




    
def reposition_fields(table_data, desired_order):
    updated_table_data = []
    
    # Get the header row
    header = table_data[0]
    
    # Create a mapping of field names to index positions
    key_mapping = {value: key for key, value in header.items()}
    
    for row in table_data:
        reordered_row = {}

        # Ensure all specified fields are placed correctly
        for i, field in enumerate(desired_order):
            reordered_row[str(i)] = row.get(key_mapping.get(field, ""), "")

        # Add remaining fields starting at index after the last specified field
        remaining_keys = [k for k in row.keys() if k not in key_mapping.values()]
        for j, key in enumerate(remaining_keys, start=len(desired_order)):
            reordered_row[str(j)] = row.get(key, "")

        updated_table_data.append(reordered_row)

    return updated_table_data

