from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Security, Depends,Body
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
import shutil
import json
import tempfile
from datetime import datetime
from country_mapping import country_mapping
from doc_intelligence_with_formatting import basic_openai,certificate_openai, experience_openai, reposition_fields, validate_parsed_resume, extract_resume_info, replace_values, replace_rank, convert_docx_to_pdf,replace_country
from rank_map_dict import rank_mapping
from dict_file import mapping_dict
import os
import asyncio
from current_doc_parsor import process_document_to_json
import ast
import pandas as pd
import io
import time
import schedule
from sqlalchemy import create_engine, text, Column, Integer, String, DateTime, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import List, Dict
from azure.storage.blob import BlobServiceClient
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from sqlalchemy import text
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from filelock import FileLock, Timeout
from azure.core.exceptions import ResourceExistsError
import logging
import sys
from dotenv import load_dotenv
import re
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("resume_parser")

load_dotenv()
app = FastAPI(title="Resume Parser API", version="1.0")

# Secure API Key Authentication
API_KEY = os.getenv("your_secure_api_key")
API_KEY_NAME = os.getenv("api_key_name")
endpoint = os.getenv("endpoint")
key = os.getenv("key")
model_id = os.getenv("model_id")
container_name = os.getenv("container_name")
connection_string = os.getenv("connection_string")
AZURE_CONNECTION_STRING=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME= os.getenv("AZURE_BLOB_CONTAINER_NAME")


basic_details_order = [
    "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender",
    "Doa", "Dob", "Address1", "Address2", "Address3", "Address4", "City",
    "State", "Country", "ZipCode", "EmailId", "MobileNo", "AlternateNo", "Rank"
]
 
 
experience_table_order = [
    "VesselName", "VesselType", "Position", "VesselSubType", "Employer",
    "Flag", "IMO", "FromDt", "ToDt", "others"
]
 
certificate_table_order = [
    "CertificateNo", "CertificateName", "PlaceOfIssue", "IssuedBy", "DateOfIssue",
    "DateOfExpiry", "Grade", "Others", "CountryOfIssue"
]
 
# Define API Key Security
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
 
   
def verify_api_key(api_key: str = Security(api_key_header)):
    """Validate API Key"""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail=" Invalid API Key")
    return api_key
 
 
def clean_vessel_names(experience_table):
    if not experience_table or len(experience_table) < 2:
        return experience_table
 
    # Identify the header index for "VesselName"
    header = experience_table[0]
    vessel_name_index = None
    for key, value in header.items():
        if value.lower() == "vesselname":
            vessel_name_index = key
            break
 
    if vessel_name_index is None:
        return experience_table  # Skip if "VesselName" not found
 
    # Updated pattern includes hyphens after prefix
    prefix_pattern = r'^(M[\s./\\]?[V|T][\s.\-]*)'
 
    # Clean each row
    for row in experience_table[1:]:
        vessel_name = row.get(vessel_name_index, "")
        if vessel_name:
            # Remove the prefix if it matches
            cleaned_name = re.sub(prefix_pattern, '', vessel_name, flags=re.IGNORECASE).strip()
            row[vessel_name_index] = cleaned_name
 
    return experience_table
 
@app.post("/upload/")
async def upload_file(
    api_key: str = Depends(verify_api_key),  # Enforce API key authentication
    file: UploadFile = File(...),
    entity: str = Form("")
):
    try:
        # Extract file extension
        suffix = os.path.splitext(file.filename)[-1]
        if suffix not in [".pdf", ".docx"]:
            raise HTTPException(status_code=400, detail="Only PDF and Word documents are allowed")
 
        # Generate custom filename with timestamp
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        temp_file_path = os.path.join(tempfile.gettempdir(), f"{timestamp}{suffix}")
 
        # Write the uploaded file to the custom temp path
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
 
        # Handle .docx conversion if needed
        if suffix == ".docx":
            temp_file_path = await convert_docx_to_pdf(temp_file_path)  
 
        # Extract JSON from document
        extracted_info = extract_resume_info(endpoint, key, model_id, temp_file_path)
 
 
        fields_only = extracted_info["fields"] 
 
        tables = extracted_info.get('tables', [])
        certificate_table = None
        experience_table = None
       
        for table in tables:
            if table.get('table_name') == 'certificate_table':
                certificate_table = table
            elif table.get('table_name') == 'experience_table':
                experience_table = table
 
       
        
        logger.info("Certificate Table: %s", certificate_table)
        logger.info("Experience Table: %s", experience_table)
        
 
        basic_out, cert_out, expe_out = await asyncio.gather(
            asyncio.to_thread(basic_openai, fields_only),
            asyncio.to_thread(certificate_openai, certificate_table),
            asyncio.to_thread(experience_openai, experience_table),
        )
 
        basic_details_merge = basic_out['basic_details']
        certificate_table_merge = cert_out['certificate_table']

        logger.info("basic_details_merge: %s", basic_details_merge)
        logger.info("certificate_table_merge: %s", certificate_table_merge)

         # --- Inject Gender Logic ---
        if basic_details_merge and len(basic_details_merge) > 1:
            header = basic_details_merge[0]
            values = basic_details_merge[1]
 
            gender_index = None
            for keyy, value in header.items():
                if value.lower() == "gender":
                    gender_index = keyy
                    break
 
            if gender_index is not None:
                gender_value = str(values.get(gender_index, "")).strip().lower()
 
                if gender_value not in ["male", "female"]:
                    values[gender_index] = "Male"
                else:
                    values[gender_index] = gender_value.capitalize()
        # --------------------------
 

        experience_table_merge = expe_out['experience_table']
        logger.info("experience_table_merge (raw): %s", experience_table_merge)
       
        experience_table_merge = clean_vessel_names(experience_table_merge)

 
        desired_basic_order = [
            "Name", "FirstName", "MiddleName", "LastName", "Nationality", "Gender",
            "Doa", "Dob", "Address1", "Address2", "Address3", "Address4",
            "City", "State", "Country", "ZipCode", "EmailId", "MobileNo",
            "AlternateNo", "Rank"
        ]
 
        desired_cert_order = [
            "CertificateNo",       # 0
            "CertificateName",     # 1
            "PlaceOfIssue",        # 2
            "IssuedBy",            # 3
            "DateOfIssue",         # 4
            "DateOfExpiry",        # 5
            "Grade",               # 6
            "Others",              # 7
            "CountryOfIssue"       # 8
        ]
 
 
        desired_experience_order = [
            "VesselName",     # 0
            "VesselType",     # 1
            "Position",       # 2
            "VesselSubType",  # 3
            "Employer",       # 4
            "Flag",           # 5
            "IMO",            # 6
            "FromDt",         # 7
            "ToDt",           # 8
            "others"          # 9
        ]
 
        def reorder_basic_details_table(data):
            if not data:
                return []
 
            # Get current header
            current_header = data[0]
           
            # Map field name to current index
            name_to_index = {v: k for k, v in current_header.items()}
           
            # New header in desired order
            new_header = {str(i): field_name for i, field_name in enumerate(desired_basic_order)}
 
            reordered_data = [new_header]
 
            for row in data[1:]:
                new_row = {}
                for new_idx, field_name in enumerate(desired_basic_order):
                    old_idx = name_to_index.get(field_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)
 
            return reordered_data
 
 
        def reorder_experience_table(data):
            if not data:
                return []
 
            # Get the current header (mapping of index to field names)
            current_header = data[0]
           
            # Create a mapping from field name to current index
            name_to_index = {v: k for k, v in current_header.items()}
           
            # Create new header in desired order
            new_header = {str(i): field_name for i, field_name in enumerate(desired_experience_order)}
 
            # Rebuild the table in the new order
            reordered_data = [new_header]
 
            for row in data[1:]:
                new_row = {}
                for new_idx, field_name in enumerate(desired_experience_order):
                    old_idx = name_to_index.get(field_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)
 
            return reordered_data
       
 
        def reorder_certificate_table(data):
            if not data:
                return []
 
            # Step 1: Get current header (first element is a dict with index keys)
            current_header = data[0]
           
            # Build mapping: "CertificateNo" => "0", etc.
            name_to_index = {v: k for k, v in current_header.items()}
           
            # Step 2: Build new header with desired order
            new_header = {str(i): col_name for i, col_name in enumerate(desired_cert_order)}
           
            # Step 3: Reorder all rows based on desired order
            reordered_data = [new_header]
           
            for row in data[1:]:
                new_row = {}
                for new_idx, col_name in enumerate(desired_cert_order):
                    old_idx = name_to_index.get(col_name)
                    new_row[str(new_idx)] = row.get(old_idx) if old_idx is not None else None
                reordered_data.append(new_row)
 
            return reordered_data
 
 
        reordered_basic = reorder_basic_details_table(basic_details_merge)
        # print(reordered_basic)
        reordered_certificates = reorder_certificate_table(certificate_table_merge)
        # print(reordered_certificates)
        reordered_experience = reorder_experience_table(experience_table_merge)
        # print(reordered_experience)
        
        logger.debug("Reordered Basic: %s", reordered_basic)
        logger.debug("Reordered Certificates: %s", reordered_certificates)
        logger.debug("Reordered Experience: %s", reordered_experience)
 
           
        final_output = {
            "status": "success",
            "data": {
                "basic_details": reordered_basic,
                "experience_table": reordered_experience,
                "certificate_table": reordered_certificates
            },
            "utc_time_stamp": datetime.utcnow().strftime("%d/%m/%Y, %H:%M:%S")
        }
 
 
        validation_errors = validate_parsed_resume(extracted_info, temp_file_path, 0.8, container_name, connection_string)
        # print(validation_errors)
        logger.info("Validation result: %s", validation_errors)
 
 
 
        # course_map = replace_values(final_output, mapping_dict)

        current_mapping_dict = load_mapping_dict_from_blob()   # NEW
        
        course_map = replace_values(final_output, current_mapping_dict)

        rank_map = replace_rank(course_map, rank_mapping)
        rank_map=replace_country(rank_map,country_mapping)
        final_output['data']['basic_details'] = replace_country(rank_map['data']['basic_details'], country_mapping)
        final_output['data']['certificate_table'] = replace_country(rank_map['data']['certificate_table'], country_mapping)
 
        # rank_map_dict = json.loads(rank_map)
 
 
       
        basic_details = rank_map.get('data', {}).get('basic_details', [])
        experience_table = rank_map.get('data', {}).get('experience_table', [])
        certificate_table = rank_map.get('data', {}).get('certificate_table', [])
 
        # Reposition columns only if the section exists
        if basic_details:
            basic_details = reposition_fields(basic_details, basic_details_order)
        if experience_table:
            experience_table = reposition_fields(experience_table, experience_table_order)
        if certificate_table:
            certificate_table = reposition_fields(certificate_table, certificate_table_order)
 
        # Update the input_json with the new order (only if they exist)
        rank_map.setdefault('data', {})
        rank_map['data']['basic_details'] = basic_details
        rank_map['data']['experience_table'] = experience_table
        rank_map['data']['certificate_table'] = certificate_table
 
        # # transformed_data = send_to_gpt(rank_map)
 
        experience_table = rank_map["data"]["experience_table"]
        if experience_table:
            # Keep the header row intact
            filtered_experience = [experience_table[0]]
   
            # Define the null threshold (remove if more than 8 are null out of 10)
            max_allowed_nulls = 8
   
            # Filter rows
            for row in experience_table[1:]:
                # Treat as valid row only if it's a full record with limited nulls
                if isinstance(row, dict):
                    null_count = sum(1 for v in row.values() if v is None)
                    if null_count < max_allowed_nulls:
                        filtered_experience.append(row)
   
            # Update the data with filtered experience_table
            rank_map["data"]["experience_table"] = filtered_experience
            logger.info("Filtered experience_table: %s", filtered_experience)
            # print(filtered_experience)
        return rank_map
       
    except HTTPException as http_exc:
        # FastAPI will return this as-is
        raise http_exc
 
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error("Unexpected error in upload_file: %s", e, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "An unexpected error occurred during resume processing.",
                "detail": str(e)
            }
        )
 
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug("Temporary file removed: %s", temp_file_path)
 
# Endpoint for uploading the file
@app.post("/upload-document/")
async def upload_file(api_key: str = Depends(verify_api_key), file: UploadFile = File(...), Doctype: str = Form("")):
   
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".pdf", ".jpg", ".jpeg", ".png", ".docx"]:
        return JSONResponse({"error": "Only PDF, DOCX, or image files allowed."}, status_code=400)
 
    # Handle DOCX conversion if the file is a DOCX
    if file_ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        try:
            pdf_path = await convert_docx_to_pdf(temp_file_path)
            result = process_document_to_json(pdf_path)

            # Load fresh mapping from Blob and normalize/flatten
            mapping_dict_from_blob = load_mapping_dict_from_blob()   # NEW
            print("load_mapping_dict_from_blob:",mapping_dict_from_blob, flush=True)
            normalized_mapping = {}
            for key, value in mapping_dict_from_blob.items():
                aliases = [alias.strip().lower() for alias in key.split("/")]
                for alias in aliases:
                    normalized_mapping[alias] = value

 
            # Perform docName mapping
            for item in result:
                original_doc_name = item.get("docName", "").strip().lower()
                mapped_doc_name = normalized_mapping.get(original_doc_name)
                if mapped_doc_name:
                    item["docName"] = mapped_doc_name
 
            return JSONResponse(content=result, media_type="application/json")
       
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            os.remove(temp_file_path)
    else:
        # Handle PDF and image files as before
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
 
        try:
            result = process_document_to_json(temp_file_path)
 
            # If result is a JSON string, convert it to a Python object
            if isinstance(result, str):
                result = json.loads(result)
 
            # Normalize and flatten the mapping dictionary
            normalized_mapping = {}
            for key, value in mapping_dict.items():
                aliases = [alias.strip().lower() for alias in key.split("/")]
                for alias in aliases:
                    normalized_mapping[alias] = value
 
            # Perform docName mapping
            for item in result:
                original_doc_name = item.get("docName", "").strip().lower()
                mapped_doc_name = normalized_mapping.get(original_doc_name)
                if mapped_doc_name:
                    item["docName"] = mapped_doc_name
 
                original_country = item.get("issuedCountry", "").strip().lower()
                for key, value in country_mapping.items():
                    if key.strip().lower() == original_country:
                        item["issuedCountry"] = value
                        break
 
            return JSONResponse(content=result, media_type="application/json")
 
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            os.remove(temp_file_path)
 
 
import ast


# ------------ Mapping dict loader (from Azure Blob) ------------
MAPPING_BLOB_NAME = "dict_file.py"

# Simple in-process cache to avoid re-downloading each request
_mapping_cache = {"etag": None, "data": {}}

def _parse_mapping_dict_py(file_text: str) -> dict:
    """
    Expects a Python file that contains:
        mapping_dict = { ... }
    Returns the dict safely using ast.literal_eval.
    """
    m = re.search(r"mapping_dict\s*=\s*(\{.*\})\s*\Z", file_text, flags=re.DOTALL)
    if not m:
        # Fallback: split once
        parts = file_text.split("=", 1)
        if len(parts) == 2:
            return ast.literal_eval(parts[1].strip())
        return {}
    return ast.literal_eval(m.group(1))

def load_mapping_dict_from_blob(force: bool = False) -> dict:
    """
    Download dict_file.py from Azure Blob and return the mapping dict.
    Uses an ETag-based cache to avoid unnecessary downloads.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=MAPPING_BLOB_NAME)

        props = blob_client.get_blob_properties()
        if (not force) and _mapping_cache["etag"] == props.etag and _mapping_cache["data"]:
            return _mapping_cache["data"]

        content_bytes = blob_client.download_blob().readall()
        content_text = content_bytes.decode("utf-8", errors="replace")
        data = _parse_mapping_dict_py(content_text)
        print("----------------------------------------------------------------data",data)
      

        _mapping_cache["etag"] = props.etag
        _mapping_cache["data"] = data
        return data
    except Exception as e:
        # Graceful fallback: keep last good cache if present
        if _mapping_cache["data"]:
            return _mapping_cache["data"]
        print(f"[WARN] load_mapping_dict_from_blob failed: {e}. Using empty mapping.", flush=True)
        return {}


# ------------------------------------------------------------------------------
# SQLAlchemy setup
# ------------------------------------------------------------------------------
Base = declarative_base()
import urllib.parse
def get_db_engine():
    user = "ai-admin"
    password = "@Oceanai2024"
    server = "oceanai.database.windows.net"
    database = "resume_parser"
    driver = "ODBC Driver 18 for SQL Server"
    # Encode credentials
    password_enc = urllib.parse.quote_plus(password)
    driver_enc = urllib.parse.quote_plus(driver)

    # Create SQLAlchemy engine
    engine_url = f"mssql+pyodbc://{user}:{password_enc}@{server}/{database}?driver={driver_enc}&Encrypt=yes&TrustServerCertificate=no"
    engine = create_engine(engine_url, pool_pre_ping=True)
    
    return engine

# ------------------------------------------------------------------------------
# ORM models
# ------------------------------------------------------------------------------
class TempDocument(Base):
    __tablename__ = 'temp_table'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String, nullable=False)
    unidentified_doc_name = Column(String, nullable=False)
    mapped_doc_name = Column(String, nullable=False)
    status = Column(String, default='pending')
    CreatedDate = Column(DateTime, default=datetime.now)

class MasterDocument(Base):
    __tablename__ = 'Master_unidentified_doc_Table'
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_name = Column(String, nullable=False)
    unidentified_doc_name = Column(String, nullable=False)
    mapped_doc_name = Column(String, nullable=False)
    uploaded_date = Column(DateTime, default=datetime.now)
    status = Column(String, default='pending')


def init_db():
    """
    Initialize Azure SQL database
    """
    try:
        engine = get_db_engine()
        
        # Test connection first
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("[DB] Azure SQL connection test successful", flush=True)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print("[DB] Azure SQL tables initialized", flush=True)
        
    except Exception as e:
        print(f"[DB] Initialization failed: {e}", flush=True)
        # Re-raise the exception to prevent app startup if DB is critical
        raise



# ------------------------------------------------------------------------------
# Fetch Mapped Documents from the Database
# ------------------------------------------------------------------------------
def fetch_mapped_documents():
    engine = get_db_engine()
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT document_name, unidentified_doc_name, mapped_doc_name 
            FROM Master_unidentified_doc_Table
            WHERE status = 'pending' OR status = 'exported'
        """))
        return result.fetchall()

# ------------------------------------------------------------------------------
# Update Mapping Dictionary and Write to dict_file.py
# ------------------------------------------------------------------------------



def download_dict_file_from_blob(container_name, blob_name, download_path):
    try:
        # Get the Azure Blob Service Client
        # connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)

        # Download the blob (dict_file.py) to the specified path
        blob_client = container_client.get_blob_client(blob_name)
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())

        print(f"[INFO] dict_file.py downloaded successfully from Blob storage to {download_path}")

    except Exception as e:
        print(f"[ERROR] Failed to download dict_file.py from Blob: {e}")


# Function to escape newlines, quotes, and special characters properly
def escape_value(value):
    value = value.replace("\n", "\\n")
    value = value.replace('"', '\\"')
    return value

def update_mapping_dict():
    rows = fetch_mapped_documents()

    # Initialize the dictionary (load existing one from Blob storage)
    mapping_dict = {}

    try:
        # Step 1: Download the current dict_file.py from Blob storage
        download_dict_file_from_blob(AZURE_CONTAINER_NAME, 'dict_file.py', 'dict_file_local.py')
        
         # Step 2: Load the downloaded dict file locally
        with open('dict_file_local.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():  # Ensure the file isn't empty
                # Load the dictionary safely from the file
                mapping_dict = ast.literal_eval(content.split('=', 1)[1].strip())  # This assumes dict_file.py contains "mapping_dict = {...}"

    except FileNotFoundError:
        print("[INFO] dict_file.py not found, starting fresh.")

    # Step 3: Loop through the fetched rows from the MasterDocument table and add new mappings if they don't exist
    for row in rows:
        unidentified_name = row[0]  # Accessing by index
        mapped_name = row[1]        # Accessing by index
        if unidentified_name not in mapping_dict:
            mapping_dict[unidentified_name] = mapped_name

        # Step 4: Write the updated mapping_dict to a temporary file
    with open('dict_file_updated.py', 'w', encoding='utf-8') as f:
        f.write("mapping_dict = {\n")
        for key, value in mapping_dict.items():
            key = escape_value(key)
            value = escape_value(value)
            f.write(f'    "{key}": "{value}",\n')
        f.write("}\n")
        
        
    # Step 5: Upload the updated dict_file.py back to Blob storage
    upload_dict_file_to_blob('dict_file_updated.py', AZURE_CONTAINER_NAME, 'dict_file.py')

    print("[INFO] dict_file.py updated and uploaded to Blob storage.")


def pandas_to_sql_fast(df: pd.DataFrame, table_name: str, engine):
    # Use raw_connection to ensure pyodbc cursor available for fast_executemany
    conn = engine.raw_connection()
    cursor = conn.cursor()
    try:
        try:
            cursor.fast_executemany = True
        except Exception:
            pass
        # Use pandas to_sql that uses the SQLAlchemy engine; commit using raw connection
        df.to_sql(table_name, con=engine, if_exists='append', index=False, method=None)
        conn.commit()
    finally:
        try:
            cursor.close()
            conn.close()
        except Exception:
            pass


def upload_dict_file_to_blob(file_path, container_name, blob_name):
    try:
        # Get the Azure Blob Service Client
        # connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)

        # Upload the file to the container
        with open(file_path, "rb") as data:
            container_client.upload_blob(name=blob_name, data=data, overwrite=True)

        print(f"[INFO] dict_file.py uploaded successfully to Blob storage as {blob_name}")

    except Exception as e:
        print(f"[ERROR] Failed to upload dict_file.py to Blob: {e}")


# ------------------------------------------------------------------------------
# API Input Model
# ------------------------------------------------------------------------------
class DocumentMapping(BaseModel):
    document_name:str
    unidentified_doc_name: str
    mapped_doc_name: str



@app.post("/insert-temp-documents/")
def insert_temp_documents(
    api_key: str = Depends(verify_api_key),
    mappings: List[Dict[str, str]] = Body(...),
):
    try:
        if not mappings or not isinstance(mappings[0], dict):
            return {"error": "Invalid format. Expected a dictionary inside a list."}

        df = pd.DataFrame([{
            "document_name": (item.get("document_name") or "").strip().lower(),
            "unidentified_doc_name": (item.get("unidentified_doc_name") or "").strip().lower(),
            "mapped_doc_name": (item.get("mapped_doc_name") or "").strip().lower()
        } for item in mappings])

        df['status'] = 'pending'
        df['CreatedDate'] = datetime.utcnow()   # use UTC

        engine = get_db_engine()
        pandas_to_sql_fast(df, 'temp_table', engine)

        return {"message": f"{len(df)} document(s) inserted as pending."}
    except Exception as e:
        logger.error("Insertion failed: %s", e, exc_info=True)
        return {"error": f"Insertion failed: {str(e)}"}

# ------------------------------------------------------------------------------
# Scheduled Task: Export to Excel and upload to Blob
# ------------------------------------------------------------------------------
def export_data_to_excel():
    print("[TASK] export_data_to_excel START", flush=True)
    try:

        engine = get_db_engine()
        with engine.begin() as conn:
            
            result = conn.execute(text("""
                SELECT * FROM temp_table
                WHERE LTRIM(RTRIM(status)) = 'pending'
                AND CONVERT(date, CreatedDate) < CONVERT(date, GETDATE())
            """))

            data = result.fetchall()
            print(f"[TASK] Fetched {len(data)} rows", flush=True)
            if not data:
                print("[TASK] Nothing to export. Returning.", flush=True)
                return

            df = pd.DataFrame(data, columns=result.keys())
            print(f"[TASK] DataFrame shape = {df.shape}", flush=True)

            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            print("[TASK] Exported data to Excel (in-memory)", flush=True)

            blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

            container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            try:
                container_client.create_container()
                print(f"[BLOB] Created container '{AZURE_CONTAINER_NAME}'", flush=True)
            except ResourceExistsError:
                pass

            blob_name = f"verification_documents_exported_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)
            blob_client.upload_blob(excel_buffer, overwrite=True)
            print(f"[TASK] Uploaded Excel to blob: {blob_name}", flush=True)

            with engine.begin() as conn2:
                conn2.execute(text("""
                    UPDATE temp_table
                    SET status='exported'
                    WHERE LTRIM(RTRIM(status)) = 'pending'
                    AND CONVERT(date, CreatedDate) < CONVERT(date, GETDATE())
                """))


                print("[TASK] Updated rows to 'exported'", flush=True)
                conn2.execute(text("""
                    DELETE FROM temp_table
                    WHERE LTRIM(RTRIM(status)) = 'exported'
                    AND CONVERT(date, CreatedDate) < CONVERT(date, GETDATE())
                """))
                # conn2.execute(text("DELETE FROM temp_table WHERE TRIM(status) = 'exported'"))
                print("[TASK] Deleted exported rows", flush=True)

    except Exception as e:
        print(f"[TASK] Export Error: {e}", flush=True)
    finally:
        print("[TASK] export_data_to_excel END", flush=True)

# ------------------------------------------------------------------------------
# Get latest replied blob
# ------------------------------------------------------------------------------
def get_latest_replied_blob_name():
    print("[TASK] get_latest_replied_blob_name START", flush=True)
    try:
        # conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        # container = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)

        blobs = list(container_client.list_blobs(name_starts_with="verification_documents_replied_"))
        replied_blobs = sorted(
            [b for b in blobs if b.name.endswith(".xlsx")],
            key=lambda b: b.last_modified,
            reverse=True
        )

        if not replied_blobs:
            raise Exception("No replied verification documents found in blob storage.")

        latest = replied_blobs[0].name
        print(f"[TASK] Latest replied blob: {latest}", flush=True)
        return latest
    finally:
        print("[TASK] get_latest_replied_blob_name END", flush=True)

# ------------------------------------------------------------------------------
# Insert data from replied blob
# ------------------------------------------------------------------------------
def insert_data_from_blob(blob_name: str):
    print(f"[TASK] insert_data_from_blob START ({blob_name})", flush=True)
    try:
        # conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        # container = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

        blob_data = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)\
            .download_blob().readall()
        df = pd.read_excel(io.BytesIO(blob_data))
        print(f"[TASK] Read {len(df)} rows from replied Excel; columns={list(df.columns)}", flush=True)

        engine = get_db_engine()
        inserted = 0
        with engine.begin() as conn:
            for _, row in df.iterrows():

                
                existing = conn.execute(text("""
                    SELECT 1 FROM Master_unidentified_doc_Table 
                    WHERE document_name = :document_name 
                    AND unidentified_doc_name = :unidentified_doc_name 
                    AND mapped_doc_name = :mapped_doc_name
                """), {
                    "document_name": row['document_name'],
                    "unidentified_doc_name": row['unidentified_doc_name'],
                    "mapped_doc_name": row['mapped_doc_name']
                }).fetchone()

                if not existing:
                    conn.execute(text("""
                        INSERT INTO Master_unidentified_doc_Table
                        (document_name, unidentified_doc_name, mapped_doc_name, uploaded_date, status)
                        VALUES (:document_name, :unidentified_doc_name, :mapped_doc_name, :uploaded_date, :status)
                    """), {
                        "document_name": row['document_name'],
                        "unidentified_doc_name": row['unidentified_doc_name'],
                        "mapped_doc_name": row['mapped_doc_name'],
                        "uploaded_date": row['CreatedDate'],
                        "status": row['status']
                    })
                    inserted += 1

        print(f"[TASK] Inserted {inserted} new rows into Master_unidentified_doc_Table", flush=True)
    except Exception as e:
        print(f"[TASK] Insertion from blob failed: {e}", flush=True)
    finally:
        print("[TASK] insert_data_from_blob END", flush=True)

# ------------------------------------------------------------------------------
# Combined scheduled task
# ------------------------------------------------------------------------------
def run_both_tasks():
    logger.info("[JOB] run_both_tasks START")
    try:
        export_data_to_excel()
        latest_blob = get_latest_replied_blob_name()
        insert_data_from_blob(latest_blob)
        update_mapping_dict()
    except Exception:
        logger.exception("[JOB] run_both_tasks failed")
    finally:
        logger.info("[JOB] run_both_tasks END")


# ------------------------------------------------------------------------------
# API: view temp documents
# ------------------------------------------------------------------------------
@app.get("/view-temp-documents/")
def view_temp_documents(api_key: str = Depends(verify_api_key)):
    print("[API] /view-temp-documents called", flush=True)
    try:
        # print(f"[SANITY] DB path (view) = {sqlite_db_path}", flush=True)
        engine = get_db_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()

        result = session.query(TempDocument).all()
        print(f"[API] Retrieved {len(result)} rows from temp_table", flush=True)

        data = [{
            "id": record.id,
            "document_name": record.document_name,
            "unidentified_doc_name": record.unidentified_doc_name, 
            "mapped_doc_name": record.mapped_doc_name,
            "status": record.status, 
            "CreatedDate": record.CreatedDate.strftime('%Y-%m-%d %H:%M:%S') if record.CreatedDate else None
        } for record in result]

        return JSONResponse(content={"data": data})
    except Exception as e:
        print(f"[API] view-temp-documents failed: {e}", flush=True)
        return {"error": f"Failed to fetch data: {str(e)}"}

@app.post("/delete-temp-documents/")
def delete_temp_documents(api_key: str = Depends(verify_api_key)):
    try:
        engine = get_db_engine()
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM temp_table"))
        return {"message": "All documents from temp_table have been deleted."}
    except Exception as e:
        return {"error": f"Deletion failed: {str(e)}"}


# ------------------------------------------------------------------------------
# Scheduler (12:45 AM IST daily) with single-instance guard
# ------------------------------------------------------------------------------
SCHED_TZ = pytz.timezone("Asia/Kolkata")
scheduler = AsyncIOScheduler(timezone=SCHED_TZ)
SCHED_LOCK_PATH = "/tmp/scheduler.lock"


def start_scheduler_guarded(): 
    if os.getenv("RUN_SCHEDULER", "0") != "1":
        print("[SCHEDULER] Skipped (RUN_SCHEDULER not set)", flush=True)
        return

    lock = FileLock(SCHED_LOCK_PATH)
    try:
        
        print("scheduler running.................................................................")
        with lock.acquire(timeout=0):  # only one worker wins
            scheduler.add_job(
                run_both_tasks,
                CronTrigger(
                    hour=14,             # Current hour
                    minute=39,           # Current minute
                    timezone=SCHED_TZ    # Correct timezone (Asia/Kolkata)
                ),
                id="run_both_tasks_now",   # Change the ID to reflect immediate execution
                replace_existing=True
            )

            scheduler.start()

            job = scheduler.get_job("run_both_tasks_now")
            print("[SCHEDULER] APScheduler started", flush=True)
            if job is not None:
                print("[SCHEDULER] Next run time:", job.next_run_time, flush=True)
            else:
                print("[SCHEDULER] Job not found!", flush=True)
    except Timeout:
        print("[SCHEDULER] Skipped (another worker owns the scheduler)", flush=True)


@app.on_event("startup")
async def on_startup():
    print("current_date:", datetime.now().strftime("%d-%m-%Y"), flush=True)
    init_db()
    start_scheduler_guarded()

@app.on_event("shutdown")
async def shutdown_scheduler():
    try:
        scheduler.shutdown(wait=False)
        print("[SCHEDULER] APScheduler stopped", flush=True)
    except Exception:
        pass