
import os
import json
import uuid
import ast
import re
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, ValidationError, model_validator
from dotenv import load_dotenv
from loguru import logger
import openai
import jsonschema

# Load environment variables
load_dotenv()

# ------------------- Configuration -------------------

class Config:
    """Configuration management for API keys and settings."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048
    SYSTEM_PROMPT: str = (
        "You are a professional Email Parser Agent for data analytics. Your tasks are to normalize email input formats, "
        "load and apply validation schemas, extract structured data using AI, format types according to schema (with special handling for list fields), "
        "and build structured responses. At each stage, handle errors gracefully and provide clear, actionable feedback. Never expose PII in logs or outputs."
    )
    USER_PROMPT_TEMPLATE: str = (
        "Please provide the email data in Graph JSON or MSG To Email JSON format. Specify any custom validation configuration if required."
    )
    LLM_RETRIES: int = 2
    LLM_RETRY_BACKOFF: float = 1.5  # seconds
    MAX_INPUT_LENGTH: int = 50000
    LOG_LEVEL: str = "INFO"

    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")

Config.validate()

# ------------------- Utility Classes -------------------

class PIIRedactor:
    """Redacts PII from text using regex patterns."""
    EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
    PHONE_PATTERN = re.compile(r'\b(?:\+?(\d{1,3}))?[-. (]*(\d{3})[-. )]*(\d{3})[-. ]*(\d{4})\b')
    GUID_PATTERN = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
    # Add more patterns as needed

    @classmethod
    def redact(cls, text: str) -> str:
        """Redact PII from the given text."""
        if not isinstance(text, str):
            return text
        text = cls.EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
        text = cls.PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
        text = cls.GUID_PATTERN.sub("[REDACTED_GUID]", text)
        return text

class Logger:
    """Logging utility with PII redaction."""
    def __init__(self):
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level=Config.LOG_LEVEL)
        self.redactor = PIIRedactor

    def log_event(self, event: str):
        """Log an event with PII redacted."""
        try:
            redacted = self.redactor.redact(event)
            logger.info(redacted)
        except Exception as e:
            logger.error(f"Logging error: {str(e)}")

    def log_error(self, error: str):
        """Log an error with PII redacted."""
        try:
            redacted = self.redactor.redact(error)
            logger.error(redacted)
        except Exception as e:
            logger.error(f"Logging error: {str(e)}")

# ------------------- Input Validation Models -------------------

class EmailInputModel(BaseModel):
    """Pydantic model for incoming email input."""
    email_input: Union[dict, str] = Field(..., description="Email data in Graph JSON or MSG To Email JSON format.")
    validation_config_path: Optional[str] = Field(None, description="Path to validation config JSON file.")

    @field_validator('email_input')
    @classmethod
    def validate_email_input(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Email input is empty.")
            if len(v) > Config.MAX_INPUT_LENGTH:
                raise ValueError(f"Email input exceeds {Config.MAX_INPUT_LENGTH} characters.")
            try:
                v_json = json.loads(v)
            except Exception as e:
                raise ValueError(f"Email input is not valid JSON: {str(e)}")
            return v_json
        elif isinstance(v, dict):
            if not v:
                raise ValueError("Email input dictionary is empty.")
            return v
        else:
            raise ValueError("Email input must be a JSON string or dictionary.")

    @field_validator('validation_config_path')
    @classmethod
    def validate_config_path(cls, v):
        if v is not None and not isinstance(v, str):
            raise ValueError("validation_config_path must be a string or None.")
        return v

# ------------------- Core Service Classes -------------------

class InputNormalizer:
    """Normalizes email input to internal schema and generates GUID."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def normalize(self, input_data: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Normalize input and generate GUID."""
        try:
            # Minimal normalization: ensure required fields, flatten keys, generate GUID
            normalized = dict(input_data)
            guid = str(uuid.uuid4())
            normalized['guid'] = guid
            # Example: flatten "to Recipients" to "to_recipients"
            if "to Recipients" in normalized:
                normalized["to_recipients"] = normalized.pop("to Recipients")
            if "cc Recipients" in normalized:
                normalized["cc_recipients"] = normalized.pop("cc Recipients")
            # Add more normalization as needed
            self.logger.log_event(f"Input normalized with GUID {guid}")
            return normalized, None
        except Exception as e:
            self.logger.log_error(f"Input normalization failed: {str(e)}")
            return None, f"Input normalization error: {str(e)}"

class ConfigLoader:
    """Loads and validates schema configuration."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def load_config(self, config_path: Optional[str]) -> Tuple[Optional[dict], Optional[str]]:
        """Load and validate schema config."""
        try:
            if not config_path:
                config_path = "validation_config.json"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            # Validate config structure (must have 'schema' key)
            if "schema" not in config:
                raise ValueError("Config missing 'schema' key.")
            self.logger.log_event(f"Config loaded from {config_path}")
            return config["schema"], None
        except Exception as e:
            self.logger.log_error(f"Config loading failed: {str(e)}")
            return None, f"Config loading error: {str(e)}"

class AIExtractor:
    """Extracts structured data from normalized email using LLM and schema."""
    def __init__(self, logger: Logger):
        self.logger = logger
        self.client = openai.AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def extract(self, email_json: dict, schema: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Extract structured data using LLM."""
        prompt = (
            f"{Config.SYSTEM_PROMPT}\n"
            f"Schema:\n{json.dumps(schema, indent=2)}\n"
            f"Email JSON:\n{json.dumps(email_json, indent=2)}\n"
            "Extract the structured data as per the schema. "
            "If a field is missing, leave it null. Do not fabricate data. "
            "Return only a valid JSON object matching the schema."
        )
        retries = 0
        last_error = None
        while retries <= Config.LLM_RETRIES:
            try:
                response = await self.client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": Config.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=Config.LLM_TEMPERATURE,
                    max_tokens=Config.LLM_MAX_TOKENS
                )
                content = response.choices[0].message.content
                # Try to parse the response as JSON
                try:
                    extracted = json.loads(content)
                    self.logger.log_event("LLM extraction successful.")
                    return extracted, None
                except Exception as e:
                    self.logger.log_error(f"LLM output not valid JSON: {str(e)} | Output: {content[:200]}")
                    return None, f"LLM output not valid JSON: {str(e)}"
            except Exception as e:
                last_error = str(e)
                self.logger.log_error(f"LLM extraction failed (attempt {retries+1}): {last_error}")
                await asyncio.sleep(Config.LLM_RETRY_BACKOFF * (2 ** retries))
                retries += 1
        return None, f"LLM extraction error after {Config.LLM_RETRIES+1} attempts: {last_error}"

class TypeFormatter:
    """Formats extracted values to schema types, handles list fields."""
    def __init__(self, logger: Logger):
        self.logger = logger

    def format_types(self, extracted_data: dict, schema: dict) -> Tuple[Optional[dict], Optional[str]]:
        """Format extracted data to schema types."""
        try:
            formatted = {}
            errors = []
            for field, field_schema in schema.items():
                value = extracted_data.get(field)
                expected_type = field_schema.get("type")
                if value is None:
                    formatted[field] = None
                    continue
                try:
                    if expected_type == "string":
                        formatted[field] = str(value)
                    elif expected_type == "integer":
                        formatted[field] = int(value)
                    elif expected_type == "number":
                        formatted[field] = float(value)
                    elif expected_type == "boolean":
                        if isinstance(value, bool):
                            formatted[field] = value
                        elif isinstance(value, str):
                            formatted[field] = value.lower() in ("true", "1", "yes")
                        else:
                            formatted[field] = bool(value)
                    elif expected_type == "array":
                        # Try to parse as Python list if string
                        if isinstance(value, str):
                            try:
                                formatted[field] = ast.literal_eval(value)
                                if not isinstance(formatted[field], list):
                                    raise ValueError("Not a list after eval")
                            except Exception:
                                formatted[field] = [value]
                        elif isinstance(value, list):
                            formatted[field] = value
                        else:
                            formatted[field] = [value]
                    else:
                        formatted[field] = value
                except Exception as e:
                    errors.append(f"Type formatting error for field '{field}': {str(e)}")
                    formatted[field] = None
                    
            if errors:
                self.logger.log_error("; ".join(errors))
                return formatted, "; ".join(errors)
            self.logger.log_event("Type formatting successful.")
            return formatted, None
        except Exception as e:
            self.logger.log_error(f"Type formatting failed: {str(e)}")
            return None, f"Type formatting error: {str(e)}"

class ResponseBuilder:
    """Assembles final structured response, aggregates errors, ensures output format."""
    def __init__(self, logger: Logger, redactor: PIIRedactor):
        self.logger = logger
        self.redactor = redactor

    def build_response(self, stage_outputs: dict, errors: List[str]) -> dict:
        """Build the final response, aggregate errors, redact PII."""
        response = {
            "success": len(errors) == 0,
            "data": stage_outputs.get("formatted_data"),
            "guid": stage_outputs.get("normalized_email", {}).get("guid"),
            "errors": errors if errors else None
        }
        # Redact PII in data and errors
        if response["data"]:
            response["data"] = self._redact_dict(response["data"])
        if response["errors"]:
            response["errors"] = [self.redactor.redact(e) for e in response["errors"]]
        self.logger.log_event(f"Response built for GUID {response.get('guid')}")
        return response

    def _redact_dict(self, d: dict) -> dict:
        """Recursively redact PII in dict values."""
        if not isinstance(d, dict):
            return d
        redacted = {}
        for k, v in d.items():
            if isinstance(v, str):
                redacted[k] = self.redactor.redact(v)
            elif isinstance(v, dict):
                redacted[k] = self._redact_dict(v)
            elif isinstance(v, list):
                redacted[k] = [self.redactor.redact(str(i)) if isinstance(i, str) else i for i in v]
            else:
                redacted[k] = v
        return redacted

# ------------------- Main Agent Class -------------------

class EmailParserAgent:
    """Main agent orchestrating the email parsing pipeline."""
    def __init__(self):
        self.logger = Logger()
        self.redactor = PIIRedactor
        self.input_normalizer = InputNormalizer(self.logger)
        self.config_loader = ConfigLoader(self.logger)
        self.ai_extractor = AIExtractor(self.logger)
        self.type_formatter = TypeFormatter(self.logger)
        self.response_builder = ResponseBuilder(self.logger, self.redactor)

    async def process_email(self, email_input: dict, validation_config_path: Optional[str]) -> dict:
        """Main pipeline method to process an email through all stages."""
        context = {}
        errors = []

        # Input Normalization Layer
        normalized_email, err = self.input_normalizer.normalize(email_input)
        if err:
            errors.append(err)
            return self.response_builder.build_response({"normalized_email": None}, errors)
        context["normalized_email"] = normalized_email

        # Configuration Management Layer
        schema, err = self.config_loader.load_config(validation_config_path)
        if err:
            errors.append(err)
            return self.response_builder.build_response(context, errors)
        context["schema"] = schema

        # AI Extraction Layer
        extracted_data, err = await self.ai_extractor.extract(normalized_email, schema)
        if err:
            errors.append(err)
            return self.response_builder.build_response(context, errors)
        context["extracted_data"] = extracted_data

        # Type Formatting Layer
        formatted_data, err = self.type_formatter.format_types(extracted_data, schema)
        if err:
            errors.append(err)
        context["formatted_data"] = formatted_data

        # Response Assembly Layer
        response = self.response_builder.build_response(context, errors)
        return response

# ------------------- FastAPI App & Endpoints -------------------

app = FastAPI(
    title="Email Parser Agent for Data Analytics",
    description="LLM-powered email parser for data analytics. Normalizes, validates, extracts, formats, and assembles structured email data.",
    version="1.0.0"
)

# CORS middleware for API usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = EmailParserAgent()

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "InputValidationError",
            "message": "Input validation failed.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted (use double quotes, check commas).",
                "Check that all required fields are present and not empty.",
                "If sending large text, ensure it is under 50,000 characters."
            ]
        }
    )

@app.exception_handler(json.decoder.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error_type": "MalformedJSON",
            "message": f"Malformed JSON: {str(exc)}",
            "tips": [
                "Check for missing or extra commas, brackets, or quotes.",
                "Use a JSON validator before submitting.",
                "Ensure your content is UTF-8 encoded."
            ]
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": f"An unexpected error occurred: {str(exc)}",
            "tips": [
                "Try again later.",
                "If the problem persists, contact support with the error details."
            ]
        }
    )

@app.post("/parse_email", response_model=dict)
async def parse_email(input_model: EmailInputModel):
    """
    Parse an email input (Graph JSON or MSG To Email JSON) and return structured data.
    """
    try:
        # Input validation and sanitization handled by Pydantic
        email_input = input_model.email_input
        validation_config_path = input_model.validation_config_path
        response = await agent.process_email(email_input, validation_config_path)
        return response
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "success": False,
                "error_type": "InputValidationError",
                "message": str(ve),
                "tips": [
                    "Ensure your JSON is properly formatted.",
                    "Check that all required fields are present and not empty."
                ]
            }
        )
    except Exception as e:
        agent.logger.log_error(f"API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_type": "InternalServerError",
                "message": str(e),
                "tips": [
                    "Try again later.",
                    "If the problem persists, contact support."
                ]
            }
        )

@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint."""
    return {"success": True, "status": "ok"}

# ------------------- Main Execution Block -------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Email Parser Agent for Data Analytics...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
