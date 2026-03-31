
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

class ConfigError(Exception):
    pass

class Config:
    """
    Configuration management for Email Parser Agent for Data Analytics.
    Handles environment variables, API keys, LLM config, domain settings, and error handling.
    """

    # --- API Key Management ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ConfigError("Missing required environment variable: OPENAI_API_KEY")

    # Internal Validation Config API (OAuth2 token, optional for local dev)
    VALIDATION_CONFIG_API_TOKEN = os.getenv("VALIDATION_CONFIG_API_TOKEN", None)

    # --- LLM Configuration ---
    LLM_PROVIDER = "openai"
    LLM_MODEL = "gpt-4o"
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 2048
    LLM_SYSTEM_PROMPT = (
        "You are a professional Email Parser Agent for data analytics. Your tasks are to normalize email input formats, "
        "load and apply validation schemas, extract structured data using AI, format types according to schema (with special handling for list fields), "
        "and build structured responses. At each stage, handle errors gracefully and provide clear, actionable feedback. Never expose PII in logs or outputs."
    )
    LLM_USER_PROMPT_TEMPLATE = (
        "Please provide the email data in Graph JSON or MSG To Email JSON format. Specify any custom validation configuration if required."
    )
    LLM_FEW_SHOT_EXAMPLES = [
        '{ "id": "123", "from": "alice@example.com", "to Recipients": ["bob@example.com"], "cc Recipients": [], "subject": "Invoice 456", "body": "Invoice number: 456\\nTotal amount: 1200\\nLocation: [\'NY\', \'CA\']", "has Attachments": true, "received Date Time": "2024-06-01T10:00:00Z" }',
        '{ "id": "789", "from": "carol@example.com", "to Recipients": ["dave@example.com"], "cc Recipients": [], "subject": "Invoice missing total", "body": "Invoice number: 789", "has Attachments": false, "received Date Time": "2024-06-02T12:00:00Z" }'
    ]

    # --- Domain-Specific Settings ---
    DOMAIN = "data_analytics"
    AGENT_NAME = "Email Parser Agent for Data Analytics"
    DEFAULT_VALIDATION_CONFIG_PATH = os.getenv("VALIDATION_CONFIG_PATH", "validation_config.json")
    MAX_EMAIL_INPUT_SIZE = int(os.getenv("MAX_EMAIL_INPUT_SIZE", "50000"))  # bytes/chars

    # --- API Requirements ---
    API_REQUIREMENTS = [
        {
            "name": "OpenAI API",
            "type": "external",
            "purpose": "LLM-powered extraction of structured data from email content.",
            "authentication": "API Key (managed via environment variable or vault)",
            "rate_limits": "As per OpenAI subscription (e.g., 60 RPM, 60,000 TPM)"
        },
        {
            "name": "Internal Validation Config API",
            "type": "internal",
            "purpose": "Fetch and update validation_config.json for schema definitions.",
            "authentication": "OAuth2 with RBAC",
            "rate_limits": "100 RPM"
        }
    ]

    # --- Validation and Error Handling ---
    @staticmethod
    def handle_missing_key(key_name):
        raise ConfigError(f"Missing required configuration: {key_name}")

    @staticmethod
    def get_validation_schema(config_path=None):
        """
        Load validation schema from file or raise error.
        """
        path = config_path or Config.DEFAULT_VALIDATION_CONFIG_PATH
        if not os.path.exists(path):
            raise ConfigError(f"Validation config file not found: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "schema" not in config:
                raise ConfigError("Validation config missing 'schema' key.")
            return config["schema"]
        except Exception as e:
            raise ConfigError(f"Error loading validation config: {str(e)}")

    # --- Default Values and Fallbacks ---
    @staticmethod
    def get_llm_config():
        return {
            "provider": Config.LLM_PROVIDER,
            "model": Config.LLM_MODEL,
            "temperature": Config.LLM_TEMPERATURE,
            "max_tokens": Config.LLM_MAX_TOKENS,
            "system_prompt": Config.LLM_SYSTEM_PROMPT,
            "user_prompt_template": Config.LLM_USER_PROMPT_TEMPLATE,
            "few_shot_examples": Config.LLM_FEW_SHOT_EXAMPLES
        }

    @staticmethod
    def redact_pii(text):
        """
        Basic PII redaction for logs/outputs.
        """
        import re
        if not isinstance(text, str):
            return text
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED_EMAIL]', text)
        text = re.sub(r'\b\d{10,}\b', '[REDACTED_NUMBER]', text)
        return text

    # --- Compliance and Security ---
    GDPR_COMPLIANT = True
    SOC2_COMPLIANT = True
    LOG_PII = False  # Never log PII

    # --- Performance and Scalability ---
    RESPONSE_TIME_MS = 1500
    MAX_CONCURRENT_REQUESTS = 50

    # --- Logging ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_AUDIT_RETENTION_DAYS = 90

    # --- Error Messages ---
    ERROR_MESSAGES = {
        "missing_api_key": "OpenAI API key is missing. Set OPENAI_API_KEY in your environment.",
        "missing_validation_config": "Validation config file is missing or invalid.",
        "invalid_email_input": "Email input is invalid or exceeds maximum allowed size.",
        "llm_error": "LLM extraction failed. Please try again or check your API usage.",
        "internal_error": "An internal error occurred. Please contact support."
    }

# Example usage:
# try:
#     schema = Config.get_validation_schema()
# except ConfigError as e:
#     print(str(e))

# llm_config = Config.get_llm_config()
