"""Central configuration. Reads from .env (see .env.example)."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "MediFlow Intelligence API"
    API_VERSION: str = "2.0.0"

    # --- Supabase ---------------------------------------------------------
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    PHARMACY_SCHEMA: str = "pharmacy_inventory"
    EQUIPMENT_SCHEMA: str = "equipmentsdeets"

    # --- Tenant -----------------------------------------------------------
    ORGANIZATION_ID: str = os.getenv(
        "ORGANIZATION_ID", "1be57784-7611-596e-af4e-e012a17d947a"
    )
    ORGANIZATION_NAME: str = os.getenv("ORGANIZATION_NAME", "NorthStar Health Network")
    CURRENCY: str = os.getenv("CURRENCY", "USD")

    # --- Auth -------------------------------------------------------------
    JWT_SECRET: str = os.getenv("JWT_SECRET")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

    # --- Azure OpenAI (gpt-4.1) ------------------------------------------
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")

    # --- Behaviour --------------------------------------------------------
    AI_ENABLED: bool = os.getenv("AI_ENABLED", "true").lower() == "true"
    AI_CACHE_MINUTES: int = int(os.getenv("AI_CACHE_MINUTES", "15"))
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,https://ubti-mediflow.web.app"
    ).split(",")

    @property
    def azure_configured(self) -> bool:
        return bool(self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_API_KEY)


settings = Settings()
