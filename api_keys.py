import os
import re
from dotenv import load_dotenv

class APIKey:
    """
    Manages reading and updating API keys in a .env file.

    Args:
        env_path (str): Full path to the .env file.
    """

    def __init__(self, env_path: str):
        self.env_path = env_path

    def read(self, key_name: str) -> str | None:
        """
        Reads an API key value from the .env file.

        Args:
            key_name (str): The variable name, e.g. "OPENAI_API_KEY".

        Returns:
            str: The key value, or None if not found.
        """
        load_dotenv(self.env_path)
        key_value = os.getenv(key_name)
        
        return key_value if key_value else None

    def update(self, service_name: str, new_key_value: str) -> None:
        """
        Updates or creates an API key in the .env file.

        Args:
            service_name (str): Service name, e.g. "OPENAI".
            new_key_value (str): The API key value to store.
        """
        var_name = f"{service_name.upper()}_API_KEY"

        try:
            with open(self.env_path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            content = ""

        if f"{var_name}=" in content:
            content = re.sub(rf"{var_name}=.*", f"{var_name}={new_key_value}", content)
            action = "Updated"
        else:
            if content and not content.endswith("\n"):
                content += "\n"
            content += f"{var_name}={new_key_value}\n"
            action = "Created"

        with open(self.env_path, "w") as f:
            f.write(content)

        print(f"{action} {var_name} in .env file.")

    def load_to_env(self, key_name: str) -> None:
        """
        Reads a key and sets it directly in os.environ.

        Args:
            key_name (str): The variable name, e.g. "OPENAI_API_KEY".
        """
        value = self.read(key_name)
        if value:
            os.environ[key_name] = value
        else:
            raise ValueError(f"{key_name} not found in {self.env_path}")