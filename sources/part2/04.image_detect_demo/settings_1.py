import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), 'settings_1.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("API_KEY")
CUSTOM_SEARCH_ENGINE = os.environ.get("CUSTOM_SEARCH_ENGINE")
