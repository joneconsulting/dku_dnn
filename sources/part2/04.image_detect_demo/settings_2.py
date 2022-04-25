import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), 'settings_2.env')
load_dotenv(dotenv_path)

CASCADE_FILE_PATH = os.environ.get("CASCADE_FILE_PATH")
