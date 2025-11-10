# my-brokers Setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 

# commands
ruff format
ruff check, and check --fix
python main.py

# Check
langChain посмотреть

# IMPORTANT!!!! FOR Chat SST downgrade transformers
pip unintall transformers
pip install transformers==4.53.2



python agent.py
# или со своим промптом:
python agent.py "Add 2 and 3 with the add tool, then call get_time."