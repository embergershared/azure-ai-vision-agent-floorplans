# Execute with: ".\Streamlit-start.ps1"
# cd "D:\GitHub\azure-ai-vision-agent-floorplans"
.venv\scripts\activate
python -m pip install -r requirements.txt
streamlit run ./frontend/app.py
