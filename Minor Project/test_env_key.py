import os

key = os.getenv("OPENAI_API_KEY")

if key:
    print("✅ API key loaded successfully")
else:
    print("❌ API key NOT found")
