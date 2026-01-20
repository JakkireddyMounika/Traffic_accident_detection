from openai import OpenAI
import os

# 🔹 Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔹 LLM function
def llm_generate_sos(accident_data):
    prompt = f"""
    Generate a professional emergency SOS message for a traffic accident.

    Details:
    Camera ID: {accident_data['camera_id']}
    Time: {accident_data['time']}
    Vehicles Involved: {accident_data['vehicles_involved']}
    Severity: {accident_data['severity']}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content


# 🔹 TEST DATA (PASTE YOUR CODE HERE ⬇️)
accident_data = {
    "camera_id": "CAMERA_01",
    "time": "07-01-2026 11:30:15",
    "vehicles_involved": [2, 5],
    "severity": "HIGH"
}

# 🔹 CALL LLM
sos_message = llm_generate_sos(accident_data)
print(sos_message)
