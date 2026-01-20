from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
