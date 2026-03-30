import requests
import json
import io
from reportlab.pdfgen import canvas

def create_mock_pdf():
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer)
    c.drawString(100, 800, "John Doe - Senior Developer")
    c.drawString(100, 780, "Technical Skills:")
    c.drawString(100, 760, "Python, Machine Learning, JavaScript, Git")
    c.drawString(100, 740, "Personal Projects:")
    c.drawString(100, 720, "1. NLP chatbot project utilizing transformers and deep learning.")
    c.drawString(100, 700, "2. MERN project using REST APIs for asynchronous data handling.")
    c.drawString(100, 680, "Experience:")
    c.drawString(100, 660, "Software Engineer intern constructing robust enterprise applications.")
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer

def test_production_endpoints():
    print("="*50)
    print("🚀 NLP ENGINEER TEST: Section Aware Output")
    print("="*50)
    
    API_URL_UPLOAD = "http://localhost:8001/upload-resume"
    API_URL_MATCH = "http://localhost:8001/match"
    
    pdf_file = create_mock_pdf()
    files = {'file': ('test_resume.pdf', pdf_file, 'application/pdf')}
    res_upload = requests.post(API_URL_UPLOAD, files=files)
    
    parsed_json = res_upload.json()
    
    payload = {
      "parsed_resume": parsed_json["parsed_sections"],
      "job_role": "NLP Engineer",
      "requirements": "Python, NLP, Machine Learning",
      "job_description": "We need a skilled NLP engineer to design intelligent conversational logic using Python and ML."
    }
    
    res_match = requests.post(API_URL_MATCH, json=payload)
    match_data = res_match.json()
    
    print("\n--- RESULTS FOR: NLP ENGINEER ---")
    print(f"🎯 Final Multi-Layer Score:  {match_data['final_score']*100:.1f} / 100")
    print(f"🧠 Overall Semantic Score:   {match_data['semantic_similarity_score']}")
    print(f"   > Skills Similarity:      {match_data['skills_similarity']}")
    print(f"   > Projects Similarity:    {match_data['projects_similarity']}")
    print(f"   > Experience Similarity:  {match_data['experience_similarity']}")

    print("\n" + "="*50)
    print("🚀 DATA SCIENTIST TEST: Alternate Segment Decay")
    print("="*50)
    
    payload["job_role"] = "Data Scientist"
    payload["requirements"] = "Python, SQL, R, Pandas, Machine Learning, Deep Learning"
    payload["job_description"] = "Seeking a Data Scientist to build deep statistical distribution pipelines operating over immense arrays using SQL and Python."
    
    res_match_2 = requests.post(API_URL_MATCH, json=payload).json()
    print("\n--- RESULTS FOR: DATA SCIENTIST ---")
    print(f"🎯 Final Multi-Layer Score:  {res_match_2['final_score']*100:.1f} / 100")
    print(f"🧠 Overall Semantic Score:   {res_match_2['semantic_similarity_score']}")
    print(f"   > Skills Similarity:      {res_match_2['skills_similarity']}")
    print(f"   > Projects Similarity:    {res_match_2['projects_similarity']}")
    print(f"   > Experience Similarity:  {res_match_2['experience_similarity']}")


if __name__ == "__main__":
    test_production_endpoints()
