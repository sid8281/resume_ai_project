import pickle
import re
import PyPDF2
import math
model_data = pickle.load(open("model.pkl", "rb"))

clf = model_data["clf"]
vectorizer = model_data["vectorizer"]
all_skills = model_data["all_skills"]
important_skills = model_data["important_skills"]

def extract_text_from_pdf(file_obj):
    text = ""
    reader = PyPDF2.PdfReader(file_obj)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return text.lower()

def extract_skills(resume_text):
    resume_words = re.findall(r"\b[a-zA-Z0-9.+#]+\b", resume_text.lower())
    extracted = [skill for skill in all_skills if skill in resume_text]
    return list(set(extracted))

def predict_job_roles(skills):
    skill_text = " ".join(skills)
    vector = vectorizer.transform([skill_text])
    probas = clf.predict_proba(vector)[0]
    top_indices = probas.argsort()[-3:][::-1]
    labels = clf.classes_
    top_roles = [(labels[i], round(probas[i] * 100, 2)) for i in top_indices]
    return top_roles

def calculate_skill_match(extracted, role):
    expected = important_skills.get(role, [])[0].split()
    matched = [skill for skill in extracted if skill in expected]
    return int((len(matched) / len(expected)) * 100)

def rate_resume_quality(resume_text, skills):
    word_count = len(resume_text.split())
    skill_count = len(skills)
    quality = min(100, int((word_count / 500) * 50 + skill_count))
    return quality

def ats_match_score(extracted_skills, job_keywords):
    job_keywords_set = set(job_keywords.lower().split())
    matched = [kw for kw in extracted_skills if kw in job_keywords_set]
    return int((len(matched) / len(job_keywords_set)) * 100)

def analyze_resume(file_obj, job_desc_text=None):
    resume_text = extract_text_from_pdf(file_obj)
    skills = extract_skills(resume_text)
    top_roles = predict_job_roles(skills)
    best_role = top_roles[0][0]
    match_percent = calculate_skill_match(skills, best_role)
    quality_score = rate_resume_quality(resume_text, skills)
    ats_score = ats_match_score(skills, job_desc_text) if job_desc_text else None

    return {
        "skills": skills,
        "top_roles": top_roles,
        "best_role": best_role,
        "match_percent": match_percent,
        "quality_score": quality_score,
        "ats_score": ats_score
    }
