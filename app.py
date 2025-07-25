import streamlit as st
import pickle
from resume_parser import analyze_resume

st.set_page_config(page_title="Resume Analyzer AI", layout="centered")

model_data = pickle.load(open("model.pkl", "rb"))
important_skills = model_data["important_skills"]

st.title("🤖 AI Resume Analyzer & Job Recommender")
st.write("Upload your resume and get smart AI-powered job insights.")

uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("📄 Optional: Paste Job Description for ATS Check")

if uploaded_file:
    result = analyze_resume(uploaded_file, job_desc)

    st.subheader("✅ Extracted Skills")
    st.write(", ".join(result["skills"]) or "No skills found.")

    st.subheader("🎯 Top 3 Job Role Predictions")
    for role, prob in result["top_roles"]:
        st.write(f"**{role}** — {prob}% match")

    st.subheader("💼 Best Fit Role")
    st.success(result["best_role"])

    st.subheader("📊 Skill Match % for Best Role")
    st.progress(result["match_percent"] / 100)
    st.write(f"{result['match_percent']}% match with ideal {result['best_role']} skill set.")

    st.subheader("⭐ Resume Quality Score")
    st.write(f"{result['quality_score']} / 100")

    if result["ats_score"] is not None:
        st.subheader("🧠 ATS Keyword Match Score")
        st.write(f"{result['ats_score']}% match with job description")
