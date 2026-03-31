import streamlit as st
import requests
import pandas as pd
import os
from typing import List, Dict, Any

# Configuration: Read API URLs from Streamlit secrets or environment variables
# This allows the app to work both locally (localhost) and on Streamlit Cloud
API_SERVICE_URL = st.secrets.get("api_service_url", os.getenv("API_SERVICE_URL", "http://localhost:8001"))

ALLOWED_EXTENSIONS = ["pdf", "docx"]

st.set_page_config(
    page_title="NexGen Resume Screening",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS STYLING FOR PREMIUM UI
# ==========================================
st.markdown("""
<style>
    /* Premium dark mode inspired typography and spacing */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .badge-excellent { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;}
    .badge-good { background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%); color: white; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;}
    .badge-medium { background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%); color: white; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;}
    .badge-poor { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); color: white; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;}
    .badge-neutral { background: #6c757d; color: white; padding: 6px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.5px;}
    
    /* Metrics Customization */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #adb5bd;
    }
    
    /* Expander Cards */
    .streamlit-expanderHeader {
        font-size: 1.15rem;
        font-weight: 600;
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Header Container */
    .header-banner {
        background: linear-gradient(135deg, #1e2530 0%, #11151c 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 5px solid #00f2fe;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    h1, h2, h3 {  
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS (WITH BUG FIXES)
# ==========================================

def get_badge_html(fit_label: str) -> str:
    lbl = fit_label.upper()
    if "EXCELLENT" in lbl:
        return f"<span class='badge-excellent'>🌟 {lbl} FIT</span>"
    elif "GOOD" in lbl:
        return f"<span class='badge-good'>✅ {lbl} FIT</span>"
    elif "MEDIUM" in lbl:
        return f"<span class='badge-medium'>⚡ {lbl} FIT</span>"
    elif "POOR" in lbl:
        return f"<span class='badge-poor'>⚠️ {lbl} FIT</span>"
    else:
        return f"<span class='badge-neutral'>🤷 {lbl}</span>"

def send_request(job_role: str, job_description: str, resume_file, scorer_mode: str) -> Dict[str, Any]:
    api_url_upload = f"{API_SERVICE_URL}/upload-resume"
    api_url_match = f"{API_SERVICE_URL}/match"
    
    try:
        # Step 1: Upload and Parse Document into Structured JSON
        files = {"file": (resume_file.name, resume_file.getvalue(), resume_file.type)}
        res_upload = requests.post(api_url_upload, files=files)
        
        if res_upload.status_code != 200:
            return {"error": f"Upload/Parse Error {res_upload.status_code}: {res_upload.text}"}
            
        parsed_json = res_upload.json()
        
        # Step 2: Send Structured Payload to Intelligence Engine
        payload = {
            "parsed_resume": parsed_json["parsed_sections"],
            "job_role": job_role,
            "requirements": "", # Handled inherently via job description blob now
            "job_description": job_description, 
            "scorer_mode": scorer_mode
        }
        
        res_match = requests.post(api_url_match, json=payload)
        
        if res_match.status_code == 200:
            match_data = res_match.json()
            # UI compat payload mapping since new API schema is robust:
            match_data["candidate_name"] = resume_file.name
            match_data["filename"] = resume_file.name
            match_data["fit_label_prediction"] = match_data.get("predicted_resume_role", "Unknown")
            return match_data
            
        return {"error": f"Engine Error {res_match.status_code}: {res_match.text}"}
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {e}"}

def sort_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        results, 
        key=lambda x: float(x.get("final_score") or 0.0), 
        reverse=True
    )

def render_metric_bar(label: str, raw: float, weighted: float, max_weight: float):
    """Displays a modern progress bar style metric for sub-components"""
    raw_val = float(raw or 0.0)
    weighted_val = float(weighted or 0.0)
    clamped_raw = max(0.0, min(1.0, raw_val))
    points_earned = weighted_val * 100
    max_pts = float(max_weight) * 100
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{label}**")
        st.progress(clamped_raw)
    with col2:
        st.markdown(f"<div style='text-align: right; padding-top: 25px; font-weight:bold; color:#adb5bd;'>{points_earned:.1f} / {max_pts:.1f} pts</div>", unsafe_allow_html=True)

def render_candidate_card(result: Dict[str, Any], rank: int):
    # Safe extractions protecting against None
    candidate_name = result.get("candidate_name") or result.get("filename") or f"Candidate {rank}"
    final_score = float(result.get("final_score") or 0.0)
    scorer_mode = result.get("scorer_mode") or "baseline"
    fit_label = (result.get("fit_label_prediction") or "UNKNOWN").upper()
    ml_score = float(result.get("ml_score") or 0.0)
    baseline_score = float(result.get("baseline_score") or 0.0)
    
    title_suffix = f" | {final_score * 100:.1f} Pts"
    if fit_label != "UNKNOWN":
        title_suffix += f" • [{fit_label}]"
        
    expander_title = f"#{rank} — {candidate_name}{title_suffix}"
    
    with st.expander(expander_title, expanded=(rank == 1)):
        warnings = result.get("warnings") or []
        for w in warnings:
            st.warning(f"⚠️ **Notice:** {w}")

        # Top Section: Title and Metrics
        st.markdown(f"## {candidate_name}")
        if scorer_mode in ["compare", "ml"] and fit_label != "UNKNOWN":
            st.markdown(get_badge_html(fit_label), unsafe_allow_html=True)
            st.write("") # spacing
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Combined Score", f"{final_score * 100:.1f} / 100")
        
        # Pull new deep metrics directly from JSON instead of old ML/Baseline buckets
        r_align = float(result.get("role_alignment_score") or 0.0)
        sem_score = float(result.get("semantic_similarity_score") or 0.0)
        
        m2.metric("Overall Semantic Similarity", f"{sem_score * 100:.1f} / 100")
        m3.metric("Role Alignment Projection", f"{r_align * 100:.1f} / 100")
            
        st.divider()
        
        # Tabs for Deep Dive
        tab_match, tab_breakdown, tab_insights = st.tabs(["🎯 Skills & Requirements", "📊 Score Breakdown", "💡 AI Insights"])
        
        with tab_match:
            jd_failed = result.get("job_description_skill_extraction_failed", False)
            if jd_failed:
                st.error("System could not extract any standard skills from the provided Job Description. Using generic matching fallback.")
            else:
                matched = result.get("matched_skills") or []
                missing = result.get("missing_skills") or []
                
                col_m, col_u = st.columns(2)
                with col_m:
                    st.success(f"**✅ Matched Skills ({len(matched)})**")
                    if matched:
                        st.markdown(" ".join([f"`{m}`" for m in matched]))
                    else:
                        st.info("No matching skills found.")
                with col_u:
                    st.error(f"**❌ Missing Required Skills ({len(missing)})**")
                    if missing:
                        st.markdown(" ".join([f"`{m}`" for m in missing]))
                    else:
                        st.success("**🌟 Complete Match!** All required skills found.")

        with tab_breakdown:
            st.markdown("#### Baseline Logic Components (3-Layer Engine)")
            
            raw_sk_sim = result.get("skills_similarity", 0.0)
            raw_proj_sim = result.get("projects_similarity", 0.0)
            raw_exp_sim = result.get("experience_similarity", 0.0)
            
            raw_sem = result.get("semantic_similarity_score", 0.0)
            raw_sk = result.get("skill_match_score", 0.0)
            raw_role = result.get("role_alignment_score", 0.0)
            
            st.markdown("##### 1. Section-Aware Sentiment Alignment")
            render_metric_bar("📚 Context: Skills Vector", raw_sk_sim, raw_sk_sim * 0.45, 0.45)
            render_metric_bar("🚀 Context: Projects Vector", raw_proj_sim, raw_proj_sim * 0.35, 0.35)
            render_metric_bar("💼 Context: Experience Vector", raw_exp_sim, raw_exp_sim * 0.20, 0.20)

            st.markdown("##### 2. Final Component Formula Weights")
            render_metric_bar("🧠 Final Combined Semantic Score", raw_sem, raw_sem * 0.40, 0.40)
            render_metric_bar("🛠️ Strict Ontology Skill Match", raw_sk, raw_sk * 0.35, 0.35)
                
            render_metric_bar("🎯 NLP Role Alignment", raw_role, raw_role * 0.25, 0.25)

        with tab_insights:
            c_ins1, c_ins2 = st.columns(2)
            
            # Dynamically derive insights from exact numerical components since flat text was deprecated
            s_dyn = []
            w_dyn = []
            
            if raw_sk > 0.85: s_dyn.append("Exceptional explicit technological/skill fit.")
            elif raw_sk < 0.40: w_dyn.append("Lacking required core technical competencies.")
                
            if raw_sem > 0.65: s_dyn.append("Strong semantic overlap with job responsibilities.")
            elif raw_sem < 0.30: w_dyn.append("Low overall semantic similarity to the job description.")
                
            if raw_role > 0.80: s_dyn.append("Candidate's detected role directly aligns with the target.")
            elif raw_role < 0.35: w_dyn.append("Candidate background implies severe domain misalignment.")
                
            with c_ins1:
                st.markdown("#### ✅ Candidate Strengths")
                if not s_dyn:
                    st.markdown("_No specific outstanding strengths detected._")
                else:
                    for s in s_dyn:
                        st.markdown(f"- {s}")
            with c_ins2:
                st.markdown("#### ⚠️ Potential Weaknesses")
                if not w_dyn:
                    st.markdown("_No critical weak areas detected._")
                else:
                    for w in w_dyn:
                        st.markdown(f"- {w}")

# ==========================================
# MAIN DASHBOARD ENTRY
# ==========================================

st.markdown('''
<div class="header-banner">
    <h1 style="margin-bottom: 0;">✨ NexGen Intelligent Resume Screener</h1>
    <p style="color: #adb5bd; font-size: 1.1rem; margin-top: 10px;">
        Unleash the power of Natural Language Processing and Machine Learning to evaluate candidates against your exact requirements.
    </p>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Evaluation Parameters")
    
    scorer_mode = st.radio(
        "Scoring Engine Configuration",
        options=["compare", "ml", "baseline"],
        format_func=lambda x: {
            "compare": "Dual-Engine Compare (Recommended)",
            "ml": "Pure Machine Learning",
            "baseline": "Strict Rule-Based Heuristics"
        }[x]
    )
    st.markdown("---")
    
    job_role = st.text_input("Job Role / Title", placeholder="e.g. Senior Software Engineer")
    job_description = st.text_area("Job Profile / Description", height=280, placeholder="Paste job description or core requirements here (e.g., Lead Engineer proficient in React, AWS, Python...)")
    
    uploaded_files = st.file_uploader("Candidate Resumes", type=ALLOWED_EXTENSIONS, accept_multiple_files=True)
    st.caption(f"Accepted document types: {', '.join(ALLOWED_EXTENSIONS).upper()}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_screening = st.button("🚀 Execute Screening", type="primary", use_container_width=True)

if run_screening:
    if not job_role.strip():
        st.toast("⚠️ Error: Job role is required.")
        st.error("Please enter a job role.")
    elif not job_description.strip():
        st.toast("⚠️ Error: Job description is required.")
        st.error("Please enter a job description to screen against.")
    elif not uploaded_files:
        st.toast("⚠️ Error: No resumes uploaded.")
        st.error("Please upload at least one candidate resume.")
    else:
        results = []
        errors = []
        
        with st.spinner(f"Initiating {scorer_mode.upper()} screening pipelines for {len(uploaded_files)} candidate(s)..."):
            for resume_file in uploaded_files:
                ext = resume_file.name.split(".")[-1].lower()
                if ext not in ALLOWED_EXTENSIONS:
                    errors.append(f"**{resume_file.name}**: File type not supported.")
                    continue
                    
                response_data = send_request(job_role, job_description, resume_file, scorer_mode)
                if "error" in response_data:
                    errors.append(f"**{resume_file.name}**: {response_data['error']}")
                else:
                    results.append(response_data)
                    
        # Save to persist state!
        st.session_state.screening_results = results
        st.session_state.screening_errors = errors
        st.session_state.last_scorer_mode = scorer_mode

# RENDER PERSISTENT STATE
if "screening_results" in st.session_state:
    results = st.session_state.screening_results
    errors = st.session_state.screening_errors
    scorer_mode = st.session_state.last_scorer_mode
    
    # Render feedback
    if errors:
        for err in errors:
            st.error(err)
            
    if results:
        sorted_results = sort_results(results)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("## 📋 Candidates Overview")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Clear & Restart", type="secondary", use_container_width=True):
                del st.session_state["screening_results"]
                st.rerun()
        
        # Construct DataFrame carefully
        table_data = []
        for res in sorted_results:
            row = {
                "Name / File": res.get("candidate_name") or res.get("filename") or "Unknown",
                "Final Score": f"{float(res.get('final_score') or 0.0) * 100:.1f}"
            }
            
            row["AI Role Prediction"] = (res.get("fit_label_prediction") or "unknown").upper()
            row["Role Match Prob"] = f"{float(res.get('role_alignment_score') or 0.0) * 100:.1f}%"
                
            matched = res.get("matched_skills") or []
            missing = res.get("missing_skills") or []
            
            row["Skills Met"] = len(matched)
            row["Skills Missing"] = len(missing)
            
            table_data.append(row)
            
        df = pd.DataFrame(table_data)
        df.index = df.index + 1 
        
        st.dataframe(df, use_container_width=True)
        
        st.divider()
        
        st.markdown("## 🔍 Deep Dive Analysis")
        for rank, result in enumerate(sorted_results, 1):
            render_candidate_card(result, rank)
