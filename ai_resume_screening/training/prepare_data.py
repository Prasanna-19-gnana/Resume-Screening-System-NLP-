import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAW_DIR = os.path.join(BASE_DIR, "ai_resume_screening", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "ai_resume_screening", "data", "processed")

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalize_classification_data():
    """ Load Kaggle 'Resume.csv' and normalize """
    path = os.path.join(BASE_DIR, "Resume and Job Description Dataset", "Resume.csv")
    out_path = os.path.join(PROCESSED_DIR, "normalized_classification.csv")
    
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}. Please place 'Resume.csv' here.")
        return
        
    logger.info("Normalizing Resume Classification Dataset...")
    try:
        df = pd.read_csv(path)
        # Expected: Resume_str, Category
        df_norm = pd.DataFrame()
        
        if 'Resume_str' in df.columns and 'Category' in df.columns:
            df_norm['resume_text'] = df['Resume_str'].fillna('')
            df_norm['resume_role'] = df['Category'].fillna('Unknown')
        else:
            logger.error(f"Unexpected columns in Resume.csv: {df.columns}. Attempting fallback.")
            return

        df_norm = df_norm[df_norm['resume_text'].str.strip() != '']
        df_norm.to_csv(out_path, index=False)
        logger.info(f"Saved normalized classification data: {len(df_norm)} rows -> {out_path}")
    except Exception as e:
        logger.error(f"Error normalizing classification data: {e}")

def normalize_jd_data():
    """ Load Kaggle 'Job Title and Job Description Dataset.csv' """
    path = os.path.join(BASE_DIR, "Job Title and Job Description Dataset.csv")
    out_path = os.path.join(PROCESSED_DIR, "normalized_jd.csv")
    
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return
        
    logger.info("Normalizing Job Title and Job Description Dataset...")
    try:
        df = pd.read_csv(path)
        df_norm = pd.DataFrame()
        
        # Expected: Job Title, Job Description
        title_col = 'Job Title' if 'Job Title' in df.columns else df.columns[1]
        desc_col = 'Job Description' if 'Job Description' in df.columns else df.columns[2]
        
        df_norm['job_title'] = df[title_col].fillna('')
        df_norm['job_description_text'] = df[desc_col].fillna('')
        df_norm['extracted_requirements'] = ''
        
        df_norm = df_norm[df_norm['job_description_text'].str.strip() != '']
        df_norm.to_csv(out_path, index=False)
        logger.info(f"Saved normalized JD data: {len(df_norm)} rows -> {out_path}")
    except Exception as e:
        logger.error(f"Error normalizing JD data: {e}")

def normalize_match_data():
    """ Load Kaggle 'Resume Classification Dataset.csv' and match it """
    path = os.path.join(BASE_DIR, "Resume Classification Dataset.csv")
    out_path = os.path.join(PROCESSED_DIR, "normalized_match.csv")
    
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return
        
    logger.info("Normalizing Resume-JD Match Dataset...")
    try:
        df = pd.read_csv(path)
        df_norm = pd.DataFrame()
        
        # Merge relevant fields to create 'resume_text'
        # e.g., skills + career_objective + degree_names + responsibilities
        cols = ['skills', 'career_objective', 'degree_names', 'responsibilities']
        df['combined_resume'] = df[[c for c in cols if c in df.columns]].fillna('').agg(' '.join, axis=1)
        
        df_norm['resume_text'] = df['combined_resume'].fillna('')
        
        # JD fields
        df_norm['job_description_text'] = df.get('skills_required', pd.Series([''] * len(df))).fillna('') + " " + df.get('responsibilities.1', pd.Series([''] * len(df))).fillna('')
        df_norm['job_role'] = df.get('job_position_name', pd.Series(['Unknown'] * len(df))).fillna('Unknown')
        
        # Scores
        if 'matched_score' in df.columns:
            df_norm['score'] = df['matched_score'].fillna(0)
        else:
            df_norm['score'] = 0
            
        df_norm = df_norm[df_norm['resume_text'].str.strip() != '']
        df_norm.to_csv(out_path, index=False)
        logger.info(f"Saved normalized match data: {len(df_norm)} rows -> {out_path}")
    except Exception as e:
        logger.error(f"Error normalizing match data: {e}")

if __name__ == "__main__":
    logger.info("=== STARTING DATA PREPARATION PIPELINE ===")
    normalize_classification_data()
    normalize_jd_data()
    normalize_match_data()
    logger.info("=== DATA PREPARATION COMPLETE ===")
