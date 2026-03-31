"""
SKILL ONTOLOGY: Hierarchical skill mapping for partial matching
Parent skills → Child skills (more specific)

Example:
- "machine learning" (parent) → ["scikit-learn", "xgboost", "classification"] (children)
- If candidate has parent but not child, give partial credit (0.5)
- If candidate has both or exact match, full credit (1.0)
"""

SKILL_ONTOLOGY = {
    # ===============================================
    # MACHINE LEARNING & AI
    # ===============================================
    "machine learning": {
        "children": [
            "scikit-learn",
            "sklearn",
            "xgboost",
            "random forest",
            "regression",
            "classification",
            "supervised learning",
            "unsupervised learning"
        ],
        "weight": 1.0  # Full weight parent skill
    },
    
    "deep learning": {
        "children": [
            "tensorflow",
            "pytorch",
            "keras",
            "neural networks",
            "convolutional neural network",
            "cnn",
            "recurrent neural network",
            "rnn",
            "lstm"
        ],
        "weight": 1.0
    },
    
    "nlp": {
        "children": [
            "natural language processing",
            "transformers",
            "bert",
            "gpt",
            "text classification",
            "sentiment analysis",
            "named entity recognition",
            "ner",
            "chatbot",
            "spacy",
            "huggingface"
        ],
        "weight": 1.0
    },
    
    "computer vision": {
        "children": [
            "image processing",
            "object detection",
            "segmentation",
            "yolo",
            "opencv",
            "cv2",
            "image classification"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # DATA SCIENCE & ANALYSIS
    # ===============================================
    "data science": {
        "children": [
            "machine learning",
            "deep learning",
            "statistics",
            "statistical modeling",
            "data analysis",
            "predictive modeling",
            "data mining"
        ],
        "weight": 1.0
    },
    
    "data analysis": {
        "children": [
            "pandas",
            "numpy",
            "data preprocessing",
            "data cleaning",
            "exploratory data analysis",
            "eda",
            "data wrangling",
            "feature engineering"
        ],
        "weight": 1.0
    },
    
    "data visualization": {
        "children": [
            "matplotlib",
            "seaborn",
            "plotly",
            "tableau",
            "power bi",
            "powerbi",
            "grafana",
            "kibana",
            "d3.js",
            "ggplot"
        ],
        "weight": 1.0
    },
    
    "big data": {
        "children": [
            "spark",
            "hadoop",
            "kafka",
            "distributed computing",
            "mapreduce",
            "hive",
            "presto",
            "impala"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # DATABASES & DATA STORAGE
    # ===============================================
    "sql": {
        "children": [
            "sql",
            "query",
            "joins",
            "aggregation",
            "stored procedures",
            "sql optimization"
        ],
        "weight": 1.0
    },
    
    "relational databases": {
        "children": [
            "postgresql",
            "postgres",
            "mysql",
            "oracle",
            "mariadb",
            "sql server",
            "sqlite"
        ],
        "weight": 1.0
    },
    
    "nosql databases": {
        "children": [
            "mongodb",
            "cassandra",
            "dynamodb",
            "couchdb",
            "redis",
            "elasticsearch",
            "elastic"
        ],
        "weight": 1.0
    },
    
    "data warehousing": {
        "children": [
            "snowflake",
            "redshift",
            "bigquery",
            "data warehouse",
            "etl",
            "dbt",
            "airflow",
            "data pipeline"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # PROGRAMMING LANGUAGES
    # ===============================================
    "python": {
        "children": [
            "python",
            "py",
            "python3"
        ],
        "weight": 1.0
    },
    
    "java": {
        "children": [
            "java",
            "spring",
            "hibernate"
        ],
        "weight": 1.0
    },
    
    "javascript": {
        "children": [
            "javascript",
            "js",
            "node.js",
            "nodejs",
            "typescript",
            "ts"
        ],
        "weight": 1.0
    },
    
    "scala": {
        "children": [
            "scala"
        ],
        "weight": 1.0
    },
    
    "r": {
        "children": [
            "r",
            "rstudio",
            "tidyverse",
            "ggplot2"
        ],
        "weight": 1.0
    },
    
    "c++": {
        "children": [
            "c++",
            "cpp"
        ],
        "weight": 1.0
    },
    
    "go": {
        "children": [
            "go",
            "golang"
        ],
        "weight": 1.0
    },
    
    "rust": {
        "children": [
            "rust"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # WEB FRAMEWORKS & DEVELOPMENT
    # ===============================================
    "frontend development": {
        "children": [
            "react",
            "reactjs",
            "vue",
            "vuejs",
            "angular",
            "html",
            "css",
            "javascript",
            "typescript",
            "webpack",
            "babel"
        ],
        "weight": 1.0
    },
    
    "react": {
        "children": [
            "react",
            "reactjs",
            "jsx",
            "redux",
            "react router"
        ],
        "weight": 1.0
    },
    
    "backend development": {
        "children": [
            "django",
            "flask",
            "fastapi",
            "node.js",
            "express",
            "spring",
            "java",
            "python",
            "rest api",
            "graphql"
        ],
        "weight": 1.0
    },
    
    "django": {
        "children": [
            "django",
            "django rest framework",
            "drf",
            "celery"
        ],
        "weight": 1.0
    },
    
    "fastapi": {
        "children": [
            "fastapi",
            "async",
            "pydantic"
        ],
        "weight": 1.0
    },
    
    "express": {
        "children": [
            "express",
            "express.js",
            "nodejs",
            "node.js"
        ],
        "weight": 1.0
    },
    
    "full stack development": {
        "children": [
            "react",
            "node.js",
            "express",
            "mongodb",
            "django",
            "fastapi",
            "frontend",
            "backend"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # DEVOPS & DEPLOYMENT
    # ===============================================
    "devops": {
        "children": [
            "docker",
            "kubernetes",
            "jenkins",
            "ci/cd",
            "nginx",
            "terraform",
            "ansible"
        ],
        "weight": 1.0
    },
    
    "docker": {
        "children": [
            "docker",
            "containerization",
            "docker compose"
        ],
        "weight": 1.0
    },
    
    "kubernetes": {
        "children": [
            "kubernetes",
            "k8s",
            "helm",
            "kubectl"
        ],
        "weight": 1.0
    },
    
    "cloud computing": {
        "children": [
            "aws",
            "azure",
            "gcp",
            "cloud",
            "ec2",
            "s3",
            "lambda",
            "serverless"
        ],
        "weight": 1.0
    },
    
    "aws": {
        "children": [
            "aws",
            "amazon",
            "ec2",
            "s3",
            "rds",
            "lambda",
            "sqs",
            "sns",
            "cloudwatch"
        ],
        "weight": 1.0
    },
    
    "azure": {
        "children": [
            "azure",
            "azure devops",
            "cosmos db",
            "azure functions"
        ],
        "weight": 1.0
    },
    
    "gcp": {
        "children": [
            "gcp",
            "google cloud",
            "bigquery",
            "dataflow",
            "cloud functions"
        ],
        "weight": 1.0
    },
    
    "ci/cd": {
        "children": [
            "continuous integration",
            "continuous deployment",
            "jenkins",
            "gitlab",
            "github",
            "github actions",
            "circleci",
            "travis"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # VERSION CONTROL & TOOLS
    # ===============================================
    "git": {
        "children": [
            "git",
            "github",
            "github actions",
            "gitlab",
            "bitbucket",
            "version control"
        ],
        "weight": 1.0
    },
    
    "linux": {
        "children": [
            "linux",
            "bash",
            "shell",
            "command line",
            "terminal"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # METHODOLOGY & PRACTICES
    # ===============================================
    "agile": {
        "children": [
            "agile",
            "scrum",
            "kanban",
            "sprint",
            "user story"
        ],
        "weight": 1.0
    },
    
    "testing": {
        "children": [
            "unit testing",
            "pytest",
            "junit",
            "test driven development",
            "tdd",
            "integration testing",
            "e2e testing"
        ],
        "weight": 1.0
    },
    
    # ===============================================
    # SPECIALIZED DOMAINS
    # ===============================================
    "time series": {
        "children": [
            "time series analysis",
            "forecasting",
            "arima",
            "lstm",
            "prophet"
        ],
        "weight": 1.0
    },
    
    "recommendation systems": {
        "children": [
            "collaborative filtering",
            "content based filtering",
            "matrix factorization",
            "embeddings",
            "recommendation algorithm"
        ],
        "weight": 1.0
    },
    
    "reinforcement learning": {
        "children": [
            "q learning",
            "policy gradient",
            "actor critic",
            "markov decision process"
        ],
        "weight": 1.0
    }
}


# Build reverse lookup: child → parent(s)
CHILD_TO_PARENTS = {}
for parent_skill, skill_data in SKILL_ONTOLOGY.items():
    for child_skill in skill_data["children"]:
        if child_skill not in CHILD_TO_PARENTS:
            CHILD_TO_PARENTS[child_skill] = []
        CHILD_TO_PARENTS[child_skill].append(parent_skill)


def get_parents_of_skill(skill: str) -> list:
    """Get all parent skills that contain this skill as a child"""
    return CHILD_TO_PARENTS.get(skill.lower(), [])


def get_children_of_skill(skill: str) -> list:
    """Get all child skills under this parent skill"""
    skill_lower = skill.lower()
    if skill_lower in SKILL_ONTOLOGY:
        return SKILL_ONTOLOGY[skill_lower]["children"]
    return []


def is_parent_skill(skill: str) -> bool:
    """Check if skill is a parent in the ontology"""
    return skill.lower() in SKILL_ONTOLOGY


def is_child_skill(skill: str) -> bool:
    """Check if skill is a child in the ontology"""
    return skill.lower() in CHILD_TO_PARENTS


class SkillOntology:
    """Wrapper class for skill ontology operations"""
    
    def __init__(self):
        self.ontology = SKILL_ONTOLOGY
        self.child_to_parents = CHILD_TO_PARENTS
    
    def get_parents_of_skill(self, skill: str) -> list:
        """Get all parent skills"""
        return get_parents_of_skill(skill)
    
    def get_children_of_skill(self, skill: str) -> list:
        """Get all child skills"""
        return get_children_of_skill(skill)
    
    def is_parent_skill(self, skill: str) -> bool:
        """Check if skill is a parent"""
        return is_parent_skill(skill)
    
    def is_child_skill(self, skill: str) -> bool:
        """Check if skill is a child"""
        return is_child_skill(skill)
