import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

job_roles = {
    "Data Scientist": ["python r sql pandas numpy matplotlib seaborn scikit-learn tensorflow keras pytorch nltk opencv"],
    "Data Analyst": ["excel sql tableau power bi python pandas numpy matplotlib seaborn"],
    "Backend Developer": ["java python node.js php c# django flask express.js spring boot ruby on rails"],
    "Frontend Developer": ["html css javascript react.js angular vue.js typescript bootstrap tailwind css figma"],
    "Full Stack Developer": ["html css javascript react.js node.js express.js mongodb python django"],
    "DevOps Engineer": ["aws azure gcp docker kubernetes jenkins terraform ansible linux bash git github"],
    "Cloud Engineer": ["aws azure gcp docker kubernetes terraform linux cloudformation load balancer"],
    "QA Engineer": ["selenium postman jmeter jest cypress mocha automation testing manual testing"],
    "Database Administrator": ["mysql postgresql oracle mongodb sqlite cassandra database tuning pl/sql"],
    "Mobile App Developer": ["android kotlin java flutter dart swift xcode react native ios"],
    "AI/ML Engineer": ["machine learning deep learning nlp tensorflow keras pytorch openai huggingface llms"],
    "Cybersecurity Specialist": ["network security penetration testing ethical hacking firewall nmap burpsuite siem kali linux"],
    "Blockchain Developer": ["solidity web3.js ethereum smart contracts truffle ganache hyperledger metamask"],
    "Game Developer": ["unity unreal engine c# c++ blender game physics shader programming"],
    "UI/UX Designer": ["figma adobe xd sketch invision wireframing prototyping user research"],
    "IT Support Specialist": ["troubleshooting windows linux networking vpn helpdesk ticketing system remote support"],
    "System Administrator": ["windows server linux bash powershell active directory virtualization vmware hyper-v"],
    "Product Manager": ["jira confluence agile scrum roadmap user stories stakeholder management"]
}

X = []
y = []

for role, skill_sets in job_roles.items():
    for skills in skill_sets:
        X.append(skills.lower())
        y.append(role)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_vectorized, y)

all_skills = sorted(set(" ".join(X).split()))

model_data = {
    "clf": clf,
    "vectorizer": vectorizer,
    "all_skills": all_skills,
    "important_skills": job_roles
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model trained and saved.")
