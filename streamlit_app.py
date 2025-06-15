import streamlit as st
from openai import OpenAI
import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Title and instructions
st.set_page_config(page_title="Career Path Chatbot", page_icon="💬")
st.title("🎓 Career Path Chatbot")
st.write(
    "Chat with this assistant to discover your career preferences and get personalized recommendations. "
    "Please enter your OpenAI API key below to get started."
)

# OpenAI API key input
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please enter your OpenAI API key to begin.", icon="🔐")
    st.stop()

# OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Define career paths and keywords
career_paths = {
    "STEM": ["AI", "robotics", "machine learning", "engineering", "coding", "physics", "mathematics"],
    "Arts": ["painting", "writing", "music", "design", "theatre", "film", "photography"],
    "Sports": ["football", "cricket", "basketball", "athletics", "fitness", "gym"],
    "Business": ["entrepreneurship", "marketing", "finance", "management", "sales"],
    "Health": ["biology", "medicine", "nursing", "mental health", "healthcare"],
    "Social Sciences": ["psychology", "history", "sociology", "philosophy", "politics"],
    "Education": ["teaching", "training", "mentoring", "academic research"],
    "Technology & IT": ["software", "cybersecurity", "cloud", "data science", "web development"]
}

# Create FAISS vectorstore for semantic search
def create_vectorstore():
    docs = []
    for path, keywords in career_paths.items():
        doc = Document(
            page_content=f"{path}: {' '.join(keywords)}",
            metadata={"path": path}
        )
        docs.append(doc)
    return FAISS.from_documents(docs, embeddings_model)

vectorstore = create_vectorstore()

# Preference extraction function using OpenAI
def extract_preferences(messages, client):
    extraction_prompt = [
        {"role": "system", "content": (
            "You are a helpful assistant that extracts user preferences for career guidance. "
            "From the conversation below, extract the user's preferences like interests, skills, career goals, "
            "preferred work environment, and any other relevant personal preferences. "
            "Format your output as a JSON dictionary with keys like 'interests', 'skills', 'goals', 'work_environment'."
        )},
        *messages
    ]
    extraction_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=extraction_prompt
    )
    return extraction_response.choices[0].message.content

# Semantic matching with LangChain + FAISS
def semantic_career_match(interests):
    results = []
    if not interests:
        return []
    for interest in interests:
        matches = vectorstore.similarity_search(interest, k=2)
        for match in matches:
            results.append(match.metadata["path"])
    return list(set(results))

# Hybrid recommendation: keyword + semantic + fallback
def hybrid_career_recommendation(extracted_prefs_json):
    try:
        prefs = json.loads(extracted_prefs_json)
        interests = prefs.get("interests", [])
        recommended_paths = {}

        # 1. Keyword match
        for path, keywords in career_paths.items():
            score = sum(1 for interest in interests if interest.lower() in [k.lower() for k in keywords])
            if score > 0:
                explanation = f"{path} is a good fit for you because you showed interest in topics like: {', '.join(set(interests) & set(keywords))}."
                recommended_paths[path] = explanation

        # 2. Semantic match
        if not recommended_paths:
            sem_matches = semantic_career_match(interests)
            for path in sem_matches:
                explanation = f"{path} could suit you based on your interests (semantic similarity)."
                recommended_paths[path] = explanation

        # 3. Fallback
        if not recommended_paths:
            recommended_paths["🔍 Clarification Needed"] = (
                "We couldn't match your interests. Please describe what excites you or what kind of work you enjoy."
            )

        return recommended_paths
    except Exception as e:
        return {"Error": f"Failed to parse preferences: {e}"}

# Session state to hold messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("💬 Say something about your interests, hobbies, or goals..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        stream=True,
    )
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 🔍 Extract preferences
    with st.expander("🔍 Extracted Preferences"):
        extracted_prefs = extract_preferences(st.session_state.messages, client)
        st.code(extracted_prefs, language="json")

    # 🎯 Recommend Career Paths
    with st.expander("🎯 Recommended Career Paths"):
        mapped_paths = hybrid_career_recommendation(extracted_prefs)
        for path, explanation in mapped_paths.items():
            st.subheader(path)
            st.write(explanation)
