
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import pickle

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        st.info(f"Scraping page {page}...")
        response = requests.get(url, headers=headers)

        # Show a portion of the HTML to debug
        if page == 1:
            st.subheader("Raw HTML (first page):")
            st.code(response.text[:2000], language="html")  # Show first 2000 characters

        soup = BeautifulSoup(response.content, "html.parser")
        job_blocks = soup.find_all("div", class_="ads-details")
        st.warning(f"Found {len(job_blocks)} job listings using class='ads-details'")

        for job in job_blocks:
            title = job.find("h4").get_text(strip=True) if job.find("h4") else "N/A"
            company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True) if job.find("a", href=lambda x: x and "Employer-Profile" in x) else "N/A"
            location = job.find("p").get_text(strip=True) if job.find("p") else "N/A"
            experience = job.find("p", class_="emp-exp").get_text(strip=True) if job.find("p", class_="emp-exp") else "N/A"
            key_skills = job.find("p", class_="job-skills").get_text(strip=True) if job.find("p", class_="job-skills") else "N/A"

            jobs_list.append({
                "Title": title,
                "Company": company,
                "Location": location,
                "Experience": experience,
                "Key Skills": key_skills
            })

    return pd.DataFrame(jobs_list)



def load_model_and_vectorizer():
    with open("karkidi_kmeans_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("karkidi_vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

def cluster_jobs(df, model, vectorizer):
    job_text = df["Title"] + " " + df["Key Skills"]
    features = vectorizer.transform(job_text)
    df["Cluster"] = model.predict(features)
    return df

# Streamlit UI
st.title("Karkidi Job Scraper and Clustering")
st.markdown("Scrape jobs from karkidi.com and cluster them using KMeans")

keyword = st.text_input("Enter Job Keyword", "data science")
pages = st.slider("Number of Pages to Scrape", 1, 10, 1)

if st.button("Scrape and Cluster Jobs"):
    with st.spinner("Scraping jobs..."):
        df = scrape_karkidi_jobs(keyword, pages)
        if df.empty:
            st.warning("No jobs found.")
        else:
            st.success(f"Scraped {len(df)} job listings.")
            model, vectorizer = load_model_and_vectorizer()
            clustered_df = cluster_jobs(df, model, vectorizer)
            st.dataframe(clustered_df)
            csv = clustered_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered CSV", csv, "clustered_jobs.csv", "text/csv")
