# app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import time

def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{}/all/India?search={}"
    jobs_list = []

    for page in range(1, pages + 1):
        st.info(f"Scraping page {page}...")
        url = base_url.format(page, keyword.replace(" ", "%20"))
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, "html.parser")
            job_blocks = soup.find_all("div", class_="ads-details")

            if not job_blocks:
                st.warning(f"No job blocks found on page {page}")
                continue

            for job in job_blocks:
                try:
                    title = job.find("h4").get_text(strip=True)
                    company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                    location = job.find("p").get_text(strip=True)
                    experience = job.find("p", class_="emp-exp").get_text(strip=True)
                    jobs_list.append({
                        "Title": title,
                        "Company": company,
                        "Location": location,
                        "Experience": experience
                    })
                except Exception as e:
                    continue
        except Exception as e:
            st.error(f"Failed to fetch page {page}: {e}")
        time.sleep(1)

    return pd.DataFrame(jobs_list)

def cluster_jobs(df, num_clusters):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df["Title"])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)
    return df

# Streamlit UI
st.title("🔍 Karkidi Job Scraper and Clustering")

keyword = st.text_input("Enter Job Keyword", "data science")
pages = st.slider("Number of Pages to Scrape", 1, 5, 2)
num_clusters = st.slider("Number of Clusters", 2, 10, 3)

if st.button("Scrape and Cluster"):
    jobs_df = scrape_karkidi_jobs(keyword, pages)
    if jobs_df.empty:
        st.warning("No jobs found.")
    else:
        st.success(f"Scraped {len(jobs_df)} jobs.")
        clustered_df = cluster_jobs(jobs_df, num_clusters)
        st.dataframe(clustered_df)

        # Optional: CSV download
        csv = clustered_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clustered_jobs.csv", "text/csv")
