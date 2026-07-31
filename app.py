import streamlit as st

st.set_page_config(
    page_title="AI Career Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("AI Career Assistant")

st.write(
    "Upload one or more master CVs and compare them against a Job Description."
)

st.divider()

st.header("1. Upload Master CVs")

uploaded_files = st.file_uploader(
    "Choose one or more PDF CVs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} CV(s) uploaded.")

    st.subheader("Uploaded Files")

    for file in uploaded_files:
        st.write(f"• {file.name}")

st.divider()

st.header("2. Paste Job Description")

job_description = st.text_area(
    "Paste the complete Job Description below",
    height=250
)

st.divider()

analyse = st.button("Analyse CVs")

if analyse:

    if not uploaded_files:
        st.warning("Please upload at least one CV.")

    elif not job_description.strip():
        st.warning("Please paste a Job Description.")

    else:
        st.success("Everything looks good!")

        st.write("Analysis module coming next...")