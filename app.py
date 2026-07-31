from utils.parser import extract_pdf_text
import streamlit as st
import os 
from utils.resume_parser import parse_resume
from utils.matcher import match_keywords

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
         pdf_text, page_count = extract_pdf_text(file)
         st.write(f"Pages detected: {page_count}")
         st.write(f"Characters extracted: {len(pdf_text)}")
         if "cv_texts" not in st.session_state:
             st.session_state.cv_texts = {}

         st.session_state.cv_texts[file.name] = pdf_text
         resume = parse_resume(pdf_text)

         st.write(f"**Name:** {resume['name']}")
         st.write(f"**Email:** {resume['email']}")
         st.write(f"**Phone:** {resume['phone']}")

         with st.expander(f"Parsed Sections - {file.name}"):
             for section_name, lines in resume["sections"].items():
                 st.markdown(f"**{section_name}**")
                 st.text("\n".join(lines))

         # Create folder if it doesn't exist
         os.makedirs("data/extracted", exist_ok=True)

    # Save extracted text to a file
         with open(
             f"data/extracted/{file.name}.txt",
             "w",
             encoding="utf-8"
        ) as f:
              f.write(pdf_text)

         st.caption(f"Characters extracted: {len(pdf_text)}")

         with st.expander(f"Preview - {file.name}"):
             st.code(pdf_text)

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
        st.success("Analysis complete!")

        for file in uploaded_files:
            cv_text = st.session_state.cv_texts[file.name]
            result = match_keywords(cv_text, job_description)

            st.subheader(file.name)
            st.metric("Keyword Match Score", f"{result['score']}%")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Matched Keywords**")
                st.write(", ".join(result["matched"]) if result["matched"] else "None")

            with col2:
                st.write("**Missing Keywords**")
                st.write(", ".join(result["missing"]) if result["missing"] else "None")