# imports
import spacy
from spacy.matcher import PhraseMatcher
import streamlit as st

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

st.title('Resume Skill Extraction')
st.text('Made by Akhilesh Sivaganesan')

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

# extract skills from job_description
job_description = st.text_area('Text to Analyze', "You are a Python developer with a solid experience in web development and can manage projects. You quickly adapt to new environments and speak fluently English and French")
st.write(skill_extractor.describe(skill_extractor.annotate(job_description)))