import spacy
nlp = spacy.load("en_core_web_lg")  # load the pre-trained word embedding model
import numpy as np

# define the paragraph and the two lists of team player and team leader skills
paragraph = "I am good at developing 'Big Picture' thinking about complex technology trends and markets, highly useful in my research, I am a team player and often find at ease in coordinating team efforts though a project. I am a non conventional thinker, and a creative person, I am also a good motivator and mentor to my peers, I enjoy exercising design and lateral thinking to develop effective solutions to complex challenges. I enjoy designing urban innovation strategies. I've consulted Harvard university in respect of current built environment digital technologies and alternative financing models over 3,5,10 years horizons and prepared by mentoring, lecturing and participating in workshops, master degrees students for future practices."

# Define the two lists of skills
team_player_skills = ['Accountability', 'Active listening', 'Adaptability', 'Attention to detail', 'Collaboration', 'Collaboration skills', 'Consensus building', 'Constructive criticism', 'Constructive feedback', 'Cooperation', 'Cultural sensitivity', 'Curiosity', 'Dedication', 'Dependability', 'Diplomacy', 'Emotional intelligence', 'Empathy', 'Empowerment of others', 'Flexibility', 'Focus on shared goals', 'Gratitude', 'Honesty', 'Interpersonal skills', 'Listening', 'Open-mindedness', 'Openness to feedback', 'Patience', 'Positive attitude', 'Positive energy', 'Positive reinforcement', 'Problem-solving skills', 'Proactivity', 'Relational skills', 'Reliability', 'Resourcefulness', 'Respect for diversity', 'Respectfulness', 'Responsiveness', 'Selflessness', 'Solution-focused mindset', 'Supportive attitude', 'Supportiveness', 'Team building', 'Teamwork skills', 'Time management', 'Trust in others\' abilities', 'Trust-building', 'Trustworthiness', 'Understanding', 'Willingness to compromise', 'Willingness to learn', 'Willingness to take direction', 'Adaptability to change', 'Accountability for actions', 'Active participation', 'Adaptability to different working styles', 'Conflict management', 'Creativity']
team_leader_skills = ['Visionary', 'Strategic thinking', 'Decision-making', 'Judgement', 'Business acumen', 'Problem-solving', 'Critical thinking', 'Innovative', 'Creative thinking', 'Risk management', 'Change management', 'Goal setting', 'Planning', 'Project management', 'Delegation', 'Time management', 'Prioritization', 'Organization', 'Analytical thinking', 'Attention to detail', 'Thoroughness', 'Execution-focused', 'Results-oriented', 'Initiative-taking', 'Entrepreneurial mindset', 'Resource management', 'Budgeting', 'Financial management', 'Negotiation', 'Influencing', 'Persuasion', 'Coaching', 'Mentoring', 'Teaching', 'Developing others', 'Empowering others', 'Motivating', 'Inspiring', 'Charisma', 'Humility', 'Emotional intelligence', 'Self-awareness', 'Self-reflection', 'Self-improvement', 'Learning mindset', 'Adaptable', 'Flexible', 'Agile', 'Resilient', 'Composed', 'Diplomatic', 'Professionalism', 'Ethics', 'Integrity', 'Accountability', 'Responsibility', 'Authenticity', 'Transparency']

# create a set of all the words in the two lists of skills
all_skills = set(team_player_skills + team_leader_skills)

# create a dictionary to store the cosine similarity scores for each skill
scores = {skill: 0 for skill in all_skills}

# tokenize the paragraph and calculate the embedding for each token
doc = nlp(paragraph)
for token in doc:
    if token.is_alpha and not token.is_stop:  # only consider alphabetical tokens that are not stop words
        token_embedding = token.vector
        # calculate the cosine similarity between the token embedding and the embedding of each skill
        for skill in all_skills:
            skill_embedding = nlp(skill).vector
            cosine_sim = token_embedding.dot(skill_embedding) / (np.linalg.norm(token_embedding) * np.linalg.norm(skill_embedding))
            scores[skill] += cosine_sim  # add the cosine similarity score to the skill's total score

# sort the skills by their total cosine similarity score in descending order
sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# print the top 10 skills with their cosine similarity scores
for skill, score in sorted_skills[:10]:
    print(f"{skill}: {score:.2f}")
