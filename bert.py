
from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained BERT model
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define the two lists of skills
team_player_skills = ['Accountability', 'Active listening', 'Adaptability', 'Attention to detail', 'Collaboration', 'Collaboration skills', 'Consensus building', 'Constructive criticism', 'Constructive feedback', 'Cooperation', 'Cultural sensitivity', 'Curiosity', 'Dedication', 'Dependability', 'Diplomacy', 'Emotional intelligence', 'Empathy', 'Empowerment of others', 'Flexibility', 'Focus on shared goals', 'Gratitude', 'Honesty', 'Interpersonal skills', 'Listening', 'Open-mindedness', 'Openness to feedback', 'Patience', 'Positive attitude', 'Positive energy', 'Positive reinforcement', 'Problem-solving skills', 'Proactivity', 'Relational skills', 'Reliability', 'Resourcefulness', 'Respect for diversity', 'Respectfulness', 'Responsiveness', 'Selflessness', 'Solution-focused mindset', 'Supportive attitude', 'Supportiveness', 'Team building', 'Teamwork skills', 'Time management', 'Trust in others\' abilities', 'Trust-building', 'Trustworthiness', 'Understanding', 'Willingness to compromise', 'Willingness to learn', 'Willingness to take direction', 'Adaptability to change', 'Accountability for actions', 'Active participation', 'Adaptability to different working styles', 'Conflict management', 'Creativity']
team_leader_skills = ['Visionary', 'Strategic thinking', 'Decision-making', 'Judgement', 'Business acumen', 'Problem-solving', 'Critical thinking', 'Innovative', 'Creative thinking', 'Risk management', 'Change management', 'Goal setting', 'Planning', 'Project management', 'Delegation', 'Time management', 'Prioritization', 'Organization', 'Analytical thinking', 'Attention to detail', 'Thoroughness', 'Execution-focused', 'Results-oriented', 'Initiative-taking', 'Entrepreneurial mindset', 'Resource management', 'Budgeting', 'Financial management', 'Negotiation', 'Influencing', 'Persuasion', 'Coaching', 'Mentoring', 'Teaching', 'Developing others', 'Empowering others', 'Motivating', 'Inspiring', 'Charisma', 'Humility', 'Emotional intelligence', 'Self-awareness', 'Self-reflection', 'Self-improvement', 'Learning mindset', 'Adaptable', 'Flexible', 'Agile', 'Resilient', 'Composed', 'Diplomatic', 'Professionalism', 'Ethics', 'Integrity', 'Accountability', 'Responsibility', 'Authenticity', 'Transparency']


# Define the paragraph to classify
paragraph = "I enjoy working with others and value collaboration. I communicate well and listen attentively to others' ideas. I am adaptable and maintain a positive attitude, and I take pride in my strong work ethic."

# Get the sentence embeddings for the paragraph and the skill sets
paragraph_embedding = model.encode(paragraph)
team_player_embeddings = model.encode(team_player_skills)
team_leader_embeddings = model.encode(team_leader_skills)

print(paragraph_embedding.shape)
print(team_player_embeddings.shape)


paragraph_embedding = model.encode(paragraph).reshape(1, -1)

# Calculate cosine similarity between paragraph and each set of skills
team_player_similarity = np.mean(np.max(np.dot(paragraph_embedding, team_player_embeddings.T), axis=1))
team_leader_similarity = np.mean(np.max(np.dot(paragraph_embedding, team_leader_embeddings.T), axis=1))

# Print the classification result
if team_player_similarity > team_leader_similarity:
    print("The paragraph fits more with team player skills", team_player_similarity, team_leader_similarity)
else:
    print("The paragraph fits more with team leader skills", team_leader_similarity)