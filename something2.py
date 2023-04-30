import spacy
import numpy as np

nlp = spacy.load("en_core_web_lg")


# Define the two lists of skills
team_player_skills = ['Accountability', 'Active listening', 'Adaptability', 'Attention to detail', 'Collaboration', 'Collaboration skills', 'Consensus building', 'Constructive criticism', 'Constructive feedback', 'Cooperation', 'Cultural sensitivity', 'Curiosity', 'Dedication', 'Dependability', 'Diplomacy', 'Emotional intelligence', 'Empathy', 'Empowerment of others', 'Flexibility', 'Focus on shared goals', 'Gratitude', 'Honesty', 'Interpersonal skills', 'Listening', 'Open-mindedness', 'Openness to feedback', 'Patience', 'Positive attitude', 'Positive energy', 'Positive reinforcement', 'Problem-solving skills', 'Proactivity', 'Relational skills', 'Reliability', 'Resourcefulness', 'Respect for diversity', 'Respectfulness', 'Responsiveness', 'Selflessness', 'Solution-focused mindset', 'Supportive attitude', 'Supportiveness', 'Team building', 'Teamwork skills', 'Time management', 'Trust in others\' abilities', 'Trust-building', 'Trustworthiness', 'Understanding', 'Willingness to compromise', 'Willingness to learn', 'Willingness to take direction', 'Adaptability to change', 'Accountability for actions', 'Active participation', 'Adaptability to different working styles', 'Conflict management', 'Creativity']
team_leader_skills = ['Visionary', 'Strategic thinking', 'Decision-making', 'Judgement', 'Business acumen', 'Problem-solving', 'Critical thinking', 'Innovative', 'Creative thinking', 'Risk management', 'Change management', 'Goal setting', 'Planning', 'Project management', 'Delegation', 'Time management', 'Prioritization', 'Organization', 'Analytical thinking', 'Attention to detail', 'Thoroughness', 'Execution-focused', 'Results-oriented', 'Initiative-taking', 'Entrepreneurial mindset', 'Resource management', 'Budgeting', 'Financial management', 'Negotiation', 'Influencing', 'Persuasion', 'Coaching', 'Mentoring', 'Teaching', 'Developing others', 'Empowering others', 'Motivating', 'Inspiring', 'Charisma', 'Humility', 'Emotional intelligence', 'Self-awareness', 'Self-reflection', 'Self-improvement', 'Learning mindset', 'Adaptable', 'Flexible', 'Agile', 'Resilient', 'Composed', 'Diplomatic', 'Professionalism', 'Ethics', 'Integrity', 'Accountability', 'Responsibility', 'Authenticity', 'Transparency']


paragraph = "As a team, we work together to achieve our goals. Everyone has a role to play and we support each other in our tasks. Communication is key and we make sure to listen to each other's ideas and concerns. Collaboration is important and we value cooperation to get things done."

token_embeddings = [nlp(word.text.lower()).vector for word in nlp(paragraph.lower()) if not word.is_stop and word.is_alpha]
team_player_embeddings = [nlp(word).vector for word in team_player_skills]
team_leader_embeddings = [nlp(word).vector for word in team_leader_skills]

team_player_similarity = np.mean([np.max(np.dot(embedding, np.transpose(token_embeddings))) for embedding in team_player_embeddings])
team_leader_similarity = np.mean([np.max(np.dot(embedding, np.transpose(token_embeddings))) for embedding in team_leader_embeddings])

if team_player_similarity > team_leader_similarity:
    print("The paragraph fits more with team player skills ", team_player_similarity)
else:
    print("The paragraph fits more with team leader skills ", team_leader_similarity)