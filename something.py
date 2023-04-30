

# Define the two lists of skills
team_player_skills = ['Accountability', 'Active listening', 'Adaptability', 'Attention to detail', 'Collaboration', 'Collaboration skills', 'Consensus building', 'Constructive criticism', 'Constructive feedback', 'Cooperation', 'Cultural sensitivity', 'Curiosity', 'Dedication', 'Dependability', 'Diplomacy', 'Emotional intelligence', 'Empathy', 'Empowerment of others', 'Flexibility', 'Focus on shared goals', 'Gratitude', 'Honesty', 'Interpersonal skills', 'Listening', 'Open-mindedness', 'Openness to feedback', 'Patience', 'Positive attitude', 'Positive energy', 'Positive reinforcement', 'Problem-solving skills', 'Proactivity', 'Relational skills', 'Reliability', 'Resourcefulness', 'Respect for diversity', 'Respectfulness', 'Responsiveness', 'Selflessness', 'Solution-focused mindset', 'Supportive attitude', 'Supportiveness', 'Team building', 'Teamwork skills', 'Time management', 'Trust in others\' abilities', 'Trust-building', 'Trustworthiness', 'Understanding', 'Willingness to compromise', 'Willingness to learn', 'Willingness to take direction', 'Adaptability to change', 'Accountability for actions', 'Active participation', 'Adaptability to different working styles', 'Conflict management', 'Creativity']
team_leader_skills = ['Visionary', 'Strategic thinking', 'Decision-making', 'Judgement', 'Business acumen', 'Problem-solving', 'Critical thinking', 'Innovative', 'Creative thinking', 'Risk management', 'Change management', 'Goal setting', 'Planning', 'Project management', 'Delegation', 'Time management', 'Prioritization', 'Organization', 'Analytical thinking', 'Attention to detail', 'Thoroughness', 'Execution-focused', 'Results-oriented', 'Initiative-taking', 'Entrepreneurial mindset', 'Resource management', 'Budgeting', 'Financial management', 'Negotiation', 'Influencing', 'Persuasion', 'Coaching', 'Mentoring', 'Teaching', 'Developing others', 'Empowering others', 'Motivating', 'Inspiring', 'Charisma', 'Humility', 'Emotional intelligence', 'Self-awareness', 'Self-reflection', 'Self-improvement', 'Learning mindset', 'Adaptable', 'Flexible', 'Agile', 'Resilient', 'Composed', 'Diplomatic', 'Professionalism', 'Ethics', 'Integrity', 'Accountability', 'Responsibility', 'Authenticity', 'Transparency']

# Define the person's description
person_description = "I am good at developing 'Big Picture' thinking about complex technology trends and markets, highly useful in my research, I am a team player and often find at ease in coordinating team efforts though a project.I am a non conventional thinker, and a creative person, I am also a good motivator and mentor to my peers, I enjoy exercising design and lateral thinking to develop effective solutions to complex challenges. I enjoy designing urban innovation strategies.I\'v consulted Harvard university in respect of current built environment digital technologies and alternative financing models over 3,5,10 years horizons and prepared by mentoring , lecturing and participating in workshops, master degrees students for future practices."

# Split the person's description into individual words
person_words = person_description.split()

# Count the number of words that match each list
team_player_matches = len(set(person_words).intersection(set(team_player_skills)))
team_leader_matches = len(set(person_words).intersection(set(team_leader_skills)))

# Calculate the percentage match with each list
team_player_percent = (team_player_matches / len(team_player_skills)) * 100
team_leader_percent = (team_leader_matches / len(team_leader_skills)) * 100

# Print the results
print("Percentage match with team player skills:", team_player_percent)
print("Percentage match with team leader skills:", team_leader_percent)