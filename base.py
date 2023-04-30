import nltk

# Define the list of keywords
team_player_skills = ['Collaboration', 'Communication', 'Adaptability', ...]
team_leader_skills = ['Team building', 'Dedication', 'Selflessness', ...]

# Define the text string to match against
text = "I am good at developing 'Big Picture' thinking about complex technology trends and markets, highly useful in my research, I am a team player and often find at ease in coordinating team efforts though a project."

# Tokenize the text into individual words
tokens = nltk.word_tokenize(text)

# Count the number of matches for each set of keywords
team_player_matches = len(set(tokens).intersection(set(team_player_skills)))
team_leader_matches = len(set(tokens).intersection(set(team_leader_skills)))

# Print the results
print("Team player skills matched:", team_player_matches)
print("Team leader skills matched:", team_leader_matches)