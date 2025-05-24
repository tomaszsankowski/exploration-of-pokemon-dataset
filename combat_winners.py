import pandas as pd
import matplotlib.pyplot as plt

# Load the combat dataset
df = pd.read_csv('combats.csv')

# Count how often the winner was the first or the second Pokémon
first_wins = (df['Winner'] == df['First_pokemon']).sum()
second_wins = (df['Winner'] == df['Second_pokemon']).sum()

# Create a bar plot
plt.bar(['First Pokémon', 'Second Pokémon'], [first_wins, second_wins], color=['blue', 'orange'])
plt.title('Number of Wins by Pokémon Order')
plt.ylabel('Number of Wins')
plt.show()
