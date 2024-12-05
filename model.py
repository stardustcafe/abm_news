import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class News:
    def __init__(self, familiarity, virality, level_of_detail, trust_in_source):
        self.familiarity = familiarity  # Familiarity to the demographics (0-1).
        self.virality = virality  # Virality of headlines (0-1), affects spread.
        self.level_of_detail = level_of_detail  # Level of details in terms of facts (0-1), higher means more factual.
        self.trust_in_source = trust_in_source  # Trust in the source of the news (0-1), higher means more trusted.

class Agent:
    def __init__(self, id, network, susceptibility, knowledge_level, propagation_probability, belief_alignment, trust_in_authority):
        self.id = id
        self.believes_fake_news = False  # Initially, they do not believe fake news.
        self.network = network  # Reference to the network of agents.
        self.susceptibility = susceptibility  # Susceptibility to believing fake news (0-1).
        self.knowledge_level = knowledge_level  # Knowledge level (0-1), higher reduces susceptibility.
        self.propagation_probability = propagation_probability  # Probability of spreading fake news if they believe.
        self.counter_news_received = False  # Indicates if the agent received counter news.
        self.belief_alignment = belief_alignment  # Represents alignment of fake news with the agent's beliefs (0-1).
        self.trust_in_authority = trust_in_authority  # Trust in authoritative sources (0-1), influences susceptibility.
        self.repeated_exposure_count = 0  # Count of how many times the agent has been exposed to fake news.

    def calculate_effective_susceptibility(self, news):
        """Calculate the agent's effective susceptibility based on agent-specific attributes and news attributes."""
        effective_susceptibility = self.susceptibility * (1 - self.knowledge_level) * news.familiarity
        effective_susceptibility *= (1 + self.belief_alignment * news.trust_in_source)
        repeated_exposure_factor = min(1.0, 0.1 * self.repeated_exposure_count)  # Repeated exposure increases susceptibility
        return effective_susceptibility * (1 + repeated_exposure_factor)

    def interact(self, news):
        # Randomly choose one of their neighbors (connected agents).
        neighbors = self.network[self.id]
        if neighbors:  # Ensure there are neighbors
            neighbor_id = random.choice(neighbors)
            neighbor = agents[neighbor_id]

            # Spread fake news if this agent believes it.
            if self.believes_fake_news and not neighbor.believes_fake_news and not neighbor.counter_news_received:
                # Increase repeated exposure count
                neighbor.repeated_exposure_count += 1

                # The neighbor might start believing the fake news based on susceptibility, knowledge level, news familiarity, and repeated exposure.
                effective_susceptibility = neighbor.calculate_effective_susceptibility(news)
                if random.random() < effective_susceptibility:
                    neighbor.believes_fake_news = True

            # Spread fake news to neighbors based on propagation probability and news virality
            if self.believes_fake_news and random.random() < self.propagation_probability * news.virality:
                neighbor.believes_fake_news = True

    def receive_counter_news(self, counter_news_effectiveness):
        # Receiving counter news probabilistically reduces the likelihood of believing fake news
        if self.believes_fake_news and random.random() < counter_news_effectiveness:
            self.believes_fake_news = False
            self.counter_news_received = True

    def __repr__(self):
        return f"Agent({self.id}, Believes: {self.believes_fake_news}, Susceptibility: {self.susceptibility}, Knowledge: {self.knowledge_level}, Propagation Probability: {self.propagation_probability})"

# Function to create a random network
def create_network(num_agents, connectivity=0.1):
    network = {i: [] for i in range(num_agents)}
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            if random.random() < connectivity:
                network[i].append(j)
                network[j].append(i)  # Undirected graph
    return network

# Function to generate more realistic demographics data
def generate_realistic_demographics(num_agents):
    # Assume age follows a normal distribution, with mean 35 years and std deviation of 15 years
    ages = np.random.normal(loc=35, scale=15, size=num_agents)
    ages = np.clip(ages, 18, 90)  # Limit the ages between 18 and 90

    # Educational level: 1 = low, 2 = medium, 3 = high
    education_level = np.random.choice([1, 2, 3], size=num_agents, p=[0.4, 0.4, 0.2])

    # Susceptibility: More susceptible to fake news if younger and less educated
    susceptibility = np.clip(np.random.normal(loc=0.5, scale=0.2, size=num_agents), 0, 1)
    susceptibility += (ages < 30) * 0.2  # Younger agents are more susceptible
    susceptibility -= (education_level == 3) * 0.2  # Higher education reduces susceptibility
    susceptibility = np.clip(susceptibility, 0, 1)

    # Knowledge level: Older and more educated people tend to have more knowledge
    knowledge_level = np.clip(np.random.normal(loc=0.7, scale=0.2, size=num_agents), 0, 1)
    knowledge_level += (ages > 50) * 0.2  # Older people tend to have more knowledge
    knowledge_level += (education_level == 3) * 0.2  # More educated individuals have higher knowledge
    knowledge_level = np.clip(knowledge_level, 0, 1)

    # Propagation probability: Younger, more socially active agents are more likely to spread fake news
    propagation_probability = np.clip(np.random.normal(loc=0.5, scale=0.2, size=num_agents), 0, 1)
    propagation_probability += (ages < 30) * 0.3  # Younger agents are more likely to spread news
    propagation_probability = np.clip(propagation_probability, 0, 1)

    # Belief alignment: More likely to believe fake news that aligns with their political/social views
    belief_alignment = np.random.uniform(0, 1, size=num_agents)

    # Trust in authority: Older and more educated people tend to trust authority more
    trust_in_authority = np.clip(np.random.normal(loc=0.6, scale=0.2, size=num_agents), 0, 1)
    trust_in_authority += (ages > 50) * 0.2  # Older individuals tend to trust authority more
    trust_in_authority += (education_level == 3) * 0.2  # More educated individuals trust authority more
    trust_in_authority = np.clip(trust_in_authority, 0, 1)

    # Combine the data into a DataFrame
    demographics_data = pd.DataFrame({
        'age': ages,
        'education_level': education_level,
        'susceptibility': susceptibility,
        'knowledge_level': knowledge_level,
        'propagation_probability': propagation_probability,
        'belief_alignment': belief_alignment,
        'trust_in_authority': trust_in_authority
    })

    return demographics_data

# Function to initialize agents with real-world demographic data
def initialize_agents_with_demographics(network, demographics_data):
    agents = []
    for i, row in demographics_data.iterrows():
        agent = Agent(i, network, row['susceptibility'], row['knowledge_level'], row['propagation_probability'],
                      row['belief_alignment'], row['trust_in_authority'])
        # Initially, set belief in fake news to False for everyone
        agent.believes_fake_news = False
        agents.append(agent)
    return agents

# Simulation function
def run_simulation(num_agents, steps, initial_fake_news_spreaders=1, counter_news_delay=10, counter_news_effectiveness=0.7, demographics_data=None, fact_checking_enabled=True, forgetting_enabled=False):
    network = create_network(num_agents)
    global agents
    if demographics_data is not None:
        agents = initialize_agents_with_demographics(network, demographics_data)
    else:
        agents = initialize_agents_with_demographics(network, pd.DataFrame())  # Fallback if no demographics data is provided

    # Create the fake news with specific attributes
    news = News(familiarity=0.7, virality=0.8, level_of_detail=0.3, trust_in_source=0.6)

    # Initially, set a few agents to believe in fake news
    for i in random.sample(range(num_agents), initial_fake_news_spreaders):
        agents[i].believes_fake_news = True

    # Store the number of believers over time for plotting and agent attributes over time
    believers_over_time = []
    agent_attributes_over_time = []

    for step in range(steps):
        # Track agent attributes at each time step (including belief status as 1 or 0)
        agent_data = [{
            'id': agent.id,
            'susceptibility': agent.susceptibility,
            'knowledge_level': agent.knowledge_level,
            'propagation_probability': agent.propagation_probability,
            'belief_alignment': agent.belief_alignment,
            'trust_in_authority': agent.trust_in_authority,
            'believes_fake_news': 1 if agent.believes_fake_news else 0
        } for agent in agents]

        agent_attributes_over_time.append(agent_data)

        # Fake news may die out and competing news takes over, causing some agents to forget the fake news
        if forgetting_enabled and step > 0 and step % 20 == 0:  # Every 20 steps, reduce fake news belief due to competing news
            for agent in agents:
                if agent.believes_fake_news and random.random() < 0.05:  # 30% chance of forgetting
                    agent.believes_fake_news = False
                elif not agent.believes_fake_news and random.random() < 0.1:  # 10% chance of still remembering
                    agent.believes_fake_news = True

        # Random exposure to fake news for a few agents
        random_exposure_agents = random.sample(agents, k=int(0.05 * num_agents))  # 5% of agents exposed randomly
        for agent in random_exposure_agents:
            if not agent.believes_fake_news and not agent.counter_news_received:
                effective_susceptibility = agent.calculate_effective_susceptibility(news)
                if random.random() < effective_susceptibility:
                    agent.believes_fake_news = True

        # Each agent interacts with the network and spreads fake news
        for agent in agents:
            agent.interact(news)

        # Introduce counter news at specified delay with a gradual effect
        if fact_checking_enabled and step > 0:
            for agent in agents:
                if agent.believes_fake_news and random.random() < (counter_news_effectiveness / 2):
                    agent.receive_counter_news(counter_news_effectiveness)

        # Count how many agents believe in fake news
        num_believers = sum(agent.believes_fake_news for agent in agents)
        believers_over_time.append(num_believers)

        print(f"Step {step + 1}: {num_believers} out of {num_agents} believe in fake news.")

    return believers_over_time, agent_attributes_over_time, network

# Plot the results of correlation between agent attributes and fake news belief
def plot_correlation(agent_attributes_over_time):
    all_data = []
    
    for step in agent_attributes_over_time:
        for agent in step:
            all_data.append(agent)

    df = pd.DataFrame(all_data)
    correlation_matrix = df.corr()
    plt.clf()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Correlation between Agent Attributes and Belief in Fake News")
    #plt.show()
    plt.savefig("correlation.png",dpi=120,bbox_inches='tight')
# Example usage:

num_agents = 100
steps = 100

# Generate realistic demographics data
demographics_data = generate_realistic_demographics(num_agents)

# Run the simulation and plot the correlation
believers_over_time, agent_attributes_over_time, network = run_simulation(
    num_agents, steps, demographics_data=demographics_data,fact_checking_enabled=True,forgetting_enabled=False
)

plot_correlation(agent_attributes_over_time)
plt.plot(range(steps), believers_over_time)
plt.xlabel('Timestep')
plt.ylabel('Number of Believers')
plt.title('Spread of Fake News Over Time')
plt.show()
