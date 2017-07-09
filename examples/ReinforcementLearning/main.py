import numpy as np
import grid_world
import algorithms as alg

from prettyplots import PrettyPlot
import pandas as pd

if __name__ == "__main__":
	# 1. Load Grid World environment of size 5 x 5
	n_rows = 5
	n_cols = 5
	env = grid_world.GridWorld(n_rows, n_cols)

	# 4 actions are possible - (left, right, up, down)
	n_actions = env.n_actions 

	# 25 states are possible since the grid is of size 5x5
	n_states = env.n_states
 	
	"""
	Get the transition probabilities
		index (i, k, j) of P is the probability of transitioning from 
		state i to j if action k was taken
	"""
	P = env.probability_transition_matrix

	"""
	Get reward for each state 
		There are 4 terminal states 
			2 bad terminal states that return reward of -1
			2 good terminal states that return reward of 1 
	"""
	R = env.rewards

	# Set the decay parameter gamma
	gamma = 0.99

	# Run policy iteration and value iteration
	results = pd.DataFrame()
	for algorithm in ["policy_iteration", "value_iteration"]:
		# Initialize state values
		V = np.zeros(n_states)

		# Randomzie policy vector
		Pi = np.random.randint(0, n_actions, n_states)    

		n_iters = 50
		rewards = np.zeros(n_iters)
		for i in range(n_iters):
			print "\nThe grid world terminal states' rewards are shown below:"
			print np.reshape(R[0,0], (n_rows, n_cols))

			print "\nTest the policy shown below:"
			print np.reshape(map(grid_world.convert, Pi), (n_rows, n_cols))


			# 1. Test how much reward current policy gets us
			total_reward = 0.
			n_episodes = 50
			for i_episode in range(n_episodes):
				# Reset environment to beginning 
				obs = env.reset() 

				for t in xrange(1000): 
					# Get action from the learned policy 
					action = Pi[obs]

					# Observe next step and get reward 
					obs, reward, done, _ = env.step(action)

					total_reward += reward

					if done:
						# If terminal state reached, terminate episode
						break

			avg_reward = total_reward / float(n_episodes)
			rewards[i] = avg_reward
			print ("Epoch %d - Average reward: %.3f\n" % 
				  (i, avg_reward))


			# 2. IMPROVE POLICY
			Pi, V = alg.apply_algorithm(algorithm, Pi, V, R, P, 
										n_states, n_actions, gamma)
 
		results[algorithm] = rewards

	# You can install this by: pip install pretty-plots
	pp = PrettyPlot(ylabel="Average reward", xlabel="Epoch")
	pp.plot_DataFrame(results)
	pp.show()