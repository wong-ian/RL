from agents.base import BaseAgent

class RandomAgent(BaseAgent):    
    def get_action(self, observation):
        return self.action_space.sample()
    
    def update(self, state, action, reward, next_state, done):
        pass  # Random agent doesn't learn
    
    def save(self, filepath):
        pass
    
    def load(self, filepath):
        pass