import math
from llama_cpp import Llama
import re
import numpy as np
import random

class MCTS_NODE:
    def __init__(self, parent, solution, critique, Q_value):
        self.parent = parent
        self.solution = solution
        self.critique = critique
        self.Q_value = Q_value
        self.visit_count = 0
        self.children = []  
        self.reward_samples = [] 
        self.fully_expanded = False 
        self.expanded_children = 0 
    
class MCTS_REASONING_LLM:
    def __init__(self, model_path, max_child=5, c=1):
        self.model = Llama(model_path, n_ctx=16384, n_gpu_layers=99, logits_all=False, verbose=False)
        self.max_child = max_child
        self.c = c
        self.nodes = []
        self.query = None
        self.dummy_answers = [
            "I Don't Know",
            "I can't understand this question.",
            "I can't help with this question.",
            "I don't know how to solve this question.",
            "I don't know the answer to this question.",
            "I don't know the answer to this question, sorry."
        ]

    def is_fully_expanded(self, idx): 
        if len(self.nodes[idx].children) >= self.max_child:
            return True
        for child_idx in self.nodes[idx].children:
            if self.nodes[child_idx].Q_value > self.nodes[idx].Q_value:
                return True
        return False

    def get_optimum_child(self, idx):
        if not self.nodes[idx].children: 
            return -1
        if self.nodes[idx].parent != -1:
            parent_visit_count = self.nodes[self.nodes[idx].parent].visit_count
        else:
            parent_visit_count = 1
        UCT = []
        for child_idx in self.nodes[idx].children:
            child_visit_count = self.nodes[child_idx].visit_count
            uct_value = (self.nodes[child_idx].Q_value + 
                        self.c * math.sqrt(math.log(parent_visit_count + 1) / (child_visit_count + 1e-6)))
            UCT.append(uct_value)
        
        if (not self.nodes[idx].fully_expanded and 
            np.max(UCT) < self.c * math.sqrt(math.log(parent_visit_count + 1) / 1e-6)):
            return -1
            
        return self.nodes[idx].children[np.argmax(UCT)]

    def generate_critique(self, query, solution):
        print(f"[generate_critique] Generating critique for solution...")
        
        prompt = f"""<|start_header_id|>user<|end_header_id|>Since we have a weak Answer, could you provide me with a reflection or feedback to correct this answer better? Analyze this Answer Strictly and Critically, point out every flaw for every possible imperfect to minus every possible score!

Question: {query}
Answer: {solution}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Let's think step by step."""

        try:
            response = self.model(prompt=prompt, max_tokens=1024, temperature=0.8)
            critique = response["choices"][0]["text"].strip()
            print(f"[generate_critique] Critique generated successfully: {critique}...")
            return critique
        except Exception as e:
            print(f"[generate_critique] ERROR: {str(e)}")
            return "The answer needs improvement."

    def generate_refined_solution(self, query, original_solution, critique):
        print(f"[generate_refined_solution] Refining solution...")
        
        prompt = f"""<|start_header_id|>user<|end_header_id|>Please refine your answer according to the Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with "[Final Answer] The answer is [answer formula]"

Question: {query}
Original Answer: {original_solution}
Feedback: {critique}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Let's think step by step."""

        try:
            response = self.model(prompt=prompt, max_tokens=2048, temperature=0.8)
            refined = response["choices"][0]["text"].strip()
            print(f"[generate_refined_solution] Refined solution generated successfully: {refined}...")
            return refined
        except Exception as e:
            print(f"[generate_refined_solution] ERROR: {str(e)}")
            return original_solution

    def extract_score_from_text(self, text):
        # Look for patterns like [Score] -50, [Score]: -50, Score: -50, etc.
        score_patterns = [
            r'\[Score\]\s*[-]?\d+',  # [Score] -50
            r'\[Score\]:\s*[-]?\d+',  # [Score]: -50
            r'Score:\s*[-]?\d+',     # Score: -50
            r'Score\s+[-]?\d+',      # Score -50
            r'score\s*[:=]\s*[-]?\d+',  # score: -50 or score = -50
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract just the number from the match
                number_match = re.search(r'[-]?\d+', match.group())
                if number_match:
                    return int(number_match.group())
        
        # Fallback: look for the last number in the text (often the final score)
        all_numbers = re.findall(r'[-]?\d+', text)
        if all_numbers:
            # Filter numbers to reasonable score range
            valid_scores = [int(num) for num in all_numbers if -100 <= int(num) <= 100]
            if valid_scores:
                return valid_scores[-1]  # Take the last valid score
        
        return 0  # Default if no score found

    def self_evaluate(self, query, solution, num_samples=3):
        print(f"[self_evaluate] Evaluating solution with {num_samples} samples...")
        
        scores = []
        for i in range(num_samples):
            prompt = f"""<|start_header_id|>user<|end_header_id|>Question: {query}
Answer: {solution}

Analyze this Answer Strictly and Critically, and point out every flaw for every possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative.

Output a score between [-100,+100].

Format: [Analysis] your analysis here [Score] your_number_here

Example: [Analysis] The solution has calculation errors and lacks proper reasoning. [Score] -45<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

            try:
                response = self.model(prompt=prompt, max_tokens=512, temperature=0.8)
                text = response["choices"][0]["text"]
                print(f"[self_evaluate] Sample {i+1} response: {text}...")
                
                # Extract score using improved method
                score = self.extract_score_from_text(text)
                
                # Full Score Suppression: reduce scores above 95
                if score > 95:
                    score = max(95, score - 10)
                
                # Clamp score to valid range
                score = max(-100, min(100, score))
                scores.append(score)
                print(f"[self_evaluate] Sample {i+1} extracted score: {score}")
                    
            except Exception as e:
                print(f"[self_evaluate] ERROR in sample {i+1}: {str(e)}")
                scores.append(0)
        
        print(f"[self_evaluate] Final scores: {scores}")
        return scores
    
    def calculate_q_value(self, reward_samples):
        """Calculate Q value using formula from paper: Q(a) = 1/2 * (min(R_a) + mean(R_a))"""
        if not reward_samples:
            return 0
        
        min_reward = min(reward_samples)
        mean_reward = sum(reward_samples) / len(reward_samples)
        q_value = 0.5 * (min_reward + mean_reward)
        
        return q_value
    
    def update_q_value_with_children(self, node_index):
        """Update Q value considering children: Q'(a) = 1/2 * (Q(a) + max_child_Q)"""
        node = self.nodes[node_index]
        
        # Calculate base Q value from own rewards
        base_q = self.calculate_q_value(node.reward_samples)
        
        # Find maximum Q value among children
        max_child_q = float('-inf')
        has_children = False
        
        for child_idx in node.children:
            child_q = self.nodes[child_idx].Q_value
            max_child_q = max(max_child_q, child_q)
            has_children = True
        
        # Update Q value: Q'(a) = 1/2 * (Q(a) + max_child_Q)
        if has_children:
            node.Q_value = 0.5 * (base_q + max_child_q)
        else:
            node.Q_value = base_q

    def mcts_init(self, query):
        print(f"[mcts_init] Initializing MCTS for query: {query[:50]}...")
        
        self.query = query
        self.nodes = []
        
        # Create root node with dummy answer
        dummy_solution = random.choice(self.dummy_answers)
        critique = self.generate_critique(self.query, dummy_solution)
        
        # Create root node
        root_node = MCTS_NODE(-1, dummy_solution, critique, 0)
        
        # Evaluate root node
        root_node.reward_samples = self.self_evaluate(query, dummy_solution)
        root_node.Q_value = self.calculate_q_value(root_node.reward_samples)
        root_node.visit_count = 1
        
        self.nodes.append(root_node)
        print(f"[mcts_init] Root node created with Q-value: {root_node.Q_value}")

    def iterator(self):
        """Single MCTS iteration combining all phases: Selection -> Expansion -> Evaluation -> Backpropagation"""
        # SELECTION PHASE: Navigate to leaf node
        current_node = 0
        previous_node = -1
        while current_node != -1:
            previous_node = current_node
            current_node = self.get_optimum_child(current_node)
        
        # EXPANSION PHASE: Create refined solution
        solution = self.generate_refined_solution(
            self.query, 
            self.nodes[previous_node].solution, 
            self.nodes[previous_node].critique 
        )
        critique = self.generate_critique(self.query, solution)
        
        # Create new child node
        new_node = MCTS_NODE(previous_node, solution, critique, 0)
        self.nodes.append(new_node)
        current_node = len(self.nodes) - 1
        
        # Add child to parent's children list
        self.nodes[previous_node].children.append(current_node)
        
        # EVALUATION PHASE: Self-evaluate the new solution
        self.nodes[current_node].reward_samples = self.self_evaluate(self.query, solution)
        self.nodes[current_node].Q_value = self.calculate_q_value(self.nodes[current_node].reward_samples)
        self.nodes[current_node].visit_count = 1
        
        # BACKPROPAGATION PHASE: Update Q values up the tree
        while previous_node != -1:
            self.nodes[previous_node].visit_count += 1
            self.update_q_value_with_children(previous_node)
            self.nodes[previous_node].fully_expanded = self.is_fully_expanded(previous_node)  # Fixed: was 'fully_expended'
            previous_node = self.nodes[previous_node].parent

    def run(self, query, iterations=100):
        """Run MCTS for specified number of iterations"""
        print(f"[run] Starting MCTS with {iterations} iterations...")
        self.mcts_init(query)
        for i in range(iterations):
            print(f"[run] Iteration {i+1}/{iterations}")
            self.iterator()
            
            # Print progress
            if (i + 1) % 10 == 0:
                best_node = max(self.nodes, key=lambda n: n.Q_value)
                print(f"[run] Best Q-value after {i+1} iterations: {best_node.Q_value}")
        
        # Return best solution
        best_node = max(self.nodes, key=lambda n: n.Q_value)
        print(f"[run] Final best Q-value: {best_node.Q_value}")
        return best_node.solution

    def get_best_solution(self):
        """Get the solution with highest Q-value"""
        if not self.nodes:
            return None
        
        best_node = max(self.nodes, key=lambda n: n.Q_value)
        return best_node.solution
    

# Initialize the model
mctsr = MCTS_REASONING_LLM("llama-3.1-8b-instruct-q4_k_m.gguf")

# Solve a problem
query = "What is the sum of the first 10 prime numbers?"
solution = mctsr.run(query, 20)

# Get tree statistics
final_answer = mctsr.get_best_solution()
print(f"Final answer: {final_answer}")