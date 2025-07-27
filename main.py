from llama_cpp import Llama
import numpy as np

class MCTS_REASONING_LLM:
    class MCTS_NODE:
        def __init__(self, parent, actions_and_probs, k):
            self.parent = parent
            self.last_visit = -1
            self.end = False
            (self.actions, self.probs) = actions_and_probs
            self.actions = np.array(self.actions)
            self.probs = np.array(self.probs)
            self.child = np.full((k,), -1)
            self.mean_reward = np.full((k,), 0)
            self.visit_count = 0
            self.child_visit_count = np.full((k,), 0)

        def get_optimum_child(self, policy_weight, explore_weight):
            score = self.mean_reward + self.probs * policy_weight + \
                    (np.sqrt(np.log(self.visit_count + 1e-6)) / (self.child_visit_count + 1e-6)) * explore_weight
            return np.argmax(score)

    def __init__(self, model_path, k, policy_weight=1, explore_weight=1):
        self.model = Llama(model_path, n_ctx=16384, n_gpu_layers=99, logits_all=True, verbose=False)
        self.clean()
        self.nodes = []
        self.policy_weight = policy_weight
        self.explore_weight = explore_weight
        self.k = k

    def push_back(self, role, content):
        self.stream += f" <|start_header_id|>{role}<|end_header_id|>" \
                       f"{content}" \
                       "<|eot_id|>"

    def clean(self):
        self.stream = ""
        self.push_back("system", "You are a helpful assistant. Always think step-by-step before answering and format your response as follows:\n"
                                 "<step 1 content>\n"
                                 "<step 2 content>\n"
                                 "...\n"
                                 "[answer]\n"
                                 "<answer content>\n"
                                 "Ensure every response follows this format, with each reasoning step on a new line and the answer preceded by [answer] on a new line, followed by its content on the next line.")

    def generate_action_list(self, prompt):
        output = self.model(prompt=prompt, max_tokens=1, logprobs=self.k, temperature=0)
        output = output["choices"][0]["logprobs"]["top_logprobs"][0]
        first_tokens = list(output.keys())
        probs = list(output.values())
        actions = []
        for token in first_tokens:
            continuation = self.model(prompt=prompt + token, max_tokens=256,
                                      logprobs=0, temperature=0, stop='\n')["choices"][0]["text"]
            actions.append(token + continuation + "\n")
        return (actions, probs)

    def generate_answer_list(self, prompt):
        output = self.model(prompt=prompt, max_tokens=1, logprobs=self.k, temperature=0)
        output = output["choices"][0]["logprobs"]["top_logprobs"][0]
        first_tokens = list(output.keys())
        probs = list(output.values())
        actions = []
        for token in first_tokens:
            continuation = self.model(prompt=prompt + token, max_tokens=512,
                                      logprobs=0, temperature=0)["choices"][0]["text"]
            actions.append(token + continuation)
        return (actions, probs)

    def answer_evaluate(self, query_answer):
        # 简单占位符评估：始终返回 1.0
        return 1.0

    def MCTS_initialize(self):
        self.nodes = []
        self.deleted = []
        actions_and_probs = self.generate_action_list(self.stream)
        root_node = self.MCTS_NODE(-1, actions_and_probs, self.k)
        self.nodes.append(root_node)
        self.set_root(0)

    def new_node(self, parent, actions_and_probs):
        new_node = self.MCTS_NODE(parent, actions_and_probs, self.k)
        if len(self.deleted):
            idx = self.deleted[0]
            self.nodes[idx] = new_node
            self.deleted = self.deleted[1:]
            return idx
        else:
            self.nodes.append(new_node)
            return len(self.nodes) - 1

    def delete_node(self, idx):
        self.deleted.append(idx)

    def delete_tree(self, idx):
        for child in self.nodes[idx].child:
            if child != -1:
                self.delete_tree(child)
        self.deleted.append(idx)

    def set_root(self, idx):
        self.root = idx
        self.nodes[idx].parent = -1

    def select_and_expand(self):
        previous_node = -1
        current_node = self.root
        last_visit = -1
        current_prompt = self.stream
        while current_node != -1 and not self.nodes[current_node].end:
            last_visit = self.nodes[current_node].get_optimum_child(self.policy_weight, self.explore_weight)
            self.nodes[current_node].last_visit = last_visit
            current_prompt += self.nodes[current_node].actions[last_visit]
            previous_node = current_node
            current_node = self.nodes[current_node].child[last_visit]

        if current_node != -1:
            return current_node, current_prompt
        else:
            if self.nodes[previous_node].actions[last_visit] == "[answer]\n":
                actions, probs = self.generate_answer_list(current_prompt)
                new_node_idx = self.new_node(previous_node, (actions, probs))
                self.nodes[previous_node].child[last_visit] = new_node_idx
                current_node = new_node_idx
                self.nodes[current_node].end = True
                for i in range(len(actions)):
                    self.nodes[current_node].mean_reward[i] = self.answer_evaluate(current_prompt + actions[i] + '<|eot_id|>')
            else:
                actions, probs = self.generate_action_list(current_prompt)
                new_node_idx = self.new_node(previous_node, (actions, probs))
                self.nodes[previous_node].child[last_visit] = new_node_idx
                current_node = new_node_idx
            return current_node, current_prompt

    def simulation(self, current_node, current_prompt):
        if self.nodes[current_node].end:
            optimum_child = self.nodes[current_node].get_optimum_child(self.policy_weight, self.explore_weight)
            self.nodes[current_node].last_visit = optimum_child
            reward = self.nodes[current_node].mean_reward[optimum_child]
            return reward
        else:
            optimum_child = self.nodes[current_node].get_optimum_child(self.policy_weight, self.explore_weight)
            self.nodes[current_node].last_visit = optimum_child
            next_prompt = current_prompt + self.nodes[current_node].actions[optimum_child]
            if self.nodes[current_node].child[optimum_child] == -1:
                if self.nodes[current_node].actions[optimum_child] == "[answer]\n":
                    actions, probs = self.generate_answer_list(next_prompt)
                    new_node_idx = self.new_node(current_node, (actions, probs))
                    self.nodes[current_node].child[optimum_child] = new_node_idx
                    child_node = new_node_idx
                    self.nodes[child_node].end = True
                    for i in range(len(actions)):
                        self.nodes[child_node].mean_reward[i] = self.answer_evaluate(next_prompt + actions[i] + '<|eot_id|>')
                else:
                    actions, probs = self.generate_action_list(next_prompt)
                    new_node_idx = self.new_node(current_node, (actions, probs))
                    self.nodes[current_node].child[optimum_child] = new_node_idx
                    child_node = new_node_idx
            else:
                child_node = self.nodes[current_node].child[optimum_child]
            return self.simulation(child_node, next_prompt)

    def backpropagation(self, current_node, reward):
        while current_node != -1:
            self.nodes[current_node].visit_count += 1
            last_visit = self.nodes[current_node].last_visit
            if last_visit != -1:
                self.nodes[current_node].child_visit_count[last_visit] += 1
                n = self.nodes[current_node].child_visit_count[last_visit]
                self.nodes[current_node].mean_reward[last_visit] += (reward - self.nodes[current_node].mean_reward[last_visit]) / n
            current_node = self.nodes[current_node].parent

    def query(self, query, iterations=100):
        self.push_back("user", query)
        self.stream += "<|start_header_id|>assistant<|end_header_id|>"
        self.MCTS_initialize()
        while not self.nodes[self.root].end:
            while self.nodes[self.root].visit_count < iterations:
                current_node, current_prompt = self.select_and_expand()
                reward = self.simulation(current_node, current_prompt)
                self.backpropagation(current_node, reward)
            optimum_child = self.nodes[self.root].get_optimum_child(self.policy_weight, self.explore_weight)
            self.stream += self.nodes[self.root].actions[optimum_child]
            for i in range(self.k):
                if i != optimum_child and self.nodes[self.root].child[i] != -1:
                    self.delete_tree(self.nodes[self.root].child[i])
            new_root = self.nodes[self.root].child[optimum_child]
            self.delete_node(self.root)
            self.set_root(new_root)
        optimum_child = self.nodes[self.root].get_optimum_child(self.policy_weight, self.explore_weight)
        respond = self.nodes[self.root].actions[optimum_child]
        self.stream += respond + '<|eot_id|>'
        return respond


model_path = "llama-3.2-1b-instruct-q4_k_m.gguf"
mcts_llm = MCTS_REASONING_LLM(model_path, k=3, policy_weight=1, explore_weight=1)
query = "What is the capital of France?"
response = mcts_llm.query(query, iterations=100)
print("Response:", response)