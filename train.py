import json
import time
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from config import *

class MarkovModel():
    def __init__(self, N) -> None:
        self.maxN = N
        self.models = self.load_saved_models(save_path=SAVE_PATH)
        self.normalised_transition_matrix = {}
        if self.models:
            self._build_probabilities()

    def _build_probabilities(self):
        self.normalised_transition_matrix = {}
        for n, data in self.models.items():
            counts = data["counts"]
            row_totals = counts.sum(axis=1)
            probs = counts.div(row_totals.replace(0, np.nan), axis=0).fillna(0).astype(np.float32)
            self.normalised_transition_matrix[n] = probs

    def train(self, training_data):
        if self.models: return
        for n in range(1, self.maxN + 1):
            if len(training_data) <= n:
                continue
            counts = defaultdict(lambda: defaultdict(int))
            for i in range(len(training_data) - n):
                context = training_data[i:i+n]
                target = training_data[i+n]
                counts[context][target] += 1
            transition_matrix = pd.DataFrame.from_dict(counts,orient='index')
            transition_matrix = transition_matrix.reindex(columns=list(ALPHABET)).fillna(0).astype("int32")
            transition_matrix.index.name = "Context"
            row_totals = transition_matrix.sum(axis=1)
            normalised_transition_matrix = transition_matrix.div(row_totals, axis=0).astype(np.float32)
            self.models[n] = { "counts": transition_matrix }
            self.normalised_transition_matrix = normalised_transition_matrix
            print(f"N={n:2d}: "f"contexts={len(counts):,}, ")
        self._build_probabilities()

    def get_longest_context(self, context):
        for n in range(min(len(context), self.maxN), 0, -1):
            shorter_ctx = context[-n:]
            if (n in self.models and
                n in self.normalised_transition_matrix and
                shorter_ctx in self.normalised_transition_matrix[n].index):
                return n, shorter_ctx
        return 0, None

    def generate(self, maxlen, prompt, p=0.9, temperature=0.7):
        prompt = prompt.lower()
        n, valid_context = self.get_longest_context(prompt)
        if not valid_context:
            print("Minimum context window reached, no match found")
            return None
        generated = list(prompt)
        prompt = valid_context

        for _ in range(maxlen):
            current_n, current_context = self.get_longest_context(prompt)
            if not current_context:
                # Fall back to the most next longest context model available if context is entirely unseen
                longest_model = self.models[max(self.models.keys())]["counts"].sum(axis=0)
                next_char = self.sample_top_p(longest_model, p=p, temperature=temperature)
            else:
                count_series = self.models[current_n]["counts"].loc[current_context]
                next_char = self.sample_top_p(count_series, p, temperature)
            if next_char is None or next_char==EOS:
                break

            generated.append(next_char)
            prompt = prompt + next_char

        text = "".join(generated).rstrip(EOS)
        print('. '.join(sentence.strip().capitalize() for sentence in text.split('. ')))
        print(f"{len(text)} characters")
        timestamp = int(time.time())
        with open(f"outputs/generated_text{timestamp}.txt", "w") as f:
            f.write('. '.join(sentence.strip().capitalize() for sentence in text.split('. ')))
        print(f"Saved to generated_text{timestamp}.txt")
        return text

    def sample_top_p(self, counts_series, p=0.9, temperature=0.7):
        valid_counts = counts_series[counts_series > 0]
        if valid_counts.empty:
            return None
        logits = np.log(valid_counts.values.astype(np.float32))
        logits /= temperature
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        order = np.argsort(probabilities)[::-1]
        sorted_chars = valid_counts.index[order]
        sorted_probs = probabilities[order]
        cumulative = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative, p, side="left")
        candidate_chars = sorted_chars[:cutoff + 1]
        candidate_probs = sorted_probs[:cutoff + 1]
        candidate_probs /= candidate_probs.sum()
        return np.random.choice(candidate_chars, p=candidate_probs)

    def display_matrix(self):
        print("\nSTATE TRANSITION MATRIX")
        print(self.models[self.maxN]["counts"])

    def save_models(self, save_path):
        print("Saving model")
        pickle.dump(self.models, open(save_path, 'wb'))

    def load_saved_models(self, save_path):
        try:
            saved_models = pickle.load(open(save_path, 'rb'))
            print("Loading saved model")
        except:
            saved_models = {}
            print("Model not found")
        return saved_models

def get_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    training_data = ' '.join(v+EOS for _,v in data.items())
    return training_data

if __name__=="__main__":
    model = MarkovModel(N=N)
    test_input = "The secret"
    training_data = get_data(CORPUS)
    model.train(training_data)
    model.display_matrix()
    model.generate(4000, test_input, 0.6, 0.8)
    #model.save_models(SAVE_PATH)