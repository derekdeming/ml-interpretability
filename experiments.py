# %%
import os
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformer_lens import HookedTransformer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_grad_enabled(False)
os.environ['TRANSFORMERS_CACHE'] = 'tinyMech/synth_word_embeddings/cache/'

model = HookedTransformer.from_pretrained("gpt2-small")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# %%
# model configs
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
d_vocab = model.cfg.d_vocab

# %%
# preprocess words
common_words = open("word_embeddings.txt", "r").read().split("\n")
num_tokens = [len(model.to_tokens(" " + word, prepend_bos=False).squeeze(0)) for word in common_words]

word_df = pd.DataFrame({"word": common_words, "num_tokens": num_tokens})
word_df = word_df.query('num_tokens < 4').reset_index(drop=True)

# train and test split while ensuring balance across absolute positions
train_word_df = word_df.sample(frac=0.8, random_state=SEED)
test_word_df = word_df.drop(train_word_df.index).reset_index(drop=True)
train_word_df = train_word_df.reset_index(drop=True)

MAX_WORD_LENGTH = 3
NUM_WORDS = 7  # num of words to append
PREFIX_LENGTH = 19  # fixed prefix length

train_word_by_length = {
    length: train_word_df[train_word_df.num_tokens == length]['word'].values
    for length in range(1, MAX_WORD_LENGTH + 1)
}

test_word_by_length = {
    length: test_word_df[test_word_df.num_tokens == length]['word'].values
    for length in range(1, MAX_WORD_LENGTH + 1)
}

# %%
def gen_batch(batch_size, word_by_length, prefixes):
    words = []
    tokens_list = []
    first_token_indices = []
    last_token_indices = []
    word_lengths_list = []

    for _ in range(batch_size):
        prefix = np.random.choice(prefixes)
        word_lengths = torch.randint(1, MAX_WORD_LENGTH + 1, (NUM_WORDS,))
        sentence = []
        idx = model.to_tokens(prefix, prepend_bos=True).shape[1]         # use the length of the tokenized prefix without squeezing

        first_indices = []
        last_indices = []

        for word_len in word_lengths:
            word_list = word_by_length[word_len.item()]
            word = " " + np.random.choice(word_list)
            # we're not squeezing the tensor
            word_tokens = model.to_tokens(word, prepend_bos=False)
            sentence.append(word)
            first_indices.append(idx)
            # will use the shape to get the leng of the tokens
            idx += word_tokens.shape[1]
            last_indices.append(idx - 1)

        full_sentence = prefix + "".join(sentence)
        # Do not squeeze the tokens
        tokens = model.to_tokens(full_sentence, prepend_bos=True)
        words.append(full_sentence)
        tokens_list.append(tokens)
        first_token_indices.append(first_indices)
        last_token_indices.append(last_indices)
        word_lengths_list.append(word_lengths)

    max_length = max(t.size(1) for t in tokens_list)
    tokens_padded = torch.zeros((batch_size, max_length), dtype=torch.long)
    for i, t in enumerate(tokens_list):
        length = t.size(1)
        tokens_padded[i, :length] = t[0, :length]

    return tokens_padded.to(device), words, word_lengths_list, first_token_indices, last_token_indices


# %%
# create data for probing
def collect_residuals(model, word_by_length, prefixes, num_batches, batch_size):
    all_residuals = []
    all_labels = []
    for _ in tqdm.tqdm(range(num_batches)):
        tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(
            batch_size, word_by_length, prefixes
        )
        _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith('resid_post'))
        residuals = cache['resid_post', LAYER]
        batch_size = tokens.size(0)
        for i in range(batch_size):
            for j in range(NUM_WORDS):
                idx = last_token_indices[i][j]
                if idx < residuals.size(1):
                    resid = residuals[i, idx].cpu().numpy()
                    all_residuals.append(resid)
                    all_labels.append(j)
    return np.array(all_residuals), np.array(all_labels)


# %%
# this will be the layer to analyze (can change) and 
# we look at all layers later on 
LAYER = 3

# prefix variations to control for prefix length
prefixes = [
    "The quick brown fox jumps over the lazy dog.",
    "In a galaxy far, far away, there was a small planet.",
    "Once upon a time, in a land called",
    "Data science is an interdisciplinary field.",
    "Artificial intelligence and machine learning are",
]

# load in residuals with varying prefixes
X_train_list = []
y_train_list = []

X_resid, y_labels = collect_residuals(
    model, train_word_by_length, prefixes, num_batches=50, batch_size=64
)
X_train_list.append(X_resid)
y_train_list.append(y_labels)

X_train = np.concatenate(X_train_list)
y_train = np.concatenate(y_train_list)


# %%
# train logistic regression probe
from sklearn.multiclass import OneVsRestClassifier

lr_model = OneVsRestClassifier(
    LogisticRegression(
        solver='saga', random_state=SEED, max_iter=1000, C=1.0
    )
)
lr_model.fit(X_train, y_train)

# %%
# check test data with varying prefixes
X_test_list = []
y_test_list = []

X_resid_test, y_labels_test = collect_residuals(
    model, test_word_by_length, prefixes, num_batches=25, batch_size=64
)
X_test_list.append(X_resid_test)
y_test_list.append(y_labels_test)

X_test = np.concatenate(X_test_list)
y_test = np.concatenate(y_test_list)


# %%
# pred + evals using log regression probe
y_pred_train = lr_model.predict(X_train)
print("Training Set Performance:")
print(classification_report(y_train, y_pred_train))

y_pred_test = lr_model.predict(X_test)
print("Test Set Performance:")
print(classification_report(y_test, y_pred_test))

# we can train a neural network probe as well
# from sklearn.neural_network import MLPClassifier

# nn_model = MLPClassifier(hidden_layer_sizes=(512,), activation='relu', max_iter=500, random_state=SEED)
# nn_model.fit(X_train, y_train)

# y_pred_train_nn = nn_model.predict(X_train)
# print("Training Set Performance with NN:")
# print(classification_report(y_train, y_pred_train_nn))

# y_pred_test_nn = nn_model.predict(X_test)
# print("Test Set Performance with NN:")
# print(classification_report(y_test, y_pred_test_nn))



# %%
# look at attention patterns
def analyze_attention_patterns(model, tokens, layer, head):
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith('attn_scores'))
    attn_scores = cache['attn_scores', layer][:, head]
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return attn_probs.cpu().numpy()

# single batch
tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(
    batch_size=1, word_by_length=test_word_by_length, prefixes=[prefixes[0]]
)
attn_probs = analyze_attention_patterns(model, tokens, layer=LAYER, head=0)

# %%
# look at the attention probabilities
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attn_probs, token_idx):
    sns.heatmap(attn_probs[0, token_idx].reshape(1, -1), cmap='viridis')
    plt.title(f'Attention Probabilities for Token Index {token_idx}')
    plt.xlabel('Source Token Position')
    plt.ylabel('')
    plt.show()

plot_attention(attn_probs, token_idx=last_token_indices[0][0])


# %%
# dimensionality reduction visuals
from sklearn.decomposition import PCA

def visualize_residuals(X, y, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Word Index")
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

visualize_residuals(X_test, y_test, 'PCA of Residuals at Layer 3')

# %%
# check other layers
for layer in range(n_layers):
    print(f"Analyzing Layer {layer}")
    X_train_layer = []
    y_train_layer = []

    X_resid, y_labels = collect_residuals(
        model, train_word_by_length, prefixes, num_batches=25, batch_size=64
    )
    X_train_layer.append(X_resid)
    y_train_layer.append(y_labels)

    X_train_layer = np.concatenate(X_train_layer)
    y_train_layer = np.concatenate(y_train_layer)

    lr_model_layer = LogisticRegression(
        multi_class='ovr', solver='saga', random_state=SEED, max_iter=1000, C=1.0
    )
    lr_model_layer.fit(X_train_layer, y_train_layer)
    y_pred_layer = lr_model_layer.predict(X_train_layer)
    print(classification_report(y_train_layer, y_pred_layer))


def analyze_all_attention_patterns(model, tokens, layer):
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name.endswith('attn_probs'))
    attn_probs = cache['attn_probs', layer]  # Shape: [batch, head, dest_pos, src_pos]
    return attn_probs.cpu().numpy()

# Analyze for all heads
attn_probs_all_heads = analyze_all_attention_patterns(model, tokens, layer=LAYER)

# Visualize attention patterns for all heads
def plot_attention_heads(attn_probs, token_idx):
    num_heads = attn_probs.shape[1]
    fig, axes = plt.subplots(nrows=num_heads // 4, ncols=4, figsize=(20, num_heads * 2))
    for head in range(num_heads):
        row = head // 4
        col = head % 4
        sns.heatmap(attn_probs[0, head, token_idx].reshape(1, -1), cmap='viridis', ax=axes[row, col])
        axes[row, col].set_title(f'Head {head}')
        axes[row, col].set_xlabel('Source Token Position')
        axes[row, col].set_ylabel('')
    plt.tight_layout()
    plt.show()

plot_attention_heads(attn_probs_all_heads, token_idx=last_token_indices[0][0])

# %%
import joblib

joblib.dump(lr_model, 'word_position_probe_layer3.pkl')

# %%
# Analyze attention patterns for all heads in a specific layer
layer_to_analyze = 3  # or any layer of interest
tokens, words, word_lengths, first_token_indices, last_token_indices = gen_batch(
    batch_size=1, word_by_length=test_word_by_length, prefixes=[prefixes[0]]
)
attn_probs_all_heads = analyze_all_attention_patterns(model, tokens, layer=layer_to_analyze)

# Compute attention to specific positions
def compute_attention_to_positions(attn_probs, target_positions):
    # attn_probs shape: [batch, head, dest_pos, src_pos]
    attention_scores = attn_probs[:, :, :, target_positions].sum(axis=-1)
    return attention_scores

# compute attention to the first token of each word
target_positions = [idx[0] for idx in first_token_indices]
attention_scores = compute_attention_to_positions(attn_probs_all_heads, target_positions)

# attention scores
num_heads = attn_probs_all_heads.shape[1]
plt.figure(figsize=(12, 6))
for head in range(num_heads):
    plt.plot(attention_scores[0, head], label=f'Head {head}')
plt.title(f'Attention to First Tokens in Layer {layer_to_analyze}')
plt.xlabel('Destination Position')
plt.ylabel('Attention Score')
plt.legend()
plt.show()

# %%
# Collect probe accuracies across layers
probe_accuracies = []

for layer in range(n_layers):
    print(f"Analyzing Layer {layer}")
    X_train_layer, y_train_layer = collect_residuals(
        model, train_word_by_length, prefixes, num_batches=25, batch_size=64
    )
    
    lr_model_layer = LogisticRegression(
        solver='saga', random_state=SEED, max_iter=1000, C=1.0
    )
    lr_model_layer.fit(X_train_layer, y_train_layer)
    
    y_pred_layer = lr_model_layer.predict(X_train_layer)
    accuracy = np.mean(y_pred_layer == y_train_layer)
    probe_accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(range(n_layers), probe_accuracies, marker='o')
plt.title('Probe Accuracy Across Layers')
plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# %%
# ablate specific attention heads
def ablate_attention_heads(model, layers_heads_to_ablate):
    def ablation_hook(module, input, output):
        for layer, heads in layers_heads_to_ablate.items():
            if module.layer == layer:
                output[:, heads, :, :] = 0
        return output
    return ablation_hook

# ablate head 0 in layer 3
layers_heads_to_ablate = {3: [0]}
model.reset_hooks()
model.add_hook('blocks.attn_scores', ablate_attention_heads(model, layers_heads_to_ablate))

# Re-run the probe evaluation with the ablated model
X_resid_ablate, y_labels_ablate = collect_residuals(
    model, test_word_by_length, prefixes, num_batches=25, batch_size=64
)
y_pred_ablate = lr_model.predict(X_resid_ablate)
print("Test Set Performance with Ablated Model:")
print(classification_report(y_labels_ablate, y_pred_ablate))

# %%
# look at the high-dimensional representations with t-SNE:
from sklearn.manifold import TSNE

def visualize_residuals_tsne(X, y, title):
    tsne = TSNE(n_components=2, random_state=SEED)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Word Index")
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

visualize_residuals_tsne(X_test, y_test, 't-SNE of Residuals at Layer 3')
